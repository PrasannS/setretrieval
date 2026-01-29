from __future__ import annotations

import os
import copy
from typing import Literal

import numpy as np
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import sentence_transformers.losses as slosses
import sentence_transformers.evaluation as sevaluation
from sentence_transformers.util import batch_to_device
from tqdm.autonotebook import trange
from peft import LoraConfig

from pylate import evaluation, losses, models, utils
from pylate.losses.contrastive import Contrastive
from pylate.scores import colbert_scores
from pylate.utils import all_gather, all_gather_with_gradients, get_rank, get_world_size
from pylate.utils.tensor import convert_to_tensor
from pylate.models.colbert import ColBERT

from setretrieval.eval.maxmax_evaluator import MaxMaxTripletEvaluator
from setretrieval.train.scores import (
    route_colbscores_multiquery,
    route_colbscores_multipos,
    maxmax_scores,
)
from setretrieval.train.pylate_monkeypatch import (
    padded_tokenize,
    newforward,
    mod_encode,
    modadj_tokenize,
)
from setretrieval.train.losses import SetContrastive

# -----------------------------------------------------------------------------
# ColBERT monkey patching (must run before creating ColBERT model)
# -----------------------------------------------------------------------------


def do_monkey_patching(dodefaulttrain: str = "no") -> None:
    if dodefaulttrain == "no":
        models.ColBERT.tokenize = padded_tokenize
        models.ColBERT.old_forward = models.ColBERT.forward
        models.ColBERT.forward = newforward
        models.ColBERT.old_encode = models.ColBERT.encode
        models.ColBERT.encode = mod_encode
    models.ColBERT.old_tokenize = models.ColBERT.tokenize
    models.ColBERT.tokenize = modadj_tokenize


# -----------------------------------------------------------------------------
# Dataset preprocessing
# -----------------------------------------------------------------------------


def _preprocess_datasets(
    train_dataset,
    eval_dataset,
    model_type: Literal["sbert", "colbert"],
    maxchars: int = 5000,
):
    """Truncate positive/negative to maxchars for ColBERT; no-op for SBERT."""
    if model_type != "colbert":
        return train_dataset, eval_dataset
    truncate = lambda x: {
        **x,
        "positive": x["positive"][:maxchars],
        "negative": x["negative"][:maxchars],
    }
    return (
        train_dataset.map(truncate),
        eval_dataset.map(truncate),
    )


# -----------------------------------------------------------------------------
# Run name and output directory
# -----------------------------------------------------------------------------


def _build_run_name(
    model_type: Literal["sbert", "colbert"],
    base_model: str,
    per_device_batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    dsetname: str,
    **colbert_run_kw,
) -> str:
    base = (
        f"contrastive-{base_model.replace('/', '_')}-bs{per_device_batch_size}"
        f"-e{num_train_epochs}-lr{learning_rate}-{dsetname}"
    )
    if model_type == "sbert":
        return f"{base}-sbert"
    # ColBERT: append colscore, div, sched, etc.
    colscore = colbert_run_kw.get("colscore", "maxmax")
    div_coeff = colbert_run_kw.get("div_coeff", 0.0)
    divq_coeff = colbert_run_kw.get("divq_coeff", 0.0)
    schedtype = colbert_run_kw.get("schedtype", "constant")
    temp = colbert_run_kw.get("temp", 0.02)
    othermod = colbert_run_kw.get("othermod", "neither")
    dodefaulttrain = colbert_run_kw.get("dodefaulttrain", "no")
    embsize = colbert_run_kw.get("embsize", 128)
    qvecs = colbert_run_kw.get("qvecs", 5)
    dvecs = colbert_run_kw.get("dvecs", 5)
    run_name = (
        f"{base}-{colscore}-divd{div_coeff}-divq{divq_coeff}-qv{qvecs}-dv{dvecs}"
        f"-{schedtype}-temp{temp}-omod{othermod}-dodefaulttrain{dodefaulttrain}-embsize{embsize}"
    )
    if colbert_run_kw.get("lora_rank", -1) > 0:
        run_name += f"-lora{colbert_run_kw['lora_rank']}"
    return run_name


def _get_output_dir(model_type: Literal["sbert", "colbert"], run_name: str) -> str:
    subdir = "sbert_training" if model_type == "sbert" else "colbert_training"
    path = f"propercache/cache/{subdir}/{run_name}"
    os.makedirs(path, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Model creation
# -----------------------------------------------------------------------------


def _create_sbert_model(base_model: str, **kwargs):
    return SentenceTransformer(base_model)


def _create_colbert_model(
    base_model: str,
    qvecs: int = 5,
    dvecs: int = 5,
    embsize: int = 128,
    dodefaulttrain: str = "no",
    lora_rank: int = -1,
    do_compile: bool | str = True,
    **kwargs,
):
    do_monkey_patching(dodefaulttrain)
    torch.set_float32_matmul_precision("high")

    lora_config = None
    if lora_rank > 0:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            bias="none",
        )

    model = models.ColBERT(
        model_name_or_path=base_model,
        query_length=qvecs,
        document_length=dvecs,
        do_query_expansion=(dodefaulttrain == "yes"),
        embedding_size=embsize,
    )
    if lora_config is not None:
        model.add_adapter(lora_config)
    if dodefaulttrain == "no":
        model.tokenizer.query_vectors = qvecs
        model.tokenizer.doc_vectors = dvecs

    if do_compile in (True, "yes"):
        model = torch.compile(model)
    return model


# -----------------------------------------------------------------------------
# Loss creation
# -----------------------------------------------------------------------------


def _create_sbert_loss(model, **kwargs):
    return slosses.MultipleNegativesRankingLoss(model=model, scale=50)


def _create_colbert_loss(
    model,
    colscore: str = "maxmax",
    temp: float = 0.02,
    div_coeff: float = 0.0,
    divq_coeff: float = 0.0,
    othermodel_name: str = "google-bert/bert-large-uncased",
    othermod: str = "neither",
    **kwargs,
):
    score_map = {
        "maxmax": maxmax_scores,
        "multipos": route_colbscores_multipos,
        "multiquery": route_colbscores_multiquery,
        "maxsim": colbert_scores,
    }
    metric = score_map.get(colscore, maxmax_scores)
    return SetContrastive(
        model=model,
        temperature=temp,
        score_metric=metric,
        div_coeff=div_coeff,
        divq_coeff=divq_coeff,
        othermodel_name=othermodel_name,
        othermod=othermod,
    )


# -----------------------------------------------------------------------------
# Evaluator creation
# -----------------------------------------------------------------------------


def _create_sbert_evaluator(eval_dataset, **kwargs):
    return sevaluation.TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="dev",
    )


def _create_colbert_evaluator(eval_dataset, colscore: str = "maxmax", **kwargs):
    if colscore in ("maxmax", "multipos"):
        return MaxMaxTripletEvaluator(
            anchors=eval_dataset["query"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
        )
    return evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )


# -----------------------------------------------------------------------------
# Training arguments
# -----------------------------------------------------------------------------


def _build_training_args(
    model_type: Literal["sbert", "colbert"],
    output_dir: str,
    run_name: str,
    num_train_epochs: int,
    per_device_batch_size: int,
    learning_rate: float,
    mini_batch_size: int | None = None,
    save_strat: str = "epoch",
    schedtype: str = "constant",
    **kwargs,
) -> SentenceTransformerTrainingArguments:
    per_device_eval_batch_size = mini_batch_size if model_type == "colbert" else per_device_batch_size
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        bf16=True,
        run_name=run_name,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        lr_scheduler_type=schedtype,
        save_only_model=True,
        save_strategy=save_strat,
        logging_steps=1,
    )


# main thing to handle training stuff
def train_retriever(
    train_dataset,
    eval_dataset,
    model_type: Literal["sbert", "colbert"],
    base_model: str = "google-bert/bert-base-uncased",
    per_device_batch_size: int = 32,
    num_train_epochs: int = 3,
    learning_rate: float = 3e-6,
    dsetname: str = "gemini_datav1",
    # ColBERT-only (ignored for sbert)
    mini_batch_size: int = 32,
    maxchars: int = 5000,
    div_coeff: float = 0.0,
    divq_coeff: float = 0.0,
    colscore: str = "maxmax",
    save_strat: str = "epoch",
    schedtype: str = "constant",
    temp: float = 0.02,
    othermodel_name: str = "google-bert/bert-large-uncased",
    othermod: str = "neither",
    qvecs: int = 5,
    dvecs: int = 5,
    dodefaulttrain: str = "no",
    do_compile: bool | str = True,
    lora_rank: int = -1,
    embsize: int = 128,
    **kwargs,
):
    """
    Train a retriever (SBERT or ColBERT). Model, loss, and evaluator are
    created via model-type-specific helpers; the rest of the loop is shared.
    """
    train_dataset, eval_dataset = _preprocess_datasets(
        train_dataset, eval_dataset, model_type, maxchars
    )

    colbert_run_kw = {
        "colscore": colscore,
        "div_coeff": div_coeff,
        "divq_coeff": divq_coeff,
        "schedtype": schedtype,
        "temp": temp,
        "othermod": othermod,
        "dodefaulttrain": dodefaulttrain,
        "embsize": embsize,
        "qvecs": qvecs,
        "dvecs": dvecs,
        "lora_rank": lora_rank,
    }
    run_name = _build_run_name(
        model_type,
        base_model,
        per_device_batch_size,
        num_train_epochs,
        learning_rate,
        dsetname,
        **colbert_run_kw,
    )
    output_dir = _get_output_dir(model_type, run_name)

    if model_type == "sbert":
        model = _create_sbert_model(base_model)
        train_loss = _create_sbert_loss(model)
        evaluator = _create_sbert_evaluator(eval_dataset)
    else:
        model = _create_colbert_model(
            base_model,
            qvecs=qvecs,
            dvecs=dvecs,
            embsize=embsize,
            dodefaulttrain=dodefaulttrain,
            lora_rank=lora_rank,
            do_compile=do_compile,
        )
        train_loss = _create_colbert_loss(
            model,
            colscore=colscore,
            temp=temp,
            div_coeff=div_coeff,
            divq_coeff=divq_coeff,
            othermodel_name=othermodel_name,
            othermod=othermod,
        )
        evaluator = _create_colbert_evaluator(eval_dataset, colscore=colscore)

    args = _build_training_args(
        model_type,
        output_dir,
        run_name,
        num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        mini_batch_size=mini_batch_size,
        save_strat=save_strat,
        schedtype=schedtype,
    )

    trainer_kw: dict = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "loss": train_loss,
        "evaluator": evaluator,
    }
    if model_type == "colbert":
        trainer_kw["data_collator"] = utils.ColBERTCollator(model.tokenize)

    print(
        f"Training on {len(train_dataset)} examples, "
        f"evaluating on {len(eval_dataset)} examples."
    )
    trainer = SentenceTransformerTrainer(**trainer_kw)
    trainer.train()
    trainer.save_model(output_dir)

    if model_type == "sbert":
        return model

