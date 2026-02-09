# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import argparse

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from pylate import models, utils, losses, evaluation
import torch
from setretrieval.train.pylate_monkeypatch import padded_tokenize, newforward, modadj_tokenize, mod_encode
from setretrieval.train.scores import mod_colbert_scores, extend_vector_scores

def main(args):
    lr = args.lr
    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]

    # 1. Load pylate model to finetune
    model = models.ColBERT(
        model_name_or_path=model_name, 
        model_kwargs={'dtype': torch.bfloat16}, 
        embedding_size=args.embdim
    )
    if args.qvecs != -1 and args.dvecs != -1:
        model.tokenizer.query_vectors = args.qvecs
        model.tokenizer.doc_vectors = args.dvecs

    # 2. Load a dataset to finetune on
    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(1_250_000))
    # train_dataset = dataset_dict["train"].select(range(100_000))

    eval_dataset = dataset_dict["test"]

    # 3. Define a loss function
    if args.qvecs != -1 and args.dvecs != -1:
        scmet = mod_colbert_scores if args.colscore == "maxsim" else extend_vector_scores
        print(f"Using {args.colscore} score metric")
        loss = losses.CachedContrastive(model, mini_batch_size=16, temperature=0.05, score_metric=scmet)  # Increase mini_batch_size if you have enough VRAM
    else:
        loss = losses.CachedContrastive(model, mini_batch_size=16, temperature=0.05)  # Increase mini_batch_size if you have enough VRAM
    run_name = f"{model_shortname}-pylate-pairwise-{lr}-qv{args.qvecs}-dv{args.dvecs}-embsize{args.embdim}"
    if args.colscore != "maxsim":
        run_name += f"-{args.colscore}"
    # 4. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"output/{model_shortname}/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        warmup_ratio=0.05,
        fp16=False,  # Set to False if GPU can't handle FP16
        bf16=True,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicates
        learning_rate=lr,
        # Optional tracking/debugging parameters:
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        logging_steps=2,
        run_name=run_name,  # Used in `wandb`, `tensorboard`, `neptune`, etc. if installed
        dataloader_num_workers=8,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
    )

    # 5. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize)
    )
    trainer.train()

    # 7. (Optional) Evaluate the trained model on the evaluator after training
    dev_evaluator(model)

    # 8. Save the model
    model.save_pretrained(f"output/{model_shortname}/{run_name}/final")

    # model.push_to_hub(run_name, private=False)

if __name__ == "__main__":

    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--embdim", type=int, default=128)
    parser.add_argument("--qvecs", type=int, default=-1)
    parser.add_argument("--dvecs", type=int, default=-1)
    parser.add_argument("--colscore", type=str, default="maxsim")
    args = parser.parse_args()
    if args.qvecs != -1 and args.dvecs != -1:
        print("Monkey patching ColBERT model")
        # do necessary monkey patching
        models.ColBERT.tokenize = padded_tokenize
        models.ColBERT.old_forward = models.ColBERT.forward
        models.ColBERT.forward = newforward
        models.ColBERT.old_encode = models.ColBERT.encode
        models.ColBERT.encode = mod_encode
        models.ColBERT.old_tokenize = models.ColBERT.tokenize
        models.ColBERT.tokenize = modadj_tokenize
    main(args)