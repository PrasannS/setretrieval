# Diffusion-style multi-forward-pass ColBERT training script.
#
# Based on scripts/train_pylate_pairwise.py.
# Key additions:
#   --num_passes      : number of document forward passes (M)
#   --backprop_all    : allow gradient through all passes (default True)
#   --per_step_loss   : compute contrastive loss at every pass step
#
# The model is a standard pylate ColBERT with padded_tokenize applied so that
# documents contain --dvecs extra [EMB] tokens.  These are the "pad tokens"
# whose hidden states are recycled between passes.
#
# Example (4 GPUs, 8 pad tokens, 3 passes):
#   torchrun --nproc_per_node=4 scripts/train_diffusion_colbert.py \
#       --dvecs 8 --num_passes 3 --backprop_all --mini_batch_size 16

import argparse
import random

import torch
from datasets import Dataset, DatasetDict, load_dataset
from pylate import evaluation, losses, models, utils

from setretrieval.train.diffusion_colbert import DiffusionCachedContrastive
from setretrieval.train.pylate_monkeypatch import (
    mod_encode,
    modadj_tokenize,
    newforward,
    padded_tokenize,
)
from setretrieval.train.scores import (
    colbert_scores_topk,
    extend_vector_scores,
    mod_colbert_scores,
    mod_colbert_scores_topk,
)
from pylate.scores import colbert_scores

from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments


def make_toy_dataset(n=2000, vocab_size=64, doc_len=30, seed=42):
    """Single-word query retrieval toy task (same as train_autoreg_colbert.py)."""
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab_size)]
    rows = []
    for _ in range(n):
        q = rng.choice(words)
        rest = [w for w in words if w != q]
        pos = rng.choices(rest, k=doc_len)
        pos.insert(rng.randint(0, doc_len), q)
        neg = rng.choices(rest, k=doc_len + 1)
        rows.append({"query": q, "positive": " ".join(pos), "negative": " ".join(neg)})
    return Dataset.from_list(rows)


def main(args):
    lr = args.lr
    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]

    if args.dataset == "reasonir":
        document_length = 8192
        query_length = 128
    elif args.dataset == "toy":
        document_length = 64
        query_length = 16
    else:
        document_length = 2000
        query_length = 32

    # 1. Load pylate ColBERT model
    model = models.ColBERT(
        model_name_or_path=model_name,
        model_kwargs={"dtype": torch.bfloat16},
        embedding_size=args.embdim,
        document_length=document_length,
        query_length=query_length,
        skiplist_words=[],
    )

    model.tokenizer.query_vectors = args.qvecs
    model.tokenizer.doc_vectors = args.dvecs
    model.tokenizer.qpass_vecs = args.passiveqvecs
    model.tokenizer.dpass_vecs = args.passivedvecs

    model.tokenizer.query_ratio = args.qratio
    model.tokenizer.document_ratio = args.dratio

    # 2. Load dataset
    if args.dataset == "toy":
        train_dataset = make_toy_dataset(n=1800, vocab_size=64, doc_len=30, seed=42)
        eval_dataset  = make_toy_dataset(n=200,  vocab_size=64, doc_len=30, seed=99)

    elif args.dataset == "msmarco":
        dataset = load_dataset(
            "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
            "triplet-hard",
            split="train",
        )
        dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
        train_dataset = dataset_dict["train"].select(range(1_250_000))
        eval_dataset = dataset_dict["test"]

    elif args.dataset == "reasonir":
        dataset = DatasetDict.load_from_disk("propercache/data/colbert_training/reasonir_hq_formatted")
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    oldlen = len(train_dataset)

    if args.debug:
        print("Debug mode: using 10k samples")
        train_dataset = train_dataset.select(range(10000))
        eval_dataset = eval_dataset.select(range(512))

    if args.dataset != "toy":
        train_dataset = train_dataset.filter(lambda x: len(x["query"]) < 8000)
    print(f"Old length: {oldlen}  New length: {len(train_dataset)}")

    # 3. Choose score metric
    if args.qvecs != -1 and args.dvecs != -1:
        scmet = mod_colbert_scores if args.colscore == "maxsim" else extend_vector_scores
        if args.topk > 1:
            print(f"Using topk score metric with k={args.topk} and alpha={args.alpha}")
            scmet = lambda a, b, **kwargs: mod_colbert_scores_topk(a, b, k=args.topk, alpha=args.alpha)
    else:
        scmet = colbert_scores
        if args.topk > 1:
            print(f"Using topk score metric with k={args.topk} and alpha={args.alpha}")
            scmet = lambda a, b, **kwargs: colbert_scores_topk(a, b, k=args.topk, alpha=args.alpha)

    # 4. Build the diffusion loss
    loss = DiffusionCachedContrastive(
        model,
        num_passes=args.num_passes,
        backprop_all_passes=args.backprop_all,
        per_step_loss=args.per_step_loss,
        mini_batch_size=args.mini_batch_size,
        temperature=args.temperature,
        score_metric=scmet,
    )
    print(
        f"DiffusionCachedContrastive: num_passes={args.num_passes}, "
        f"backprop_all_passes={args.backprop_all}, "
        f"per_step_loss={args.per_step_loss}"
    )

    # 5. Build run name
    run_name = (
        f"{model_shortname}-diffcolbert-{args.dataset}-{lr}"
        f"-qv{args.qvecs}-dv{args.dvecs}"
        f"-pqv{args.passiveqvecs}-pdv{args.passivedvecs}"
        f"-embsize{args.embdim}"
        f"-passes{args.num_passes}"
    )
    if not args.backprop_all:
        run_name += "-nobackpropall"
    if args.per_step_loss:
        run_name += "-persteploss"
    if args.colscore != "maxsim":
        run_name += f"-{args.colscore}"
    if args.topk > 1:
        run_name += f"-topk{args.topk}-alpha{args.alpha}"
    if args.qratio != 0 or args.dratio != 0:
        run_name += f"-qratio{args.qratio}-dratio{args.dratio}"

    # 6. Training arguments
    targs = SentenceTransformerTrainingArguments(
        output_dir=f"output/{model_shortname}/{run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.big_batch_size,
        per_device_eval_batch_size=args.big_batch_size,
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        learning_rate=lr,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        logging_steps=2,
        run_name=run_name,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )

    # 7. Evaluator
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )

    # 8. Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    model.save_pretrained(f"output/{model_shortname}/{run_name}/final")
    dev_evaluator(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--embdim", type=int, default=128)
    parser.add_argument("--qvecs", type=int, default=-1)
    parser.add_argument("--dvecs", type=int, default=8,
                        help="Number of pad [EMB] tokens appended to documents (required > 0 for diffusion).")
    parser.add_argument("--passiveqvecs", type=int, default=0)
    parser.add_argument("--passivedvecs", type=int, default=0)
    parser.add_argument("--colscore", type=str, default="maxsim")
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="msmarco")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--big_batch_size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--qratio", type=float, default=0)
    parser.add_argument("--dratio", type=float, default=0)

    # Diffusion-specific arguments
    parser.add_argument("--num_passes", type=int, default=3,
                        help="Number of forward passes for document encoding.")
    parser.add_argument("--backprop_all", action="store_true", default=True,
                        help="Allow gradients to flow through all passes (default True).")
    parser.add_argument("--no_backprop_all", dest="backprop_all", action="store_false",
                        help="Detach hidden states between passes (only last pass gets gradient).")
    parser.add_argument("--per_step_loss", action="store_true",
                        help="Compute contrastive loss at every forward-pass step (no GradCache).")

    args = parser.parse_args()

    if args.dvecs <= 0 and args.num_passes > 1:
        raise ValueError(
            "--dvecs must be > 0 when using --num_passes > 1. "
            "The pad tokens are what gets refined across passes."
        )

    # Apply monkey-patches (same logic as train_pylate_pairwise.py)
    # padded_tokenize is always needed to add [EMB] tokens to docs/queries.
    # newforward is needed if qvecs > 0 (to extract query CLS tokens).
    if args.qvecs != -1 or args.dvecs != -1 or args.qratio > 0 or args.dratio > 0:
        print("Monkey patching ColBERT model")
        models.ColBERT.tokenize = padded_tokenize
        models.ColBERT.old_forward = models.ColBERT.forward
        models.ColBERT.forward = newforward
        models.ColBERT.old_encode = models.ColBERT.encode
        models.ColBERT.encode = mod_encode
        models.ColBERT.old_tokenize = models.ColBERT.tokenize
        models.ColBERT.tokenize = modadj_tokenize

    main(args)
