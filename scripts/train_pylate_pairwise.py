# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import argparse

from datasets import load_dataset, DatasetDict
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
from setretrieval.train.scores import mod_colbert_scores, extend_vector_scores, mod_colbert_scores_topk, colbert_scores_topk
from pylate.scores import colbert_scores

def main(args):
    lr = args.lr
    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]

    if args.dataset == "reasonir":
        document_length = 8192 # TODO this is unnecessary maybe?
        query_length = 128
    else:
        document_length = 2000
        query_length = 32
    # 1. Load pylate model to finetune
    model = models.ColBERT(
        model_name_or_path=model_name, 
        model_kwargs={'dtype': torch.bfloat16}, 
        embedding_size=args.embdim,
        document_length=document_length, 
        query_length=query_length,
        skiplist_words=[],
    )
    if args.qvecs != -1 and args.dvecs != -1:
        model.tokenizer.query_vectors = args.qvecs
        model.tokenizer.doc_vectors = args.dvecs
        model.tokenizer.qpass_vecs = args.passiveqvecs
        model.tokenizer.dpass_vecs = args.passivedvecs

    # 2. Load a dataset to finetune on
    if args.dataset == "msmarco":
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
        print("Debug mode: using 500 samples")
        train_dataset = train_dataset.select(range(10000))
        eval_dataset = eval_dataset.select(range(512))
        
    train_dataset = train_dataset.filter(lambda x: len(x["query"]) < 8000)
    print("Old length: ", oldlen, "New length: ", len(train_dataset))
    
    # 3. Define a loss function (TODO should be possible to clean this up)
    if args.qvecs != -1 and args.dvecs != -1:
        scmet = mod_colbert_scores if args.colscore == "maxsim" else extend_vector_scores
        if args.topk > 1:
            print(f"Using topk score metric with k={args.topk} and alpha={args.alpha}")
            scmet = lambda a, b, **kwargs: mod_colbert_scores_topk(a, b, k=args.topk, alpha=args.alpha)
        print(f"Using {args.colscore} score metric")
        loss = losses.CachedContrastive(model, mini_batch_size=args.mini_batch_size, temperature=args.temperature, score_metric=scmet)  # Increase mini_batch_size if you have enough VRAM
    else:
        # just use defaults in this case (need masking for normal stuff)
        scmet = colbert_scores
        if args.topk > 1:
            print(f"Using topk score metric with k={args.topk} and alpha={args.alpha}")
            scmet = lambda a, b, **kwargs: colbert_scores_topk(a, b, k=args.topk, alpha=args.alpha, **kwargs)
        loss = losses.CachedContrastive(model, mini_batch_size=args.mini_batch_size, temperature=args.temperature)  # Increase mini_batch_size if you have enough VRAM

    
    run_name = f"{model_shortname}-pylate-pairwise-{args.dataset}-{lr}-qv{args.qvecs}-dv{args.dvecs}-pqv{args.passiveqvecs}-pdv{args.passivedvecs}-embsize{args.embdim}"
    if args.colscore != "maxsim":
        run_name += f"-{args.colscore}"
    if args.topk > 1:
        run_name += f"-topk{args.topk}-alpha{args.alpha}"

    # 4. (Optional) Specify training arguments
    targs = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"output/{model_shortname}/{run_name}",
        # Optional training parameters:
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.big_batch_size,
        per_device_eval_batch_size=args.big_batch_size,
        warmup_ratio=0.05,
        fp16=False,  # Set to False if GPU can't handle FP16
        bf16=True,  # Set to True if GPU supports BF16
        learning_rate=lr,
        # Optional tracking/debugging parameters:
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        logging_steps=2,
        run_name=run_name,  # Used in `wandb`, `tensorboard`, `neptune`, etc. if installed
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )

    # 5. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    # dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize)
    )
    trainer.train()


    # 8. Save the model
    model.save_pretrained(f"output/{model_shortname}/{run_name}/final")

    # 7. (Optional) Evaluate the trained model on the evaluator after training
    dev_evaluator(model)
    
    # model.push_to_hub(run_name, private=False)

if __name__ == "__main__":

    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--embdim", type=int, default=128)
    parser.add_argument("--qvecs", type=int, default=-1)
    parser.add_argument("--dvecs", type=int, default=-1)
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