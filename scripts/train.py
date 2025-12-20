"""
Train some different set retrieval approaches, for now let's start with doing ColBERT, will also test single-vector via PyLate.
"""
import setretrieval.train.colbert_train as colbert_train
from datasets import DatasetDict
import argparse
import os
import wandb

# wandb.init(mode='offline', project='setretrieval')

os.environ["WANDB_DIR"] = os.path.abspath("./propercache/cache/wandb")


if __name__ == "__main__":
    print("Starting training")
    # arguments for batch size, num epochs, learning rate
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--traintype", type=str, default="colbert")
    parser.add_argument("--dataset", type=str, default="gemini_datav1")
    parser.add_argument("--div_coeff", type=float, default=1.0)
    parser.add_argument("--colscore", type=str, default="maxsim")
    parser.add_argument("--querylen", type=int, default=256)
    args = parser.parse_args()

    dataset = DatasetDict.load_from_disk(f"propercache/data/colbert_training/{args.dataset}")
    traindata = dataset["train"]
    evaldata = dataset["test"]

    # breakpoint()

    # only keep query, positive, negative columns
    # traindata = traindata.select(["query", "positive", "negative"])
    # evaldata = evaldata.select(["query", "positive", "negative"])

    if args.traintype == "colbert":
        print("Training ColBERT")
        colbert_train.train_colbert(traindata, evaldata, per_device_batch_size=args.batch_size, num_train_epochs=args.num_epochs, learning_rate=args.learning_rate, base_model=args.model_name, dsetname=args.dataset, div_coeff=args.div_coeff, colscore=args.colscore, querylen=args.querylen)
    elif args.traintype == "sbert":
        print("Training SBERT")
        colbert_train.train_sbert(traindata, evaldata, per_device_batch_size=args.batch_size, num_train_epochs=args.num_epochs, learning_rate=args.learning_rate, base_model=args.model_name, dsetname=args.dataset)