"""
Train some different set retrieval approaches, for now let's start with doing ColBERT, will also test single-vector via PyLate.
"""
import setretrieval.train.colbert_train as colbert_train
from datasets import DatasetDict
import argparse
import os

os.environ["WANDB_DIR"] = os.path.abspath("./cache/wandb")


if __name__ == "__main__":
    # arguments for batch size, num epochs, learning rate
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--traintype", type=str, default="colbert")
    args = parser.parse_args()

    dataset = DatasetDict.load_from_disk("data/colbert_training/gemini_datav1")
    traindata = dataset["train"]
    evaldata = dataset["test"]

    if args.traintype == "colbert":
        colbert_train.train_colbert(traindata, evaldata, per_device_batch_size=args.batch_size, num_train_epochs=args.num_epochs, learning_rate=args.learning_rate, base_model=args.model_name)
    elif args.traintype == "sbert":
        colbert_train.train_sbert(traindata, evaldata, per_device_batch_size=args.batch_size, num_train_epochs=args.num_epochs, learning_rate=args.learning_rate, base_model=args.model_name)