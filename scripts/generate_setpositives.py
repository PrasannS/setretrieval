# script to generate set evaluation data
from setretrieval.datagen.generate_setdata import hierarchical_positive_search
import argparse
import os
from datasets import DatasetDict, Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/datastores/wikipedia_docs_15k")
    parser.add_argument("--starter_question_set", type=str, default="data/colbert_training/gemini_ntrain_ptest")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    dataset = Dataset.load_from_disk(args.dataset_path)
    # try to take another node...
    question_set = DatasetDict.load_from_disk(args.starter_question_set)['train'].select(range(5000, 10000))

    selectresults = hierarchical_positive_search(dataset["text"], question_set["query"], "passagesearchtrain_v1_5k", models=["Qwen/Qwen3-4B"])