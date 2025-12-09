# script to generate set evaluation data
from setretrieval.datagen.generate_setdata import hierarchical_positive_search
import argparse
import os   
from datasets import DatasetDict, Dataset
from setretrieval.utils.utils import pickload
from tqdm import tqdm
from setretrieval.datagen.generate_setdata import chunks_to_inds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/datastores/wikipedia_docs_15k")
    parser.add_argument("--starter_question_set", type=str, default="data/colbert_training/gemini_ntrain_ptest")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    dataset = Dataset.load_from_disk(args.dataset_path)
    # try to take another node...
    question_set = DatasetDict.load_from_disk(args.starter_question_set)['train'].select(range(5200, 10000))

    if False:
        selectresults = hierarchical_positive_search(dataset["text"], question_set["query"], "passagesearchtrain_v1_5.2k", models=["Qwen/Qwen3-4B"])
    else:
        tmpres = pickload("cache/gendata/passagesearchtrain_v1_5.2k_0.pkl")
        # tmpres = tmpres[:2]
        allpos = [[]]
        if True:
            for r in tqdm(tmpres, desc="Converting to indices"):
                # each r is a list of 
                inds = chunks_to_inds(r)
                allpos.append([dataset[ind]['text'] for ind in inds])
        # allpos = [[] for _ in range(len(tmpres))]
        selectresults = hierarchical_positive_search(allpos, question_set["query"], "passagesearchtrain_v1_5.2k_4Bstartv2", actualpassages=dataset["text"], models=["Qwen/Qwen3-8B", "gpt-5-mini"])