# script to generate set evaluation data
from setretrieval.datagen.generate_setdata import hierarchical_positive_search
import argparse
import os   
from datasets import DatasetDict, Dataset
from setretrieval.utils.utils import pickload, get_deterministic_hash
from tqdm import tqdm
from setretrieval.datagen.generate_setdata import chunks_to_inds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="propercache/data/datastores/wikipedia_docs_10k_decont")
    parser.add_argument("--starter_question_set", type=str, default="propercache/data/colbert_training/gemini_ntrain_ptest")
    parser.add_argument("--startindex", type=int, default=0)
    parser.add_argument("--endindex", type=int, default=1500)
    parser.add_argument("--comparison", type=str, default="no")
    parser.add_argument("--modelcnt", type=int, default=1)
    parser.add_argument("--testcnt", type=int, default=0)
    parser.add_argument("--max_s1_pps", type=int, default=1500)
    parser.add_argument("--temp_minlimit", type=int, default=1000)

    args = parser.parse_args()

    # make folder called gendata in cache
    os.makedirs("propercache/cache/gendata", exist_ok=True)

    dataset = Dataset.load_from_disk(args.dataset_path)
    # try to take another node...
    question_set = DatasetDict.load_from_disk(args.starter_question_set)['test'].select(range(args.startindex, args.endindex))

    test = args.testcnt > 0
    keyval = f"passagesearchtrain_v2{get_deterministic_hash(args.dataset_path)}_{get_deterministic_hash(args.starter_question_set)}_{args.startindex}_{args.endindex}"
    questions = question_set["query"]
    if test:
        keyval = keyval + "_test"
        questions = questions[:args.testcnt]
    if True:
        modord = ["Qwen/Qwen3-8B", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
        modord = modord[:args.modelcnt]
        selectresults = hierarchical_positive_search(dataset["text"], questions, keyval, models=modord, comparison=args.comparison=="yes", max_s1_pps=args.max_s1_pps, temp_minlimit=args.temp_minlimit)
    else:
        # debug thing, forget for now
        tmpres = pickload("cache/gendata/passagesearchtrain_v1_5.2k_0.pkl")
        # tmpres = tmpres[:2]
        allpos = []
        if True:
            for r in tqdm(tmpres, desc="Converting to indices"):
                # each r is a list of 
                inds = chunks_to_inds(r)
                allpos.append([dataset[ind]['text'] for ind in inds])
        # allpos = [[] for _ in range(len(tmpres))]
        selectresults = hierarchical_positive_search(allpos, question_set["query"], f"passagesearchtrain_v1_{args.startindex}_{args.endindex}", actualpassages=dataset["text"], models=["Qwen/Qwen3-8B", "gemini-2.5-flash"])