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
    parser.add_argument("--dataset_path", type=str, default="propercache/data/datastores/wikipedia_docs_10k_decont")
    parser.add_argument("--starter_question_set", type=str, default="propercache/data/colbert_training/gemini_ntrain_ptest")
    parser.add_argument("--startindex", type=int, default=0)
    parser.add_argument("--endindex", type=int, default=1500)
    args = parser.parse_args()

    # make folder called gendata in cache
    os.makedirs("propercache/cache/gendata", exist_ok=True)

    dataset = Dataset.load_from_disk(args.dataset_path)
    # try to take another node...
    question_set = DatasetDict.load_from_disk(args.starter_question_set)['train'].select(range(args.startindex, args.endindex))

    test = False
    keyval = f"passagesearchtrain_v2_{args.startindex}_{args.endindex}"
    questions = question_set["query"]
    if test:
        keyval = keyval + "_test"
        questions = questions[:3]
    if True:
        selectresults = hierarchical_positive_search(dataset["text"], questions, keyval, models=["/accounts/projects/sewonm/prasann/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"])
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