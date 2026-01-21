### notebook that will get results for wikipedia eval set (TODO may add other datasets later)
from setretrieval.inference.easy_indexer import BM25EasyIndexer, ColBERTEasyIndexer, SingleEasyIndexer, RandomEasyIndexer, TokenColBERTEasyIndexer
import argparse
from datasets import Dataset
import os
from setretrieval.utils.utils import pickdump, pickload, check_process_tset, preds_to_chunks
from statistics import mean
import json

def indexer_eval_row(predictions, truepositives, idxer, idx_id, metric="precision"):

    useddocs = [idxer.documents[idx_id][pred['index']]['text'] for pred in predictions]
    useddocs = set(useddocs)
    truepositives = set(truepositives)

    # return precision, recall
    if metric == "precision":
        return len(useddocs.intersection(truepositives)) / len(useddocs)
    elif metric == "recall":
        return len(useddocs.intersection(truepositives)) / min(len(truepositives), len(useddocs))
    elif metric == "atleastone": 
        return 1 if len(useddocs.intersection(truepositives)) > 0 else 0
    else:
        raise ValueError(f"Invalid metric: {metric}")

def ds_load_indexer(indextype, modelname, datasetpath, redo=False, qmod_name=None, qvecs=-1, dvecs=-1, ebsize=128, usefast="yes"):
    if indextype == "bm25":
        indexer = BM25EasyIndexer()
    elif indextype == "colbert":
        indexer = ColBERTEasyIndexer(model_name=modelname, qmod_name=qmod_name, qvecs=qvecs, dvecs=dvecs, use_bsize=ebsize, usefast=usefast)
    elif indextype == "divcolbert": 
        indexer = TokenColBERTEasyIndexer(model_name=modelname)
    elif indextype == "single":
        indexer = SingleEasyIndexer(model_name=modelname)
    elif indextype == "random": 
        indexer = RandomEasyIndexer() # works like BM25 technically
    else:
        raise ValueError(f"Invalid index type: {indextype}")

    # process the dataset in one go, make an index for it
    index_id = indexer.index_dataset(datasetpath, redo=redo)
    return indexer, index_id

def do_eval(indextype, modelname, datasetpath, evalsetpath, k, forceredo="no", qmod_name=None, qvecs=-1, dvecs=-1, ebsize=128, usefast="yes"):

    eval_set = Dataset.load_from_disk(evalsetpath)
    eval_set = check_process_tset(eval_set)

    indexer, index_id = ds_load_indexer(indextype, modelname, datasetpath, redo=forceredo=="yes", qmod_name=qmod_name, qvecs=qvecs, dvecs=dvecs, ebsize=ebsize, usefast=usefast)
    # process all queries in one go (assume that all queries are searching over the same data, and that index has everything)
    results = indexer.search(list(eval_set["question"]), index_id, k)

    # evaluate the results
    metres = {'precision': [], 'recall': [], 'atleastone': []}
    for met in metres.keys():
        metres[met].extend([indexer_eval_row(results[i], eval_set[i]["pos_chunks"], indexer, index_id, met) for i in range(len(results))])

    predictions_chunks = preds_to_chunks(results, indexer.documents[index_id])
    return metres, list(eval_set["question"]), predictions_chunks, list(eval_set['pos_chunks'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_type", type=str, default="bm25")
    parser.add_argument("--model_name", type=str, default="bm25")
    parser.add_argument("--dataset_path", type=str, default="propercache/data/datastores/wikipedia_docs_15k")
    parser.add_argument("--eval_set_path", type=str, default="propercache/data/evalsets/settest_v1_paraphrased")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--forceredo", type=str, default="no")
    parser.add_argument("--save_preds", type=str, default="no")
    parser.add_argument("--qmod_name", type=str, default=None)
    parser.add_argument("--colbert_qvecs", type=int, default=-1)
    parser.add_argument("--colbert_dvecs", type=int, default=-1)
    parser.add_argument("--colbert_ebsize", type=int, default=128)
    parser.add_argument("--colbert_usefast", type=str, default="yes")
    args = parser.parse_args()

    print("starting eval")
    
    os.makedirs("propercache/cache/setresults/", exist_ok=True)
    
    resultkey = args.eval_set_path.split("/")[-1]
    dsrep = args.dataset_path.replace("/", "_")
    methodkey = f"{args.index_type}-{args.model_name}-{args.k}-{dsrep}"
    if args.qmod_name is not None:
        methodkey += f"-{args.qmod_name.replace('/', '_')}"

    if os.path.exists("propercache/cache/setresults/"+resultkey+".pkl"): 
        metresults = pickload("propercache/cache/setresults/"+resultkey+".pkl")
    else:
        metresults = {}

    # we have at least 3 metrics
    if methodkey in metresults and len(metresults[methodkey]) == 3 and args.forceredo == "no":
        print(f"Loading results from cache for {resultkey}")
        metres = metresults[methodkey]
    else:
        metres, questions, preds, golds = do_eval(args.index_type, args.model_name, args.dataset_path, args.eval_set_path, args.k, args.forceredo, args.qmod_name, args.colbert_qvecs, args.colbert_dvecs, args.colbert_ebsize, args.colbert_usefast)
        metresults[methodkey] = metres
        pickdump(metresults, "propercache/cache/setresults/"+resultkey+".pkl")
        if len(args.save_preds) > 0:
            # save questions, preds, golds to jsonl file
            with open("propercache/cache/setresults/"+resultkey+"_"+args.save_preds+"_preds2.jsonl", "w") as f:
                for q, p, g in zip(questions, preds, golds):
                    f.write(json.dumps({"question": q, "preds": list(p), "golds": list(g)}) + "\n")

    print(f"Mean precision: {mean(metres['precision'])}")
    print(f"Mean recall: {mean(metres['recall'])}")
    print(f"Mean atleastone: {mean(metres['atleastone'])}")

    