import argparse
import os
from setretrieval.inference.easy_indexer import SingleEasyIndexer
from datasets import Dataset
from setretrieval.utils.utils import pickload, pickdump, check_process_tset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_path", type=str, default="propercache/data/evalsets/wikirands100k-wikigemini_tinybigtest")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--cachekey", type=str, default="qwen4bwiki100ktest")
    
    args = parser.parse_args()

    savepath = f"propercache/cache/embdiversity/{args.cachekey}"
    os.makedirs(savepath, exist_ok=True)
    embeds = None
    if os.path.exists(savepath+"/state.json"):
        embeds = Dataset.load_from_disk(savepath)

    # load the data
    data = Dataset.load_from_disk(args.testset_path)
    data = check_process_tset(data)
    # breakpoint()
    questions = list(data["question"])
    poschunks = list(data["pos_chunks"])

    if embeds is None:
        model = SingleEasyIndexer(model_name=args.model_path)
        # embed the positive chunks (flatten and then re-group by original lengths)
        plens = [len(p) for p in poschunks]
        flatposchunks = [item for sublist in poschunks for item in sublist]
        embeds = model.embed_with_multi_gpu(flatposchunks)
        ind = 0
        # re-group by original length
        regroupedembeds = []
        for p in plens:
            regroupedembeds.append(embeds[ind:ind+p])
            ind += p
        queryembeds = model.embed_with_multi_gpu(questions)
        # breakpoint()
        ds = Dataset.from_dict({
            'question': questions, 
            'pos_chunks': poschunks,
            'query_embeds': queryembeds,
            'pos_embeds': regroupedembeds,
        })
        ds.save_to_disk(savepath)