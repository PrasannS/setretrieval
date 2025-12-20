# given a colbert model path, a datastore (dataset with a text column), and a set of queries, embed both (cache the embeddings)
# and compute diversity (average pairwise similarity between embeddings) within a query, as well as across random queries

import argparse
import os
import wandb
from setretrieval.inference.easy_indexer import ColBERTEasyIndexer
from datasets import Dataset
from setretrieval.utils.utils import pickload, pickdump

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="propercache/data/datastores/wikipedia_docs_15k_decont")
    # query or document
    parser.add_argument("--data_type", type=str, default="document", choices=["query", "document"])
    # use bert-base by default
    # their trained model is lightonai/colbertv2.0
    parser.add_argument("--model_path", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--cachekey", type=str, default="bertwikidefault")
    
    args = parser.parse_args()

    cacchefolder = f"propercache/cache/embdiversity/{args.cachekey}"
    os.makedirs(cacchefolder, exist_ok=True)

    if os.path.exists(os.path.join(cacchefolder, f"{args.data_type}.pkl")):
        embeds = pickload(os.path.join(cacchefolder, f"{args.data_type}.pkl"))
    else:
        embeds = None

    # load the data
    data = Dataset.load_from_disk(args.data_path)
    if args.data_type == "document":
        documents = list(data["text"])
    elif args.data_type == "query":
        documents = list(data["question"])


    if embeds is None:
        model = ColBERTEasyIndexer(model_name=args.model_path)
        # embed the text
        embeds = model.embed_with_multi_gpu(documents, qtype=args.data_type)
        pickdump(embeds, os.path.join(cacchefolder, f"{args.data_type}.pkl"))

    # breakpoint()