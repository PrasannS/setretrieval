# do single or colbert easy-index beforehand for a dataset (given path)

import argparse
from wikipedia_eval import ds_load_indexer # HACK move to utils at some point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_type", type=str, default="bm25")
    parser.add_argument("--model_name", type=str, default="bm25")
    parser.add_argument("--dataset_path", type=str, default="propercache/data/datastores/wikipedia_docs_15k")
    args = parser.parse_args()

    indexer, index_id = ds_load_indexer(args.index_type, args.model_name, args.dataset_path)
    print(f"Indexed {len(indexer.documents[index_id])} documents for index {index_id}")