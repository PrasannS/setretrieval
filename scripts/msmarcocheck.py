from pylate import evaluation
import numpy as np
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_dataset_direct(dataset, split, sample=False):
    home = os.path.expanduser("~")
    out_dir = home + "/datasets"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    new_corpus = []
    for key in corpus:
        new_corpus.append(corpus[key]["text"] + " " + corpus[key]["title"])
    
    if sample:
        rng = np.random.default_rng(seed=42)

        qids_str = [key for key in qrels]
        dids_str = [list(qrels[qid].keys())[0] for qid in qids_str]
        dids_int = np.array([int(did) for did in dids_str]).astype(np.uint32)

        sample_size = 1000000
        inds = np.arange(len(new_corpus))
        rng.shuffle(inds)
        inds = inds[:sample_size]
        inds = np.union1d(dids_int, inds)
        sample_size = len(inds)

        new_corpus = [new_corpus[i] for i in inds]
        print(f"sampled corpus into {len(new_corpus)} docs")
    corpus = new_corpus

    new_queries = []
    for key in queries:
        new_queries.append(queries[key])
    queries = new_queries

    return corpus, queries, qrels

if __name__ == "__main__":
    dataset = "msmarco"
    
    corpus, queries, qrels = load_dataset_direct(dataset, "dev")
    breakpoint()