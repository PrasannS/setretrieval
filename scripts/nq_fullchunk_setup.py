"""
Build a Natural Questions eval set and BM25-filtered corpus from the BEIR benchmark.

Produces:
  - evalset:   {question: str, pos_chunks: list[str]}  saved via Dataset.save_to_disk
  - datastore: {text: str}  containing only the BM25 top-k documents per query (deduplicated)

Usage:
    python scripts/make_nq_evalset.py \
        --output_dir propercache/data \
        --top_k 100 \
        --split test
"""
# old ds version is 4.5.0, switch to 3.6.0 for just this script

import argparse
import os

import bm25s
import Stemmer
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NQ eval set with BM25-filtered corpus")
    parser.add_argument(
        "--origchunks",
        type=str,
        default="propercache/data/datastores/nqcorpus_fullchunks",
        help="Root output directory (will create evalsets/ and datastores/ subdirs)",
    )

    args = parser.parse_args()

    # load in nq corpus (full)
    corpus_ds = load_dataset("BeIR/nq", "corpus", split="corpus")
    oldcorp = load_from_disk(args.origchunks)

    oldcorptexts = set(oldcorp['text'])
    ldocind = 0
    ldoctitle = corpus_ds[0]['title']
    found = False
    keepinds = []
    findstmp = []
    for i, row in enumerate(tqdm(corpus_ds, desc="Processing corpus")):
        is_positive = row['title'] + "\n" + row['text'] in oldcorptexts

        # treat 5 chunk intervals as "documents" — flush BEFORE updating found
        # so that a positive landing on a boundary isn't lost
        if row['title'] != ldoctitle or (i - ldocind) > 5:
            if found:
                keepinds.extend(list(range(ldocind, i)))
            found = False
            ldocind = i
            ldoctitle = row['title']
            findstmp = []

        if is_positive:
            findstmp.append(i)
            found = True
    
    # flush the last group
    if found:
        keepinds.extend(list(range(ldocind, len(corpus_ds))))

    print("Keeping %d documents" % len(keepinds))
    subsel = corpus_ds.select(keepinds)
    subsel.save_to_disk(args.origchunks+"_fulldocs")