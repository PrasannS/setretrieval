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
from datasets import Dataset, load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Build NQ eval set with BM25-filtered corpus")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="propercache/data",
        help="Root output directory (will create evalsets/ and datastores/ subdirs)",
    )
    parser.add_argument("--top_k", type=int, default=100, help="Number of BM25 results per query")
    parser.add_argument(
        "--split", type=str, default="test", help="BEIR NQ split to use (default: test)"
    )
    parser.add_argument(
        "--evalset_name", type=str, default="nq_testset", help="Name for the output eval set"
    )
    parser.add_argument(
        "--corpus_name", type=str, default="nqcorpus_bm25", help="Name for the output corpus"
    )
    args = parser.parse_args()

    # ---- Load BEIR Natural Questions ----
    print("Loading BEIR NQ corpus...")
    corpus_ds = load_dataset("BeIR/nq", "corpus", split="corpus")

    print(f"Loading BEIR NQ queries (split={args.split})...")
    queries_ds = load_dataset("BeIR/nq", "queries", split="queries")

    print("Loading BEIR NQ qrels...")
    qrels_ds = load_dataset("BeIR/nq-qrels", split=args.split)

    # ---- Build lookup maps ----
    print("Building corpus id -> text map...")
    # Combine title and text for each corpus document (BEIR convention)
    cid_to_idx = {}
    corpus_texts = []
    for i, row in enumerate(tqdm(corpus_ds, desc="Processing corpus")):
        cid_to_idx[str(row["_id"])] = i
        title = row.get("title", "")
        text = row.get("text", "")
        full_text = f"{title}\n{text}".strip() if title else text
        corpus_texts.append(full_text)

    print(f"Corpus size: {len(corpus_texts)}")

    qid_to_text = {str(row["_id"]): row["text"] for row in queries_ds}
    print(f"Number of queries: {len(qid_to_text)}")

    # Build qrels: query_id -> list of positive corpus_ids
    qrels = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = row["score"]
        if score > 0:
            qrels.setdefault(qid, []).append(cid)

    # Filter to queries that have qrels and exist in the query set
    valid_qids = [qid for qid in qrels if qid in qid_to_text]
    print(f"Queries with positive relevance judgments: {len(valid_qids)}")

    # ---- Build BM25 index over full corpus ----
    print("Tokenizing corpus for BM25...")
    stemmer = Stemmer.Stemmer("english")
    corpus_tokenized = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)

    print("Building BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokenized)

    # ---- Retrieve top-k per query and collect corpus subset ----
    print(f"Retrieving BM25 top-{args.top_k} for each query...")
    selected_corpus_indices = set()
    query_texts = [qid_to_text[qid] for qid in valid_qids]

    # Process in batches to manage memory
    batch_size = 256
    all_results = []
    for start in tqdm(range(0, len(query_texts), batch_size), desc="BM25 retrieval"):
        batch = query_texts[start : start + batch_size]
        tokenized_queries = bm25s.tokenize(batch, stemmer=stemmer)
        results, scores = retriever.retrieve(tokenized_queries, k=min(args.top_k, len(corpus_texts)))
        all_results.append(results)

    import numpy as np

    all_results = np.concatenate(all_results, axis=0)

    # Collect all retrieved indices
    for i in range(len(valid_qids)):
        for idx in all_results[i]:
            selected_corpus_indices.add(int(idx))

    # ---- BM25 recall stats (before adding positives to corpus) ----
    total_pos = 0
    found_pos = 0
    queries_with_all_pos = 0
    queries_with_any_pos = 0
    for qid in valid_qids:
        pos_cids = [cid for cid in qrels[qid] if cid in cid_to_idx]
        q_total = len(pos_cids)
        q_found = sum(1 for cid in pos_cids if cid_to_idx[cid] in selected_corpus_indices)
        total_pos += q_total
        found_pos += q_found
        if q_found == q_total:
            queries_with_all_pos += 1
        if q_found > 0:
            queries_with_any_pos += 1

    print(f"\n=== BM25 Top-{args.top_k} Recall Stats ===")
    print(f"Total positive documents across all queries: {total_pos}")
    print(f"Positives found in BM25 top-{args.top_k}: {found_pos}/{total_pos} ({100*found_pos/total_pos:.1f}%)")
    print(f"Queries with ALL positives retrieved: {queries_with_all_pos}/{len(valid_qids)} ({100*queries_with_all_pos/len(valid_qids):.1f}%)")
    print(f"Queries with at least one positive retrieved: {queries_with_any_pos}/{len(valid_qids)} ({100*queries_with_any_pos/len(valid_qids):.1f}%)")
    print()

    # Also ensure all positive documents are included in the corpus
    for qid in valid_qids:
        for cid in qrels[qid]:
            if cid in cid_to_idx:
                selected_corpus_indices.add(cid_to_idx[cid])

    print(f"Selected corpus size (deduplicated, with positives added): {len(selected_corpus_indices)}")

    # ---- Build the filtered corpus ----
    selected_indices_sorted = sorted(selected_corpus_indices)
    # Map old corpus index -> new corpus index (not needed for storage, but useful for verification)
    filtered_texts = [corpus_texts[i] for i in selected_indices_sorted]

    # ---- Build the eval set ----
    questions = []
    pos_chunks = []
    skipped = 0
    for qid in valid_qids:
        pos_texts = []
        for cid in qrels[qid]:
            if cid in cid_to_idx:
                pos_texts.append(corpus_texts[cid_to_idx[cid]])
        if not pos_texts:
            skipped += 1
            continue
        questions.append(qid_to_text[qid])
        pos_chunks.append(pos_texts)

    if skipped:
        print(f"Skipped {skipped} queries with no valid positive documents")

    print(f"Final eval set: {len(questions)} queries")
    print(f"Final corpus: {len(filtered_texts)} documents")

    # ---- Save ----
    evalset_path = os.path.join(args.output_dir, "evalsets", args.evalset_name)
    corpus_path = os.path.join(args.output_dir, "datastores", args.corpus_name)
    os.makedirs(os.path.dirname(evalset_path), exist_ok=True)
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

    eval_ds = Dataset.from_dict({"question": questions, "pos_chunks": pos_chunks})
    eval_ds.save_to_disk(evalset_path)
    print(f"Saved eval set to: {evalset_path}")

    corpus_ds_out = Dataset.from_dict({"text": filtered_texts})
    corpus_ds_out.save_to_disk(corpus_path)
    print(f"Saved corpus to: {corpus_path}")

    print("Done.")


if __name__ == "__main__":
    main()
