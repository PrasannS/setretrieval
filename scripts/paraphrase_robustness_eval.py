"""
Evaluate ColBERT retrieval robustness after paraphrasing corpus chunks.

Given a ColBERT model, an eval set, a corpus, and a paraphrase prompt, this
script paraphrases either:
  - 'pos'  : corpus docs that appear as positives for some query
  - 'neg'  : corpus docs that are NOT positives for any query
  - 'all'  : every corpus doc

It then builds a fresh ColBERT index on the modified corpus and reports recall,
precision, and atleastone @ k.

Usage:
    python scripts/paraphrase_robustness_eval.py \\
        --model_name <colbert_checkpoint> \\
        --eval_set_path propercache/data/evalsets/settest_v1 \\
        --corpus_path propercache/data/datastores/wikipedia_docs_15k \\
        --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" \\
        --mode pos \\
        --llm_model gemini-2.5-flash \\
        --k 10
"""

import argparse
import hashlib
import os
from statistics import mean

from datasets import Dataset, load_from_disk

from setretrieval.indexers.colbert_faissindexer import ColBERTMaxSimIndexer
from setretrieval.inference.oai_request_client import ParallelResponsesClient, PRICING
from setretrieval.utils.utils import check_process_tset

def paraphrase_texts(texts, prompt_template, llm_model, max_concurrent=100):
    """Paraphrase a list of texts using an LLM via ParallelResponsesClient.

    Args:
        texts: List of strings to paraphrase.
        prompt_template: String with '{}' placeholder that wraps each text.
        llm_model: Model name accepted by ParallelResponsesClient.
        max_concurrent: Max parallel API requests.

    Returns:
        List of paraphrased strings (falls back to original on failure).
    """
    client = ParallelResponsesClient(max_concurrent=max_concurrent)
    prompts = [prompt_template.format(text) for text in texts]
    results = client.run(model=llm_model, prompts=prompts)

    paraphrased = []
    for orig, result in zip(texts, results):
        if result["success"] and result["response"]:
            paraphrased.append(result["response"])
        else:
            print(f"  Warning — paraphrase failed (keeping original): {result.get('error', 'unknown')}")
            paraphrased.append(orig)

    stats = client.get_stats()
    print(
        f"  Paraphrasing done. Cost: ${stats['total_cost_usd']:.4f}, "
        f"API calls: {stats['api_calls']}, Cache hits: {stats['cache_hits']}"
    )
    client.close()
    return paraphrased


def eval_with_paraphrased_chunks(
    model_name,
    eval_set,
    corpus_path,
    paraphrase_prompt,
    mode,
    llm_model="gemini-2.5-flash",
    k=10,
    max_concurrent=100,
    cache_dir="propercache/cache/paraphrase_eval",
    force_paraphrase=False,
):
    assert "{}" in paraphrase_prompt, "paraphrase_prompt must contain '{}' as the text placeholder"
    assert mode in ("pos", "neg", "all", "none"), "mode must be one of: pos, neg, all"

    eval_set = check_process_tset(eval_set)
    corpus = load_from_disk(corpus_path)


    # --- Build text → corpus-index mapping -----------------------------------
    corpus_texts = list(corpus["text"])
    text_to_idx = {text: i for i, text in enumerate(corpus_texts)}

    query_pos_indices = []
    missing = 0
    for row in eval_set:
        idxs = set()
        for chunk in row["pos_chunks"]:
            if chunk in text_to_idx:
                idxs.add(text_to_idx[chunk])
            else:
                missing += 1
        query_pos_indices.append(idxs)

    if missing:
        print(f"Warning: {missing} pos_chunks not found in corpus (skipped in eval).")

    if mode == "none":
        modified_corpus_path = corpus_path
        modified_dataset = corpus
    else:
        # --- Collect positive texts and map each query to its positive indices ----
        all_pos_texts = set()
        for row in eval_set:
            for chunk in row["pos_chunks"]:
                all_pos_texts.add(chunk)


        # --- Determine which corpus indices to paraphrase ------------------------
        pos_indices = {text_to_idx[t] for t in all_pos_texts if t in text_to_idx}
        if mode == "all":
            to_paraphrase = set(range(len(corpus_texts)))
        elif mode == "pos":
            to_paraphrase = pos_indices
        else:  # neg
            to_paraphrase = set(range(len(corpus_texts))) - pos_indices

        print(
            f"Mode '{mode}': will paraphrase {len(to_paraphrase):,} / {len(corpus_texts):,} corpus docs."
        )
        # we don't need caching I think
        # --- Cache the modified corpus so re-runs with same config are cheap -----
        prompt_hash = hashlib.sha256(paraphrase_prompt.encode()).hexdigest()[:12]
        corpus_cache_key = f"{mode}_{llm_model.replace('/', '_')}_{prompt_hash}"
        modified_corpus_path = os.path.join(cache_dir, "corpora", corpus_cache_key)

        sorted_indices = sorted(to_paraphrase)
        texts_to_paraphrase = [corpus_texts[i] for i in sorted_indices]
        print(f"Paraphrasing {len(texts_to_paraphrase):,} docs with {llm_model}...")
        paraphrased = paraphrase_texts(texts_to_paraphrase, paraphrase_prompt, llm_model, max_concurrent)

        new_texts = corpus_texts.copy()
        for idx, new_text in zip(sorted_indices, paraphrased):
            new_texts[idx] = new_text

        os.makedirs(os.path.join(cache_dir, "corpora"), exist_ok=True)
        modified_dataset = Dataset.from_dict({"text": new_texts})
        modified_dataset.save_to_disk(modified_corpus_path)
        print(f"Saved modified corpus to {modified_corpus_path}")

    # --- Build ColBERT index on the modified corpus --------------------------
    indexer = ColBERTMaxSimIndexer(model_name=model_name)
    index_id = indexer.index_dataset(modified_corpus_path)

    # TODO maybe want to clean up / debug how index re-usage works a little bit...

    # --- Retrieve and evaluate -----------------------------------------------
    questions = list(eval_set["question"])
    results = indexer.search(questions, index_id, k=k)

    # breakpoint()
    metres = {"precision": [], "recall": [], "atleastone": []}
    for preds, pos_idxs in zip(results, query_pos_indices):
        retrieved_idxs = {int(pred["index"]) for pred in preds}
        intersection = retrieved_idxs & pos_idxs
        n_retrieved = len(retrieved_idxs)
        n_pos = len(pos_idxs)

        metres["precision"].append(len(intersection) / n_retrieved if n_retrieved else 0.0)
        # Recall denominator: min(n_pos, k) — can't retrieve more than k docs
        metres["recall"].append(len(intersection) / min(n_pos, k) if n_pos else 0.0)
        metres["atleastone"].append(1.0 if intersection else 0.0)

    return metres


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ColBERT recall after paraphrasing corpus chunks."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Path to ColBERT checkpoint")
    parser.add_argument(
        "--eval_set_path", type=str, required=True, help="Path to HF eval dataset (question, pos_chunks)"
    )
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="Path to HF corpus dataset (text column)"
    )
    parser.add_argument(
        "--paraphrase_prompt",
        type=str,
        required=True,
        help="Prompt template with '{}' placeholder for the document text",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pos", "neg", "all", "none"],
        help="Which corpus docs to paraphrase: pos (positive docs), neg (non-positive docs), all",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gemini-2.5-flash-lite",
        choices=list(PRICING.keys()),
        help="LLM for paraphrasing",
    )
    parser.add_argument("--k", type=int, default=10, help="Retrieval cutoff for recall/precision")
    parser.add_argument(
        "--max_concurrent", type=int, default=100, help="Max parallel API requests"
    )
    parser.add_argument("--colbert_qvecs", type=int, default=-1)
    parser.add_argument("--colbert_dvecs", type=int, default=-1)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="propercache/cache/paraphrase_eval",
        help="Directory for caching modified corpora and indices",
    )
    parser.add_argument(
        "--force_paraphrase",
        action="store_true",
        help="Redo paraphrasing even if a cached modified corpus exists",
    )
    args = parser.parse_args()

    eval_set = Dataset.load_from_disk(args.eval_set_path)
    print(f"Loaded corpus docs | eval set: {len(eval_set)} queries")

    metres = eval_with_paraphrased_chunks(
        model_name=args.model_name,
        eval_set=eval_set,
        corpus_path=args.corpus_path,
        paraphrase_prompt=args.paraphrase_prompt,
        mode=args.mode,
        llm_model=args.llm_model,
        k=args.k,
        max_concurrent=args.max_concurrent,
        cache_dir=args.cache_dir,
        force_paraphrase=args.force_paraphrase,
    )

    print(f"\n--- Results (k={args.k}, mode={args.mode}) ---")
    print(f"Mean precision:  {mean(metres['precision']):.4f}")
    print(f"Mean recall:     {mean(metres['recall']):.4f}")
    print(f"Mean atleastone: {mean(metres['atleastone']):.4f}")
