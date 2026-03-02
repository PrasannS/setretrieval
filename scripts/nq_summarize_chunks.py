"""
Summarize NQ corpus chunks using an LLM and update the eval set accordingly.

Takes the output of nq_fullchunk_setup.py (BeIR-format dataset with _id/title/text
columns), groups consecutive chunks with the same title into intervals of group_size,
summarizes each group using an LLM, and rewrites the eval set so pos_chunks point to
the summarized text of the group that contained the original chunk.

Usage:
    python scripts/nq_summarize_chunks.py \
        --corpus_path propercache/data/datastores/nqcorpus_fullchunks_fulldocs \
        --eval_set_path propercache/data/evalsets/nq_testset \
        --output_corpus_path propercache/data/datastores/nqcorpus_summarized \
        --output_eval_set_path propercache/data/evalsets/nq_testset_summarized \
        --group_size 5 \
        --llm_model gemini-2.5-flash-lite
"""

import argparse
import os

from datasets import Dataset, load_from_disk
from tqdm import tqdm

from setretrieval.inference.oai_request_client import PRICING, ParallelResponsesClient

DEFAULT_PROMPT = (
    "The following are consecutive excerpts from a Wikipedia article. "
    "Write a concise, factual summary that captures all information present using as few words as possible:\n\n{}"
)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_corpus_chunks(corpus, group_size=5):
    """Group consecutive rows with the same title into intervals of up to group_size.

    Returns a list of dicts:
        {
            'title': str,
            'texts': list[str],           # raw text fields (no title prefix)
            'combined_keys': list[str],   # "title\\ntext".strip() for each chunk
        }
    """
    groups = []
    current_title = None
    current_texts = []
    current_keys = []

    def flush():
        if current_texts:
            groups.append(
                {
                    "title": current_title,
                    "texts": current_texts[:],
                    "combined_keys": current_keys[:],
                }
            )

    for row in tqdm(corpus, desc="Grouping chunks"):
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        combined_key = f"{title}\n{text}".strip()

        new_title = title != current_title
        full_group = len(current_texts) >= group_size

        if new_title or full_group:
            flush()
            current_title = title
            current_texts = [text]
            current_keys = [combined_key]
        else:
            current_texts.append(text)
            current_keys.append(combined_key)

    flush()
    return groups


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarize_groups(groups, prompt_template, llm_model, max_concurrent=100):
    """Call the LLM to summarize each group; falls back to concatenation on error."""
    client = ParallelResponsesClient(max_concurrent=max_concurrent)

    prompts = []
    for group in groups:
        body = f"Title: {group['title']}\n\n" + "\n\n".join(group["texts"])
        prompts.append(prompt_template.format(body))

    print(f"Summarizing {len(prompts):,} groups with {llm_model}...")
    results = client.run(model=llm_model, prompts=prompts)

    summaries = []
    for group, result in zip(groups, results):
        if result["success"] and result["response"]:
            summaries.append(result["response"])
        else:
            fallback = " ".join(group["texts"])
            print(
                f"  Warning — summarization failed, concatenating chunks: "
                f"{result.get('error', 'unknown')}"
            )
            summaries.append(fallback)

    stats = client.get_stats()
    print(
        f"  Done. Cost: ${stats['total_cost_usd']:.4f}, "
        f"API calls: {stats['api_calls']}, Cache hits: {stats['cache_hits']}"
    )
    client.close()
    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize NQ corpus chunks and update the eval set"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Path to fulldocs corpus (output of nq_fullchunk_setup.py)",
    )
    parser.add_argument(
        "--eval_set_path",
        type=str,
        required=True,
        help="Path to eval set (output of make_nq_evalset.py)",
    )
    parser.add_argument(
        "--output_corpus_path",
        type=str,
        required=True,
        help="Where to save the summarized corpus",
    )
    parser.add_argument(
        "--output_eval_set_path",
        type=str,
        required=True,
        help="Where to save the updated eval set",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=5,
        help="Max number of consecutive same-title chunks to merge per summary",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gemini-2.5-flash-lite",
        choices=list(PRICING.keys()),
        help="LLM model to use for summarization",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=100,
        help="Max parallel API requests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt template with '{}' placeholder for the combined chunk text",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Summarize only 5 groups, print full before/after, and exit without saving",
    )
    args = parser.parse_args()

    assert "{}" in args.prompt, "Prompt must contain '{}' as the text placeholder"

    # ---- Load ---------------------------------------------------------------
    corpus = load_from_disk(args.corpus_path)
    eval_set = load_from_disk(args.eval_set_path)
    print(f"Corpus: {len(corpus):,} chunks | Eval set: {len(eval_set):,} queries")

    # ---- Group chunks -------------------------------------------------------
    groups = group_corpus_chunks(corpus, group_size=args.group_size)
    print(f"Grouped into {len(groups):,} groups (group_size={args.group_size})")

    if args.debug:
        debug_groups = groups[:5]
        print(f"\n--- DEBUG MODE: summarizing {len(debug_groups)} groups ---\n")
        debug_summaries = summarize_groups(
            debug_groups, args.prompt, args.llm_model, args.max_concurrent
        )
        sep = "-" * 72
        for i, (group, summary) in enumerate(zip(debug_groups, debug_summaries)):
            print(sep)
            print(f"Group {i+1} | title: {group['title']} | {len(group['texts'])} chunk(s)")
            print(sep)
            for j, text in enumerate(group["texts"]):
                print(f"  [chunk {j+1}] {text}")
            print(f"\n  --> SUMMARY:\n  {summary}\n")
        print(sep)
        return

    # ---- Summarize ----------------------------------------------------------
    summaries = summarize_groups(
        groups, args.prompt, args.llm_model, args.max_concurrent
    )

    # ---- Print examples -----------------------------------------------------
    print("\nExamples (before → after):")
    for i in range(min(3, len(groups))):
        n = len(groups[i]["texts"])
        combined = " | ".join(t[:60] for t in groups[i]["texts"])
        print(f"  [{groups[i]['title']}] {n} chunks: {combined[:120]}...")
        print(f"  Summary: {summaries[i][:200]}\n")

    # ---- Build mapping: original combined key → summary ---------------------
    # combined key = "title\ntext".strip(), matching make_nq_evalset.py format
    orig_to_summary = {}
    for group, summary in zip(groups, summaries):
        for key in group["combined_keys"]:
            orig_to_summary[key] = summary

    print(
        f"Built mapping: {len(orig_to_summary):,} original chunks "
        f"→ {len(groups):,} summaries"
    )

    # ---- Build new corpus (one entry per group) -----------------------------
    new_corpus_ds = Dataset.from_dict({"text": summaries})
    new_corpus_ds.save_to_disk(args.output_corpus_path)
    print(f"Saved summarized corpus ({len(summaries):,} docs) to {args.output_corpus_path}")

    # ---- Update eval set ----------------------------------------------------
    new_questions = []
    new_pos_chunks = []
    skipped = 0
    missing_chunks = 0
    total_orig_pos = 0
    total_new_pos = 0

    for row in eval_set:
        total_orig_pos += len(row["pos_chunks"])
        new_pos = []
        seen = set()
        for chunk in row["pos_chunks"]:
            summary = orig_to_summary.get(chunk)
            if summary is None:
                missing_chunks += 1
                continue
            if summary not in seen:
                new_pos.append(summary)
                seen.add(summary)

        if not new_pos:
            skipped += 1
            continue

        new_questions.append(row["question"])
        new_pos_chunks.append(new_pos)
        total_new_pos += len(new_pos)

    if missing_chunks:
        print(
            f"Warning: {missing_chunks} pos_chunks had no mapping in the corpus "
            f"(these were dropped)"
        )
    if skipped:
        print(f"Skipped {skipped} queries with no mappable pos_chunks")

    avg_orig = total_orig_pos / len(eval_set) if eval_set else 0
    avg_new = total_new_pos / len(new_questions) if new_questions else 0
    print(
        f"Eval set: {len(eval_set):,} → {len(new_questions):,} queries | "
        f"avg pos_chunks per query: {avg_orig:.2f} → {avg_new:.2f} "
        f"(merged across groups)"
    )

    new_eval_ds = Dataset.from_dict({"question": new_questions, "pos_chunks": new_pos_chunks})
    new_eval_ds.save_to_disk(args.output_eval_set_path)
    print(f"Saved updated eval set to {args.output_eval_set_path}")
    print("Done.")


if __name__ == "__main__":
    main()
