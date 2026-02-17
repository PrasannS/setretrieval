"""
Transform a corpus by shortening or lengthening all documents using an API model.

Usage:
    python scripts/transform_corpus.py \
        --corpus_path data/my_corpus \
        --mode shorten \
        --output_path data/my_corpus_short \
        --model gemini-2.5-flash
"""

import argparse
from datasets import load_from_disk, Dataset
from setretrieval.inference.oai_request_client import ParallelResponsesClient, PRICING
from tqdm import tqdm

SHORTEN_PROMPT = (
    "Rewrite the following document to be significantly shorter while preserving all key "
    "information and meaning. Remove redundancy, simplify sentences, and condense where possible. "
    "Output ONLY the rewritten document, nothing else.\n\nDocument:\n{}"
)

LENGTHEN_PROMPT = (
    "Rewrite the following document to be significantly longer and more detailed. Expand on the "
    "existing points with more explanation, examples, and context while staying faithful to the "
    "original content. Do not add fabricated facts. "
    "Output ONLY the rewritten document text. Do not include any preamble, headers, meta-commentary, "
    "or phrases like 'Here is the rewritten document'. Just start directly with the rewritten content."
    "\n\nDocument:\n{}"
)


def main():
    parser = argparse.ArgumentParser(description="Shorten or lengthen corpus documents via an API model.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to HF dataset with 'text' column")
    parser.add_argument("--mode", type=str, required=True, choices=["shorten", "lengthen"])
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the transformed dataset")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", choices=list(PRICING.keys()))
    parser.add_argument("--max_concurrent", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5000, help="Process in batches of this size")
    parser.add_argument("--max_output_tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N documents")
    parser.add_argument("--debug", action="store_true", help="Run on 10 examples and print before/after")
    args = parser.parse_args()

    if args.debug:
        args.limit = 10

    corpus = load_from_disk(args.corpus_path)
    if args.limit:
        corpus = corpus.select(range(min(args.limit, len(corpus))))
    print(f"Loaded corpus with {len(corpus)} documents")

    prompt_template = SHORTEN_PROMPT if args.mode == "shorten" else LENGTHEN_PROMPT

    client = ParallelResponsesClient(max_concurrent=args.max_concurrent)

    texts = [row["text"] for row in corpus]
    transformed = []

    for start in tqdm(range(0, len(texts), args.batch_size), desc="Batches"):
        batch = texts[start : start + args.batch_size]
        prompts = [prompt_template.format(doc) for doc in batch]
        results = client.run(
            model=args.model,
            prompts=prompts,
            max_output_tokens=args.max_output_tokens,
        )
        for orig, result in zip(batch, results):
            if result["success"] and result["response"]:
                transformed.append(result["response"])
            else:
                print(f"Failed for doc (keeping original): {result.get('error', 'unknown')}")
                transformed.append(orig)
        # print total cost so far
        print(f"Total cost so far: ${client.total_cost:.4f}")

    if args.debug:
        print("\n" + "=" * 80)
        print(f"DEBUG: Before/After comparison ({args.mode})")
        print("=" * 80)
        for i, (orig, new) in enumerate(zip(texts, transformed)):
            print(f"\n--- Document {i} (original, {len(orig)} chars) ---")
            print(orig[:500])
            print(f"\n--- Document {i} ({args.mode}d, {len(new)} chars) ---")
            print(new[:500])
            print()

    stats = client.get_stats()
    print(f"Done. Cost: ${stats['total_cost_usd']:.4f}, API calls: {stats['api_calls']}, Cache hits: {stats['cache_hits']}")

    # Build output dataset preserving all original columns, replacing 'text'
    out_data = {col: corpus[col] for col in corpus.column_names if col != "text"}
    out_data["text"] = transformed
    out_dataset = Dataset.from_dict(out_data)
    out_dataset.save_to_disk(args.output_path)
    print(f"Saved transformed corpus to {args.output_path}")

    client.close()


if __name__ == "__main__":
    main()
