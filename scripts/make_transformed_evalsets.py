"""
Given transformed (short/long) corpora and the original eval set, produce:
  - 2 new eval sets (short/long) with pos_chunks replaced by transformed text
  - 4 new corpora:
      - {mode}_posonly:  only eval-set positives are replaced, rest stays original
      - {mode}_negonly:  only non-positives are replaced, positives stay original

Usage:
    python scripts/make_transformed_evalsets.py \
        --corpus_path propercache/data/datastores/fiqacorpus \
        --short_corpus_path propercache/data/datastores/fiqacorpus_short \
        --long_corpus_path propercache/data/datastores/fiqacorpus_long \
        --eval_set_path propercache/data/evalsets/fiqa_testset \
        --output_dir propercache/data
"""

import argparse
from datasets import load_from_disk, Dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--short_corpus_path", type=str, required=True)
    parser.add_argument("--long_corpus_path", type=str, required=True)
    parser.add_argument("--eval_set_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    breakpoint()
    corpus = load_from_disk(args.corpus_path)
    short_corpus = load_from_disk(args.short_corpus_path)
    long_corpus = load_from_disk(args.long_corpus_path)
    evalset = load_from_disk(args.eval_set_path)

    orig_texts = corpus["text"]
    short_texts = short_corpus["text"]
    long_texts = long_corpus["text"]

    # Map original text -> corpus index
    text_to_idx = {t: i for i, t in enumerate(orig_texts)}

    # Find all corpus indices that are eval-set positives
    pos_indices = set()
    for row in tqdm(evalset, desc="Finding positive chunks"):
        for chunk in row["pos_chunks"]:
            assert chunk in text_to_idx, f"pos_chunk not found in corpus: {chunk[:100]}"
            pos_indices.add(text_to_idx[chunk])

    print(f"Eval set: {len(evalset)} queries, {len(pos_indices)} unique positive docs out of {len(corpus)}")

    print("Here 1")
    for mode, transformed_texts in [("short", short_texts), ("long", long_texts)]:
        print("Here 2")
        # Build original->transformed text map
        orig_to_transformed = {orig_texts[i]: transformed_texts[i] for i in tqdm(range(len(orig_texts)), desc="Building original->transformed map")}

        # --- New eval set: replace pos_chunks with transformed text ---
        new_questions = []
        new_pos_chunks = []
        for row in tqdm(evalset, desc="Building new eval set"):
            new_questions.append(row["question"])
            new_pos_chunks.append([orig_to_transformed[c] for c in row["pos_chunks"]])

        eval_out = Dataset.from_dict({"question": new_questions, "pos_chunks": new_pos_chunks})
        eval_out_path = f"{args.output_dir}/evalsets/fiqa_testset_{mode}"
        # breakpoint()
        eval_out.save_to_disk(eval_out_path)
        print(f"Saved eval set: {eval_out_path}")

        # --- Corpus: posonly (only positives replaced) ---
        posonly_texts = [
            transformed_texts[i] if i in pos_indices else orig_texts[i]
            for i in tqdm(range(len(orig_texts)), desc="Building posonly corpus")
        ]
        posonly_data= {'text': posonly_texts}

        posonly_out = Dataset.from_dict(posonly_data)
        posonly_path = f"{args.output_dir}/datastores/fiqacorpus_{mode}_posonly"
        # breakpoint()
        posonly_out.save_to_disk(posonly_path)
        print(f"Saved corpus: {posonly_path}")

        # --- Corpus: negonly (only non-positives replaced) ---
        negonly_texts = [
            orig_texts[i] if i in pos_indices else transformed_texts[i]
            for i in range(len(orig_texts))
        ]
        negonly_data = {'text': negonly_texts}
        negonly_out = Dataset.from_dict(negonly_data)
        negonly_path = f"{args.output_dir}/datastores/fiqacorpus_{mode}_negonly"
        # breakpoint()
        negonly_out.save_to_disk(negonly_path)
        print(f"Saved corpus: {negonly_path}")

    print("Done.")


if __name__ == "__main__":
    main()
