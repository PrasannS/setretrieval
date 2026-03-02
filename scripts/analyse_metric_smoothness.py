from datasets import Dataset
import os
import argparse
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt


# ── Metrics ──────────────────────────────────────────────────────────────────

def recall_at_k(preds, golds, k=10):
    return len(set(preds[:k]).intersection(set(golds))) / min(len(set(golds)), k)


def precision_at_k(preds, golds, k=10):
    return len(set(preds[:k]).intersection(set(golds))) / k


def f1_at_k(preds, golds, k=10):
    p = precision_at_k(preds, golds, k)
    r = recall_at_k(preds, golds, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr_at_k(preds, golds, k=10):
    gold_set = set(golds)
    for rank, pred in enumerate(preds[:k], start=1):
        if pred in gold_set:
            return 1.0 / rank
    return 0.0


def ap_at_k(preds, golds, k=10):
    """Average Precision@k (MAP when averaged across queries)."""
    gold_set = set(golds)
    hits = 0
    running_precision = 0.0
    for rank, pred in enumerate(preds[:k], start=1):
        if pred in gold_set:
            hits += 1
            running_precision += hits / rank
    if hits == 0:
        return 0.0
    return running_precision / min(len(gold_set), k)


def dcg_at_k(preds, golds, k=10):
    gold_set = set(golds)
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, pred in enumerate(preds[:k], start=1)
        if pred in gold_set
    )


def ndcg_at_k(preds, golds, k=10):
    ideal_hits = min(len(set(golds)), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(preds, golds, k) / idcg


def hit_at_k(preds, golds, k=10):
    """Binary: 1 if any gold appears in top-k predictions."""
    return float(bool(set(preds[:k]).intersection(set(golds))))


def r_precision(preds, golds, k=10):
    """Precision at R where R = number of relevant docs (capped at k)."""
    r = min(len(set(golds)), k)
    return len(set(preds[:r]).intersection(set(golds))) / r if r > 0 else 0.0


METRICS = {
    "Recall@10":    lambda p, g: recall_at_k(p, g, 10),
    "Precision@10": lambda p, g: precision_at_k(p, g, 10),
    "F1@10":        lambda p, g: f1_at_k(p, g, 10),
    "MRR@10":       lambda p, g: mrr_at_k(p, g, 10),
    "MAP@10":       lambda p, g: ap_at_k(p, g, 10),
    "NDCG@10":      lambda p, g: ndcg_at_k(p, g, 10),
    "Hit@10":       lambda p, g: hit_at_k(p, g, 10),
    "R-Prec":       lambda p, g: r_precision(p, g, 10),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def dataset_scores(dset, metric_fn):
    return sum(metric_fn(row["preds"], row["golds"]) for row in dset) / len(dset)


def pairwise_agreement(scores_a, scores_b):
    """Fraction of dataset pairs where both metrics agree on which is better.

    scores_a / scores_b: list of floats, one per dataset (same order).
    A 'comparison' is consistent when sign(a_i - a_j) == sign(b_i - b_j).
    Ties on either metric are skipped.
    """
    n = len(scores_a)
    agree = 0
    total = 0
    for i, j in itertools.combinations(range(n), 2):
        da = scores_a[i] - scores_a[j]
        db = scores_b[i] - scores_b[j]
        if da == 0 or db == 0:
            continue
        total += 1
        if (da > 0) == (db > 0):
            agree += 1
    return agree / total if total > 0 else float("nan")


def margin_correlation(scores_a, scores_b):
    """Pearson correlation of pairwise margins between two metrics.

    For every pair (i, j) with i < j, compute the margin delta_a = a_i - a_j
    and delta_b = b_i - b_j, then return the Pearson r over all such pairs.
    """
    margins_a, margins_b = [], []
    for i, j in itertools.combinations(range(len(scores_a)), 2):
        margins_a.append(scores_a[i] - scores_a[j])
        margins_b.append(scores_b[i] - scores_b[j])
    if len(margins_a) < 2:
        return float("nan")
    return float(np.corrcoef(margins_a, margins_b)[0, 1])


def print_matrix(names, matrix):
    col_w = max(len(n) for n in names) + 2
    header = " " * col_w + "".join(n.rjust(col_w) for n in names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        row = name.ljust(col_w) + "".join(f"{matrix[i][j]:.3f}".rjust(col_w) for j in range(len(names)))
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_path", type=str, default="propercache/cache/setresults")
    parser.add_argument("--searchkey", type=str, default="fiqa_testset")
    parser.add_argument("--predlen", type=int, default=10)
    args = parser.parse_args()

    # ── Load datasets ──
    allfiles = {}
    for file in sorted(os.listdir(args.metric_path)):
        if args.searchkey in file and "preds2" in file and ".jsonl" in file:
            dset = Dataset.from_json(os.path.join(args.metric_path, file))
            if len(dset[0]["preds"]) == args.predlen:
                allfiles[file] = dset
                print(f"Loaded: {file}  ({len(dset)} rows)")

    if len(allfiles) < 2:
        print("Need at least 2 datasets to compute a comparison matrix.")
    else:
        dataset_names = list(allfiles.keys())
        datasets = [allfiles[n] for n in dataset_names]
        metric_names = list(METRICS.keys())

        # ── Compute per-dataset scores for every metric ──
        # scores[metric_idx][dataset_idx]
        scores = {}
        print("\n── Per-dataset scores ──────────────────────────────────────────────")
        header = f"{'Dataset':<50}" + "".join(f"{m:>12}" for m in metric_names)
        print(header)
        print("-" * len(header))
        for dname, dset in zip(dataset_names, datasets):
            row_scores = {}
            for mname, mfn in METRICS.items():
                s = dataset_scores(dset, mfn)
                row_scores[mname] = s
                scores.setdefault(mname, []).append(s)
            print(f"{dname:<50}" + "".join(f"{row_scores[m]:>12.4f}" for m in metric_names))

        # ── Pairwise agreement matrix ──
        print("\n── Pairwise metric agreement (fraction of consistent dataset rankings) ──")
        m = len(metric_names)
        agreement = [[0.0] * m for _ in range(m)]
        for i, mi in enumerate(metric_names):
            for j, mj in enumerate(metric_names):
                agreement[i][j] = pairwise_agreement(scores[mi], scores[mj])

        print_matrix(metric_names, agreement)

        # ── Margin correlation matrix ──
        print("\n── Pairwise margin correlation (Pearson r on score differences) ──")
        correlation = [[0.0] * m for _ in range(m)]
        for i, mi in enumerate(metric_names):
            for j, mj in enumerate(metric_names):
                correlation[i][j] = margin_correlation(scores[mi], scores[mj])

        print_matrix(metric_names, correlation)

        # ── Scatter plots for lowest-correlation pair per metric ──
        os.makedirs("figures", exist_ok=True)
        seen_pairs = set()
        for i, mi in enumerate(metric_names):
            # find lowest off-diagonal correlation
            best_j, best_corr = None, float("inf")
            for j, mj in enumerate(metric_names):
                if i != j and correlation[i][j] < best_corr:
                    best_corr = correlation[i][j]
                    best_j = j
            mj = metric_names[best_j]
            pair = tuple(sorted((mi, mj)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            margins_x = [scores[mi][a] - scores[mi][b]
                         for a, b in itertools.combinations(range(len(datasets)), 2)]
            margins_y = [scores[mj][a] - scores[mj][b]
                         for a, b in itertools.combinations(range(len(datasets)), 2)]

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(margins_x, margins_y, alpha=0.4, s=12)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.set_xlabel(f"{mi} margin")
            ax.set_ylabel(f"{mj} margin")
            ax.set_title(f"{mi} vs {mj}\n(Pearson r = {best_corr:.3f})")
            fig.tight_layout()
            fname = f"figures/margin_scatter_{mi.replace('@','').replace('-','_')}_vs_{mj.replace('@','').replace('-','_')}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"Saved: {fname}")
