"""Training script for auto-regressive multi-vector retrieval (AutoReg-ColBERT).

Each document is encoded by:
  1. Generating n_rollout_tokens using the LM head (vLLM-style rollout, simplified here
     to HF generate() — swap in vLLM for production speed).
  2. Encoding [original_doc | rollout_tokens] with a ColBERT projection head.

Queries are encoded with standard ColBERT (pad-tokens, no rollout).

Loss:
  - Contrastive (in-batch negatives) on ColBERT MaxSim scores
  - REINFORCE: weight log-prob of rollout tokens by retrieval margin reward

Toy dataset:
  query     = one word from a vocabulary of size V
  positive  = doc of length L that CONTAINS the query word
  negative  = doc of length L that does NOT contain it

Usage:
  # Toy test (CPU/small GPU, uses GPT-2 by default):
  python scripts/train_autoreg_colbert.py --dataset toy

  # Real data (MSMARCO):
  python scripts/train_autoreg_colbert.py \\
      --dataset msmarco \\
      --model_name meta-llama/Llama-3.2-1B \\
      --embedding_size 128 --n_rollout_tokens 32 --batch_size 16
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from setretrieval.train.autoreg_colbert_model import AutoRegColBERT


# ── ColBERT scoring ───────────────────────────────────────────────────────────

def colbert_scores(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    query_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    All-pairs MaxSim ColBERT scores.

    query_embs : (Q, q_len, E)
    doc_embs   : (D, d_len, E)
    query_mask : (Q, q_len) bool/float — 1 for real tokens, 0 for padding.
                 Padding positions are excluded from the query-side sum.
    Returns    : (Q, D)
    """
    sim = torch.einsum("ash,bth->abst", query_embs, doc_embs)  # (Q, D, q_len, d_len)
    maxsim = sim.max(dim=-1).values                             # (Q, D, q_len)
    if query_mask is not None:
        # zero out padding positions before summing
        maxsim = maxsim * query_mask.float().unsqueeze(1)       # broadcast over D
    return maxsim.sum(dim=-1)                                   # (Q, D)


# ── Loss functions ────────────────────────────────────────────────────────────

def contrastive_loss(score_matrix: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """In-batch contrastive loss; diagonal entries are positive pairs."""
    labels = torch.arange(len(score_matrix), device=score_matrix.device)
    return F.cross_entropy(score_matrix / temperature, labels)


def reinforce_loss(log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """
    REINFORCE policy-gradient loss.

    log_probs : (B, n_rollout) — log-prob of each generated token
    rewards   : (B,)           — scalar reward (higher = better rollout)

    We normalize rewards within the batch for variance reduction (simple baseline).
    Gradient: increases log-prob of tokens that led to above-average rewards.
    """
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return -(log_probs.sum(dim=-1) * adv.detach()).mean()


# ── Toy dataset ───────────────────────────────────────────────────────────────

def make_toy_dataset(
    n: int = 2000,
    vocab_size: int = 64,
    doc_len: int = 30,
    seed: int = 42,
) -> Dataset:
    """
    Single-word query retrieval task:
      query    = one word from a vocabulary of <vocab_size> synthetic words
      positive = a document of <doc_len> words that CONTAINS the query word
      negative = a document of <doc_len> words that does NOT contain it

    A model that learns to compute ColBERT embeddings correctly should achieve
    near-100% accuracy on this task (the query token will match its exact copy
    in the positive document via MaxSim).  The rollout can additionally help by
    generating tokens that summarize document content.
    """
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab_size)]
    rows = []
    for _ in range(n):
        q = rng.choice(words)
        rest = [w for w in words if w != q]
        # positive: contains q at a random position
        pos = rng.choices(rest, k=doc_len)
        pos.insert(rng.randint(0, doc_len), q)
        # negative: never contains q
        neg = rng.choices(rest, k=doc_len + 1)
        rows.append({"query": q, "positive": " ".join(pos), "negative": " ".join(neg)})
    return Dataset.from_list(rows)


# ── Tokenization helper ───────────────────────────────────────────────────────

def tokenize(tokenizer, texts, max_len, device):
    enc = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, dataset, args, device, n_eval: int = 200):
    """Triplet accuracy: fraction of pairs where score(q, pos) > score(q, neg)."""
    model.eval()
    correct = total = 0
    for i in range(0, min(n_eval, len(dataset)), args.batch_size):
        b = dataset[i : i + args.batch_size]
        q_ids, q_mask = tokenize(tokenizer, b["query"],    args.query_len, device)
        p_ids, p_mask = tokenize(tokenizer, b["positive"], args.doc_len,   device)
        n_ids, n_mask = tokenize(tokenizer, b["negative"], args.doc_len,   device)

        q_embs        = model.encode_query(q_ids, q_mask)
        p_embs, _     = model.encode_doc_with_rollout(p_ids, p_mask)
        n_embs, _     = model.encode_doc_with_rollout(n_ids, n_mask)

        B  = q_embs.size(0)
        ps = colbert_scores(q_embs, p_embs, q_mask)[range(B), range(B)]
        ns = colbert_scores(q_embs, n_embs, q_mask)[range(B), range(B)]
        correct += (ps > ns).sum().item()
        total   += B

    print(f"  Eval acc = {correct}/{total} = {correct / max(total, 1):.3f}", flush=True)
    model.train()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── dataset ───────────────────────────────────────────────────────────────
    if args.dataset == "toy":
        train_ds = make_toy_dataset(args.toy_n, args.toy_vocab, args.toy_doc_len, seed=42)
        eval_ds  = make_toy_dataset(300, args.toy_vocab, args.toy_doc_len, seed=99)
    elif args.dataset == "msmarco":
        ds = load_dataset(
            "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
            "triplet-hard",
            split="train",
        )
        sp       = ds.train_test_split(test_size=500, seed=42)
        train_ds = sp["train"].select(range(min(100_000, len(sp["train"]))))
        eval_ds  = sp["test"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ── model ─────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoRegColBERT(
        model_name=args.model_name,
        embedding_size=args.embedding_size,
        n_rollout_tokens=args.n_rollout_tokens,
    ).to(device)

    # Separate LRs: projection head trains fast, backbone trains slow.
    # This prevents destroying pretrained representations while still
    # allowing the backbone to adapt to retrieval over time.
    optimizer = AdamW(
        [
            {"params": model.projection.parameters(), "lr": args.proj_lr},
            {"params": model.backbone.parameters(),   "lr": args.backbone_lr},
        ],
        weight_decay=1e-2,
    )
    print(f"LRs — backbone: {args.backbone_lr}  projection: {args.proj_lr}", flush=True)

    # ── train ─────────────────────────────────────────────────────────────────
    train_ds = train_ds.shuffle(seed=0)
    for epoch in range(args.epochs):
        for step in range(0, len(train_ds), args.batch_size):
            b = train_ds[step : step + args.batch_size]
            if len(b["query"]) < 2:
                continue

            q_ids, q_mask = tokenize(tokenizer, b["query"],    args.query_len, device)
            p_ids, p_mask = tokenize(tokenizer, b["positive"], args.doc_len,   device)
            n_ids, n_mask = tokenize(tokenizer, b["negative"], args.doc_len,   device)

            # Encode
            q_embs      = model.encode_query(q_ids, q_mask)               # (B, qL, E)
            p_embs, p_lp = model.encode_doc_with_rollout(p_ids, p_mask)   # (B, dL+R, E), (B, R)
            n_embs, n_lp = model.encode_doc_with_rollout(n_ids, n_mask)

            B = q_embs.size(0)

            # In-batch contrastive loss (queries vs positive docs only)
            score_mat = colbert_scores(q_embs, p_embs, q_mask)  # (B, B)
            l_contrast = contrastive_loss(score_mat, args.temp)

            # Diagonal scores for the reward signal
            p_sc = score_mat[range(B), range(B)]                         # (B,)
            n_sc = colbert_scores(q_embs, n_embs, q_mask)[range(B), range(B)]  # (B,)
            reward = (p_sc - n_sc).detach()                              # (B,)

            # REINFORCE: reward positive rollouts, penalise negative rollouts
            l_rl = reinforce_loss(p_lp, reward) + reinforce_loss(n_lp, -reward)

            loss = l_contrast + args.rl_weight * l_rl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % (10 * args.batch_size) == 0:
                print(
                    f"Ep {epoch}  step {step // args.batch_size:4d}  "
                    f"loss={loss.item():.4f}  "
                    f"contrast={l_contrast.item():.4f}  "
                    f"rl={l_rl.item():.4f}  "
                    f"reward={reward.mean().item():.3f}",
                    flush=True,
                )

        evaluate(model, tokenizer, eval_ds, args, device)

    # ── save ──────────────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(model.projection.state_dict(), out / "projection.pt")
    print(f"Saved to {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--model_name",       type=str,   default="gpt2",
                   help="Any HF CausalLM (gpt2, Llama-3.2-1B, Mistral-7B, …)")
    p.add_argument("--embedding_size",   type=int,   default=64)
    p.add_argument("--n_rollout_tokens", type=int,   default=8,
                   help="Number of tokens generated per document during rollout")
    # Data
    p.add_argument("--dataset",    type=str, default="toy", choices=["toy", "msmarco"])
    p.add_argument("--query_len",  type=int, default=16)
    p.add_argument("--doc_len",    type=int, default=64)
    # Training
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--backbone_lr", type=float, default=1e-5,
                   help="LR for the pretrained backbone (keep small to preserve representations)")
    p.add_argument("--proj_lr",    type=float, default=1e-3,
                   help="LR for the ColBERT projection head (can be much larger)")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--temp",       type=float, default=0.05,
                   help="Contrastive loss temperature")
    p.add_argument("--rl_weight",  type=float, default=0.05,
                   help="Weight for REINFORCE loss relative to contrastive loss")
    p.add_argument("--output_dir", type=str,   default="output/autoreg_colbert")
    # Toy dataset
    p.add_argument("--toy_n",       type=int, default=2000,
                   help="Number of training examples for toy dataset")
    p.add_argument("--toy_vocab",   type=int, default=64,
                   help="Vocabulary size for toy dataset")
    p.add_argument("--toy_doc_len", type=int, default=30,
                   help="Document length (words) for toy dataset")
    args = p.parse_args()
    train(args)
