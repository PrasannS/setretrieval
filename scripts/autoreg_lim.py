"""
Parallel Token Prediction Experiment
=====================================
Hypothesis: Can a causal LM predict N tokens in parallel (via pad tokens) as well as autoregressively?

Tasks:
  Task 1: [w1, w2, ..., wN] -> [w1, w2, ..., wN]  (repeat in order)
  Task 2: [w1, w2, ..., wN] -> shuffled order      (repeat in random order)

Inference modes:
  - Autoregressive: model sees [INPUT w1..wN] and generates outputs one by one
  - Parallel:       model sees [INPUT w1..wN] + [PAD x N] and must predict all N outputs at once

We train a GPT-2 style model from scratch (small) to isolate the mechanism, then evaluate
exact-match accuracy as a function of N (sequence length) and number of layers.
"""

import os
import random
import argparse
import itertools
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

import nltk
from nltk.corpus import wordnet

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Download WordNet ────────────────────────────────────────────────────────
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ════════════════════════════════════════════════════════════════════════════
# 1.  VOCABULARY / TOKENIZER
# ════════════════════════════════════════════════════════════════════════════

def build_wordnet_vocab(max_words: int = 20000, pretrained_tokenizer=None) -> List[str]:
    """Return a sorted list of short, common English WordNet lemmas.

    If pretrained_tokenizer is given, filter to words that are single tokens
    in that tokenizer (avoids subword complexity with pretrained models).
    """
    words = set()
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            w = lemma.name().replace("_", "").lower()
            if w.isalpha() and 3 <= len(w) <= 8:
                words.add(w)
    words = sorted(words)[:max_words]

    if pretrained_tokenizer is not None:
        filtered = []
        for w in words:
            token_ids = pretrained_tokenizer.encode(w, add_special_tokens=False)
            if len(token_ids) == 1:
                filtered.append(w)
        print(f"[vocab] Filtered {len(words)} -> {len(filtered)} single-token words")
        words = filtered

    return words


def build_tokenizer(vocab_words: List[str]) -> PreTrainedTokenizerFast:
    """
    Build a word-level tokenizer with fully deterministic token IDs.

    Special token layout (fixed, verified by assertion):
        0: [PAD]   1: [UNK]   2: [BOS]   3: [EOS]   4: [SEP]
    Word tokens occupy indices 5 ... 5+len(vocab_words)-1.

    Key design decisions to avoid the vocab-size mismatch / OOB embedding bug:
    - ALL tokens (specials + words) are inserted into the WordLevel vocab dict
      before constructing PreTrainedTokenizerFast, so IDs are set by us, not HF.
    - We do NOT call add_special_tokens() or add_tokens() afterwards; doing so
      would silently append extra entries and inflate len(tokenizer) beyond
      vocab_size, causing the embedding-table OOB CUDA assertion.
    - We assert the exact IDs and vocab_size after construction so any future
      regression crashes loudly at startup, not deep in a CUDA kernel.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIALS)}
    for i, w in enumerate(vocab_words):
        vocab[w] = len(SPECIALS) + i

    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_obj.pre_tokenizer = Whitespace()
    # No post-processor: BOS/SEP/EOS are inserted manually in __getitem__.

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        model_max_length=1024,
    )

    # ── Hard assertions: crash early if IDs are wrong ────────────────────
    assert hf_tok.pad_token_id == 0, f"Expected PAD=0, got {hf_tok.pad_token_id}"
    assert hf_tok.unk_token_id == 1, f"Expected UNK=1, got {hf_tok.unk_token_id}"
    assert hf_tok.bos_token_id == 2, f"Expected BOS=2, got {hf_tok.bos_token_id}"
    assert hf_tok.eos_token_id == 3, f"Expected EOS=3, got {hf_tok.eos_token_id}"
    assert hf_tok.sep_token_id == 4, f"Expected SEP=4, got {hf_tok.sep_token_id}"

    # vocab_size must equal our dict — no hidden additions allowed
    expected_size = len(SPECIALS) + len(vocab_words)
    assert hf_tok.vocab_size == expected_size, (
        f"vocab_size mismatch: got {hf_tok.vocab_size}, expected {expected_size}. "
        "PreTrainedTokenizerFast silently added tokens."
    )

    # Spot-check that word token IDs are in-range and not [UNK]
    unk_id = hf_tok.unk_token_id
    for w in vocab_words[:20]:
        wid = hf_tok.convert_tokens_to_ids(w)
        assert wid != unk_id, f"Word '{w}' unexpectedly maps to [UNK]"
        assert wid < hf_tok.vocab_size, (
            f"Word '{w}' id={wid} >= vocab_size={hf_tok.vocab_size}"
        )

    print(
        f"[tokenizer] vocab_size={hf_tok.vocab_size}  "
        f"PAD={hf_tok.pad_token_id} UNK={hf_tok.unk_token_id} "
        f"BOS={hf_tok.bos_token_id} EOS={hf_tok.eos_token_id} "
        f"SEP={hf_tok.sep_token_id}"
    )
    return hf_tok


# ════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    n_values: List[int]        = field(default_factory=lambda: [2, 4, 8, 16, 32])
    n_layers_values: List[int] = field(default_factory=lambda: [1, 2, 4, 6])
    n_heads: int               = 4
    d_model: int               = 128
    d_ff: int                  = 512
    task: str                  = "task1"  # "task1" (copy) | "task2" (shuffle)
    train_samples_per_n: int   = 4000
    eval_samples_per_n: int    = 500
    output_dir: str            = "./results"
    epochs: int                = 10
    batch_size: int            = 64
    lr: float                  = 3e-4


class ParallelPredictionDataset(Dataset):
    """
    Sequence layout
    ───────────────
    Parallel mode (parallel_mode=True):
        input_ids : [BOS] w1...wN [SEP] [PAD]xN [EOS]
        labels    :  -100  -100    -100   t1..tN  eos

    Autoregressive mode (parallel_mode=False):
        input_ids : [BOS] w1...wN [SEP] t1...tN [EOS]
        labels    :  -100  -100    -100   t1..tN  eos

    Loss is only computed on the target segment (labels != -100).
    """

    def __init__(
        self,
        vocab_words: List[str],
        tokenizer,
        n: int,
        num_samples: int,
        task: str = "task1",
        parallel_mode: bool = True,
        split: str = "train",
        is_pretrained: bool = False,
    ):
        self.tokenizer     = tokenizer
        self.n             = n
        self.task          = task
        self.parallel_mode = parallel_mode
        self.is_pretrained = is_pretrained

        rng = random.Random(SEED + abs(hash(split)) % 10_000)
        self.samples: List[List[str]] = [
            rng.sample(vocab_words, n) for _ in range(num_samples)
        ]

        # Pre-verify every word in the vocab has a valid in-range ID
        unk_id = tokenizer.unk_token_id
        for w in vocab_words:
            wid = self._encode(w)
            if wid == unk_id:
                raise ValueError(
                    f"Word '{w}' -> id={wid} is [UNK]. "
                    "Tokenizer construction is broken."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, w: str) -> int:
        if self.is_pretrained:
            ids = self.tokenizer.encode(w, add_special_tokens=False)
            assert len(ids) == 1, f"Word '{w}' is not a single token: {ids}"
            return ids[0]
        return self.tokenizer.convert_tokens_to_ids(w)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tok   = self.tokenizer
        words = self.samples[idx]

        if self.task == "task1":
            targets = list(words)
        else:  # task2: per-example deterministic shuffle
            rng = random.Random(SEED + idx)
            targets = list(words)
            rng.shuffle(targets)

        bos = tok.bos_token_id   # 2
        sep = tok.sep_token_id   # 4
        eos = tok.eos_token_id   # 3
        pad = tok.pad_token_id   # 0

        input_word_ids = [self._encode(w) for w in words]
        target_ids     = [self._encode(w) for w in targets]

        # Prefix is [BOS] + N words + [SEP] => N+2 positions ignored in loss
        prefix_len = 1 + self.n + 1
        ignore     = [-100] * prefix_len

        if self.parallel_mode:
            # Input:  [BOS] inputs [SEP] [PAD]xN [EOS]
            # Labels: ignore        ignore  tgts   eos
            input_ids = [bos] + input_word_ids + [sep] + [pad] * self.n + [eos]
            labels    = ignore + target_ids + [eos]
        else:
            # Input:  [BOS] inputs [SEP] tgts [EOS]
            # Labels: ignore        ignore tgts eos
            input_ids = [bos] + input_word_ids + [sep] + target_ids + [eos]
            labels    = ignore + target_ids + [eos]

        assert len(input_ids) == len(labels), (
            f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}"
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════════════════
# 3.  MODEL FACTORY
# ════════════════════════════════════════════════════════════════════════════

def make_model(
    tokenizer: PreTrainedTokenizerFast,
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_ff: int,
) -> GPT2LMHeadModel:
    """
    Build a GPT-2 model whose vocab_size and special-token IDs are derived
    directly from the tokenizer -- never hardcoded.

    n_positions=512: a safe fixed upper bound for all N values we test.
    GPT-2 position embeddings are a fixed table of size n_positions; any
    sequence longer than this would crash with an OOB index. 512 is large
    enough for our largest N=32 (max seq = 1+32+1+32+1 = 67 tokens) with
    ample headroom, and costs negligible memory at our d_model=128.
    """
    vocab_size = tokenizer.vocab_size  # guaranteed correct by build_tokenizer

    config = GPT2Config(
        vocab_size   = vocab_size,
        n_positions  = 512,            # fixed safe upper bound
        n_embd       = d_model,
        n_layer      = n_layers,
        n_head       = n_heads,
        n_inner      = d_ff,
        resid_pdrop  = 0.0,
        embd_pdrop   = 0.0,
        attn_pdrop   = 0.0,
        bos_token_id = tokenizer.bos_token_id,  # 2
        eos_token_id = tokenizer.eos_token_id,  # 3
        pad_token_id = tokenizer.pad_token_id,  # 0
    )

    model = GPT2LMHeadModel(config)
    model.apply(model._init_weights)

    # Verify embedding table size matches vocab_size
    emb_size = model.transformer.wte.weight.shape[0]
    assert emb_size == vocab_size, (
        f"Embedding table size {emb_size} != vocab_size {vocab_size}"
    )
    return model


def load_pretrained_model(model_name: str, device: torch.device):
    """Load a pretrained causal LM and its tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure pad token exists (e.g. GPT-2 has no pad token by default)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # Ensure sep token exists
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
        model.resize_token_embeddings(len(tokenizer))

    # Ensure bos token exists
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "[BOS]"})
        model.resize_token_embeddings(len(tokenizer))

    # Ensure eos token exists
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        model.resize_token_embeddings(len(tokenizer))

    print(
        f"[pretrained] model={model_name}  vocab_size={len(tokenizer)}  "
        f"PAD={tokenizer.pad_token_id} BOS={tokenizer.bos_token_id} "
        f"EOS={tokenizer.eos_token_id} SEP={tokenizer.sep_token_id}"
    )

    return model.to(device), tokenizer


# ════════════════════════════════════════════════════════════════════════════
# 3b. DEBUG PREDICTION PRINTING
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def print_debug_predictions(
    model,
    dataset: ParallelPredictionDataset,
    tokenizer,
    n: int,
    mode: str,
    device: torch.device,
    num_samples: int = 5,
) -> None:
    """Print example predictions for debugging."""
    if num_samples <= 0:
        return

    model.eval()
    sep_id = tokenizer.sep_token_id
    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id

    print(f"\n  --- Debug predictions (mode={mode}, N={n}) ---")
    for idx in range(min(num_samples, len(dataset))):
        words = dataset.samples[idx]

        if dataset.task == "task1":
            targets = list(words)
        else:
            rng = random.Random(SEED + idx)
            targets = list(words)
            rng.shuffle(targets)

        if mode == "autoregressive":
            # Greedy AR decode
            prefix = [bos_id] + [dataset._encode(w) for w in words] + [sep_id]
            input_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            generated = []
            for _ in range(n):
                out = model(input_ids=input_tensor)
                next_tok = int(out.logits[0, -1].argmax())
                generated.append(next_tok)
                input_tensor = torch.cat(
                    [input_tensor, torch.tensor([[next_tok]], device=device)], dim=1
                )
            pred_words = [tokenizer.decode([g]).strip() for g in generated]
        else:
            # Parallel: single forward pass
            item = dataset[idx]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            out = model(input_ids=input_ids)
            logits = out.logits[0]
            # Target positions start after prefix (BOS + N words + SEP)
            prefix_len = 1 + n + 1
            pred_ids = logits[prefix_len - 1 : prefix_len - 1 + n].argmax(dim=-1).tolist()
            pred_words = [tokenizer.decode([p]).strip() for p in pred_ids]

        print(f"    [{idx}] input:    {words}")
        print(f"         expected: {targets}")
        print(f"         predicted: {pred_words}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# 4.  COLLATOR
# ════════════════════════════════════════════════════════════════════════════

def pad_collate(batch: List[Dict], pad_id: int, max_pos: int = 512) -> Dict[str, torch.Tensor]:
    """
    Right-pad a batch to the longest sequence in the batch.
    Clips to max_pos (= model n_positions) to prevent position OOB.
    """
    max_len = min(max(b["input_ids"].size(0) for b in batch), max_pos)
    bs = len(batch)
    input_ids      = torch.full((bs, max_len), pad_id, dtype=torch.long)
    labels         = torch.full((bs, max_len), -100,   dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len,           dtype=torch.long)
    for i, b in enumerate(batch):
        l = min(b["input_ids"].size(0), max_len)
        input_ids[i, :l]      = b["input_ids"][:l]
        labels[i, :l]         = b["labels"][:l]
        attention_mask[i, :l] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ════════════════════════════════════════════════════════════════════════════
# 5.  METRICS
# ════════════════════════════════════════════════════════════════════════════

def _unordered_match(pred_tgt: np.ndarray, true_tgt: np.ndarray) -> int:
    """Count how many predicted tokens can be matched to target tokens (multiset intersection)."""
    remaining = list(true_tgt)
    matched = 0
    for p in pred_tgt:
        if p in remaining:
            remaining.remove(p)
            matched += 1
    return matched


def compute_exact_match(
    predictions: np.ndarray, labels: np.ndarray, unordered: bool = False
) -> Dict[str, float]:
    """Token-level and sequence-level exact match on the target segment only.

    If unordered=True, uses multiset matching (order doesn't matter).
    """
    token_correct = token_total = seq_correct = seq_total = 0
    for i in range(predictions.shape[0]):
        mask     = labels[i] != -100
        pred_tgt = predictions[i][mask]
        true_tgt = labels[i][mask]
        if len(true_tgt) == 0:
            continue
        if unordered:
            n_ok = _unordered_match(pred_tgt, true_tgt)
        else:
            n_ok = int((pred_tgt == true_tgt).sum())
        token_correct += n_ok
        token_total   += len(true_tgt)
        seq_correct   += int(n_ok == len(true_tgt))
        seq_total     += 1
    return {
        "token_accuracy":    token_correct / token_total if token_total else 0.0,
        "sequence_accuracy": seq_correct   / seq_total   if seq_total   else 0.0,
    }


def make_compute_metrics(_pad_id: int, task: str = "task1"):
    unordered = (task == "task2")
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        # Causal LM: logits at position i predict token at position i+1.
        # Shift so predictions align with labels.
        predictions = np.argmax(logits[:, :-1, :], axis=-1)
        labels = labels[:, 1:]
        return compute_exact_match(predictions, labels, unordered=unordered)
    return compute_metrics


# ════════════════════════════════════════════════════════════════════════════
# 6.  AUTOREGRESSIVE INFERENCE EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_autoregressive(
    model,
    dataset: ParallelPredictionDataset,
    tokenizer,
    n: int,
    device: torch.device,
    max_samples: int = 200,
) -> Dict[str, float]:
    """
    Greedy autoregressive decode for n steps after [SEP].
    Works regardless of which training mode the model used.
    """
    model.eval()
    sep_id = tokenizer.sep_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id

    token_correct = token_total = seq_correct = seq_total = 0

    for idx in range(min(max_samples, len(dataset))):
        words = dataset.samples[idx]

        if dataset.task == "task1":
            targets = list(words)
        else:
            rng = random.Random(SEED + idx)
            targets = list(words)
            rng.shuffle(targets)

        target_ids = [dataset._encode(w) for w in targets]

        # Build prefix: [BOS] w1..wN [SEP]
        prefix = [bos_id] + [dataset._encode(w) for w in words] + [sep_id]
        input_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

        generated: List[int] = []
        for _ in range(n):
            out      = model(input_ids=input_tensor)
            next_tok = int(out.logits[0, -1].argmax())
            generated.append(next_tok)
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_tok]], device=device)], dim=1
            )
            if next_tok == eos_id:
                break

        # Exactly n tokens: pad if model stopped early, trim if it ran over
        generated = (generated + [pad_id] * n)[:n]

        if dataset.task == "task2":
            # Unordered multiset match for task2
            remaining = list(target_ids)
            n_ok = 0
            for g in generated:
                if g in remaining:
                    remaining.remove(g)
                    n_ok += 1
        else:
            n_ok = sum(g == t for g, t in zip(generated, target_ids))
        token_correct += n_ok
        token_total   += n
        seq_correct   += int(n_ok == n)
        seq_total     += 1

    return {
        "ar_token_accuracy":    token_correct / token_total if token_total else 0.0,
        "ar_sequence_accuracy": seq_correct   / seq_total   if seq_total   else 0.0,
    }


# ════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_single_experiment(
    cfg: ExperimentConfig,
    n: int,
    n_layers: int,
    mode: str,              # "parallel" | "autoregressive"
    vocab_words: List[str],
    tokenizer,
    device: torch.device,
    pretrained_model=None,
    debug_samples: int = 0,
) -> Dict[str, float]:
    """Train and evaluate one (N, n_layers, mode) cell."""

    parallel_mode = (mode == "parallel")
    is_pretrained = pretrained_model is not None

    print(f"\n{'='*60}")
    print(f"  Task={cfg.task}  N={n}  Layers={n_layers}  Mode={mode}")
    print(f"{'='*60}")

    train_ds = ParallelPredictionDataset(
        vocab_words, tokenizer, n,
        num_samples=cfg.train_samples_per_n,
        task=cfg.task, parallel_mode=parallel_mode, split="train",
        is_pretrained=is_pretrained,
    )
    eval_ds = ParallelPredictionDataset(
        vocab_words, tokenizer, n,
        num_samples=cfg.eval_samples_per_n,
        task=cfg.task, parallel_mode=parallel_mode, split="eval",
        is_pretrained=is_pretrained,
    )

    if pretrained_model is not None:
        import copy
        model = copy.deepcopy(pretrained_model).to(device)
    else:
        model = make_model(
            tokenizer=tokenizer,
            n_layers=n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
        ).to(device)

    model_tag = "pretrained" if is_pretrained else f"L{n_layers}"
    run_dir = os.path.join(
        cfg.output_dir, "runs",
        f"{cfg.task}_N{n}_{model_tag}_{mode}"
    )
    os.makedirs(run_dir, exist_ok=True)

    pad_id   = tokenizer.pad_token_id
    max_pos  = getattr(model.config, "n_positions",
               getattr(model.config, "max_position_embeddings", 1024))
    collator = lambda batch, _mp=max_pos: pad_collate(batch, pad_id, max_pos=_mp)

    training_args = TrainingArguments(
        output_dir                  = run_dir,
        num_train_epochs            = cfg.epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        learning_rate               = cfg.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        eval_strategy               = "epoch",
        save_strategy               = "no",
        logging_steps               = 50,
        report_to                   = "none",
        seed                        = SEED,
        dataloader_num_workers      = 0,
        fp16                        = (device.type == "cuda"),
        label_names                 = ["labels"],
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        data_collator   = collator,
        compute_metrics = make_compute_metrics(pad_id, task=cfg.task),
    )

    trainer.train()

    # ── Parallel-slot eval (HF Trainer, one forward pass per batch) ──────
    eval_results = trainer.evaluate()
    results = {
        "parallel_token_acc": eval_results.get("eval_token_accuracy",    float("nan")),
        "parallel_seq_acc":   eval_results.get("eval_sequence_accuracy", float("nan")),
    }

    # ── Greedy AR eval (always runs, regardless of training mode) ────────
    ar_eval_ds = ParallelPredictionDataset(
        vocab_words, tokenizer, n,
        num_samples=cfg.eval_samples_per_n,
        task=cfg.task, parallel_mode=False, split="eval",
        is_pretrained=is_pretrained,
    )
    if mode == "autoregressive":
        ar_metrics = evaluate_autoregressive(model, ar_eval_ds, tokenizer, n, device)
        results.update(ar_metrics)

    print(f"  Results: {results}")

    # Debug predictions
    if debug_samples > 0:
        print_debug_predictions(
            model, eval_ds, tokenizer, n, mode, device, num_samples=debug_samples
        )
        if mode == "parallel":
            # Also show AR predictions for comparison
            print_debug_predictions(
                model, ar_eval_ds, tokenizer, n, "autoregressive", device,
                num_samples=debug_samples,
            )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ════════════════════════════════════════════════════════════════════════════
# 8.  ANALYSIS & PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_results(df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    modes = df["mode"].unique()
    metric_labels = {
        "parallel_token_acc":   "Parallel Token Acc",
        "parallel_seq_acc":     "Parallel Seq Acc",
        "ar_token_accuracy":    "AR Token Acc",
        "ar_sequence_accuracy": "AR Seq Acc",
    }

    # ── Plot 1: Accuracy vs N (one panel per layer count) ─────────────────
    for metric in ("parallel_seq_acc", "ar_sequence_accuracy"):
        ncols = len(cfg.n_layers_values)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
        if ncols == 1:
            axes = [axes]
        for ax, nl in zip(axes, cfg.n_layers_values):
            sub = df[df["n_layers"] == nl]
            for mode in modes:
                s = sub[sub["mode"] == mode].sort_values("n")
                ax.plot(s["n"], s[metric], marker="o", label=mode)
            ax.set_title(f"Layers={nl}")
            ax.set_xlabel("N (sequence length)")
            ax.set_ylabel(metric_labels[metric])
            ax.legend(fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Task: {cfg.task.upper()} | {metric_labels[metric]} vs N", fontsize=13)
        plt.tight_layout()
        path = os.path.join(cfg.output_dir, f"{cfg.task}_{metric}_vs_N.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    # ── Plot 2: Accuracy vs Layers (one panel per N) ──────────────────────
    for metric in ("parallel_seq_acc", "ar_sequence_accuracy"):
        ncols = len(cfg.n_values)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
        if ncols == 1:
            axes = [axes]
        for ax, nv in zip(axes, cfg.n_values):
            sub = df[df["n"] == nv]
            for mode in modes:
                s = sub[sub["mode"] == mode].sort_values("n_layers")
                ax.plot(s["n_layers"], s[metric], marker="s", label=mode)
            ax.set_title(f"N={nv}")
            ax.set_xlabel("Number of Layers")
            ax.set_ylabel(metric_labels[metric])
            ax.legend(fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Task: {cfg.task.upper()} | {metric_labels[metric]} vs Layers", fontsize=13)
        plt.tight_layout()
        path = os.path.join(cfg.output_dir, f"{cfg.task}_{metric}_vs_layers.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    # ── Plot 3: Heatmap N x Layers ────────────────────────────────────────
    for mode in modes:
        for metric in ("parallel_seq_acc", "ar_sequence_accuracy"):
            sub   = df[df["mode"] == mode]
            pivot = sub.pivot_table(index="n_layers", columns="n", values=metric)
            nr, nc = len(pivot.index), len(pivot.columns)
            fig, ax = plt.subplots(figsize=(max(5, nc * 1.2), max(3, nr)))
            im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                           vmin=0, vmax=1, cmap="RdYlGn")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(nc))
            ax.set_xticklabels([str(c) for c in pivot.columns])
            ax.set_yticks(range(nr))
            ax.set_yticklabels([str(r) for r in pivot.index])
            ax.set_xlabel("N (sequence length)")
            ax.set_ylabel("Num Layers")
            ax.set_title(f"Task={cfg.task.upper()} | Mode={mode} | {metric_labels[metric]}")
            for i in range(nr):
                for j in range(nc):
                    ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                            ha="center", va="center", fontsize=8)
            plt.tight_layout()
            path = os.path.join(cfg.output_dir, f"{cfg.task}_heatmap_{mode}_{metric}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Saved: {path}")


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False, float_format="{:.3f}".format))


# ════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Token Prediction Experiment")
    parser.add_argument("--task",            choices=["task1", "task2", "both"], default="task1")
    parser.add_argument("--n_values",        type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--n_layers_values", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--n_heads",         type=int,   default=4)
    parser.add_argument("--d_model",         type=int,   default=128)
    parser.add_argument("--d_ff",            type=int,   default=512)
    parser.add_argument("--train_samples",   type=int,   default=4000)
    parser.add_argument("--eval_samples",    type=int,   default=500)
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--output_dir",      type=str,   default="./results")
    parser.add_argument("--modes",
                        choices=["parallel", "autoregressive", "both"], default="both",
                        help="Training mode: parallel-slot, autoregressive, or both")
    parser.add_argument("--pretrained",      type=str, default="answerdotai/ModernBERT-large",
                        help="HuggingFace model ID (e.g. 'gpt2', 'gpt2-medium'). "
                             "When set, uses a pretrained model instead of training from scratch.")
    parser.add_argument("--debug_samples",   type=int, default=0,
                        help="Number of debug prediction examples to print (0 to disable)")
    args = parser.parse_args()

    tasks = ["task1", "task2"] if args.task == "both" else [args.task]
    modes = ["parallel", "autoregressive"] if args.modes == "both" else [args.modes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pretrained_model = None

    if args.pretrained:
        # Pretrained mode: load model and tokenizer from HuggingFace
        pretrained_model, tokenizer = load_pretrained_model(args.pretrained, device)
        vocab_words = build_wordnet_vocab(max_words=200000, pretrained_tokenizer=tokenizer)
        print(f"Vocab size (single-token words): {len(vocab_words)}")

        # Get actual layer count from the model config for sweep/plot compatibility
        model_config = pretrained_model.config
        actual_layers = getattr(model_config, "n_layer",
                        getattr(model_config, "num_hidden_layers", 1))
        n_layers_values = [actual_layers]
        print(f"[pretrained] Using actual layer count: {actual_layers}")
    else:
        # From-scratch mode: build custom tokenizer
        vocab_words = build_wordnet_vocab(max_words=200000)
        tokenizer   = build_tokenizer(vocab_words)
        print(f"Vocab size: {tokenizer.vocab_size}")
        n_layers_values = args.n_layers_values

    all_rows: List[Dict] = []

    for task in tasks:
        cfg = ExperimentConfig(
            n_values            = args.n_values,
            n_layers_values     = n_layers_values,
            n_heads             = args.n_heads,
            d_model             = args.d_model,
            d_ff                = args.d_ff,
            task                = task,
            train_samples_per_n = args.train_samples,
            eval_samples_per_n  = args.eval_samples,
            output_dir          = os.path.join(args.output_dir, task),
            epochs              = args.epochs,
            batch_size          = args.batch_size,
            lr                  = args.lr,
        )

        for n, n_layers, mode in itertools.product(cfg.n_values, cfg.n_layers_values, modes):
            res = run_single_experiment(
                cfg, n, n_layers, mode, vocab_words, tokenizer, device,
                pretrained_model=pretrained_model,
                debug_samples=args.debug_samples,
            )
            all_rows.append({"task": task, "n": n, "n_layers": n_layers, "mode": mode, **res})

        task_df = pd.DataFrame([r for r in all_rows if r["task"] == task])
        plot_results(task_df, cfg)

    df = pd.DataFrame(all_rows)
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "all_results.csv")
    df.to_csv(results_path, index=False)
    print(f"\nAll results saved to: {results_path}")
    print_summary_table(df)


if __name__ == "__main__":
    main()