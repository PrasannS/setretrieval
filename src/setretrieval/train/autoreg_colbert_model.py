"""Auto-regressive multi-vector (ColBERT) retrieval model.

Document encoding pipeline:
    1. _rollout: generate n_rollout_tokens auto-regressively with the LM head (no grad)
    2. Single differentiable forward pass on [original_doc | rollout_tokens]
       → hidden states   → projected + normalized ColBERT token embeddings
       → logits          → log-probs of rollout tokens (for REINFORCE)

Query encoding pipeline:
    Standard ColBERT: forward pass on padded query tokens, no rollout.

Training signal:
    - Contrastive loss  : differentiable through the projection head
    - REINFORCE loss    : differentiable through the LM head (log-prob weighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Tuple


class AutoRegColBERT(nn.Module):
    """
    Decoder-only LM augmented with a ColBERT projection head.

    Args:
        model_name       : HF model id (must be a CausalLM, e.g. gpt2, Llama-3.2-1B)
        embedding_size   : ColBERT token embedding dimension
        n_rollout_tokens : number of tokens to generate per document
        dtype            : dtype for the backbone (bfloat16 recommended on A100+)
    """

    def __init__(
        self,
        model_name: str,
        embedding_size: int = 128,
        n_rollout_tokens: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        )
        # Projection head always in float32: bfloat16 Adam updates lose small
        # increments (value + tiny_update == value in bf16), so training stalls.
        self.projection = nn.Linear(
            self.backbone.config.hidden_size, embedding_size, bias=False
        )  # float32
        self.n_rollout_tokens = n_rollout_tokens
        self.embedding_size = embedding_size
        self.pad_id = self.backbone.config.eos_token_id or 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states → L2-normalized ColBERT token embeddings."""
        return F.normalize(self.projection(hidden.float()), dim=-1)

    @torch.no_grad()
    def _rollout(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate n_rollout_tokens for every document in one batched generate() call.

        Strategy: convert right-padded inputs → left-padded (standard for causal-LM
        generation), run a single batched generate(), then return a right-padded
        expanded batch so the rest of the pipeline stays simple.

        Returns:
            expanded_ids   (B, max_true_len + n_rollout)
            expanded_mask  (B, same)
            orig_lens      (B,) — unpadded length of each original document
        """
        B, T = input_ids.shape
        orig_lens = attention_mask.sum(dim=-1)          # (B,)  int lengths
        max_true_len = int(orig_lens.max())

        # Build left-padded batch for generation
        left_ids  = input_ids.new_full((B, max_true_len), self.pad_id)
        left_mask = input_ids.new_zeros(B, max_true_len)
        for i in range(B):
            ol = int(orig_lens[i])
            left_ids[i,  max_true_len - ol:] = input_ids[i, :ol]
            left_mask[i, max_true_len - ol:] = 1

        # Single batched generate call — new tokens are appended on the right
        gen = self.backbone.generate(
            input_ids=left_ids,
            attention_mask=left_mask,
            max_new_tokens=self.n_rollout_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.pad_id,
        )  # (B, max_true_len + n_rollout)

        new_tokens = gen[:, max_true_len:]  # (B, n_rollout)

        # Rebuild right-padded expanded batch: [content | rollout | padding]
        result_len = int(orig_lens.max()) + self.n_rollout_tokens
        expanded_ids  = input_ids.new_zeros(B, result_len)
        expanded_mask = input_ids.new_zeros(B, result_len)
        for i in range(B):
            ol = int(orig_lens[i])
            expanded_ids[i,  :ol]                       = input_ids[i, :ol]
            expanded_ids[i,  ol : ol + self.n_rollout_tokens] = new_tokens[i]
            expanded_mask[i, :ol + self.n_rollout_tokens]     = 1

        return expanded_ids, expanded_mask, orig_lens

    def _forward_full(
        self, ids: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single backbone forward pass → (last_hidden_state, logits)."""
        out = self.backbone(
            input_ids=ids, attention_mask=mask, output_hidden_states=True
        )
        return out.hidden_states[-1], out.logits  # (B, T, H), (B, T, V)

    # ── public API ────────────────────────────────────────────────────────────

    def encode_query(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard ColBERT query encoding (no rollout).

        Returns normalized token embeddings (B, T, E) for all positions
        where attention_mask == 1.
        """
        hidden, _ = self._forward_full(input_ids, attention_mask)
        return self._project(hidden)  # (B, T, E)

    def encode_doc_with_rollout(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode document with LM rollout:
            1. Generate n_rollout_tokens (no grad, discrete sampling)
            2. Single differentiable forward pass on [doc | rollout]

        Returns:
            embeddings  (B, T_expanded, E)  — grad flows through projection head
            log_probs   (B, n_rollout)       — grad flows through LM head (REINFORCE)
        """
        # Step 1: rollout (no gradient)
        expanded_ids, expanded_mask, orig_lens = self._rollout(input_ids, attention_mask)

        # Step 2: single differentiable forward pass over full expanded sequence
        hidden, logits = self._forward_full(expanded_ids, expanded_mask)

        # ── ColBERT embeddings (all token positions) ─────────────────────────
        embeddings = self._project(hidden)  # (B, T_exp, E)

        # ── Log-probs of rollout tokens for REINFORCE ─────────────────────────
        # In a causal LM: token[t]'s probability = softmax(logits[t-1])[token[t]]
        # Rollout token i (0-indexed) sits at position orig_lens[b]+i in expanded_ids
        # Its predictor logit is at position orig_lens[b]+i-1 in logits
        lp_all = F.log_softmax(logits, dim=-1)  # (B, T_exp, V)
        log_prob_list = []
        for b in range(input_ids.size(0)):
            ol = orig_lens[b].item()
            # predictor positions: [ol-1 .. ol+n_rollout-2]
            pred_lp  = lp_all[b, ol - 1 : ol + self.n_rollout_tokens - 1, :]  # (n_rollout, V)
            gen_toks = expanded_ids[b, ol : ol + self.n_rollout_tokens]        # (n_rollout,)
            tok_lp   = pred_lp.gather(-1, gen_toks.unsqueeze(-1)).squeeze(-1)  # (n_rollout,)
            log_prob_list.append(tok_lp)
        log_probs = torch.stack(log_prob_list, dim=0)  # (B, n_rollout)

        return embeddings, log_probs
