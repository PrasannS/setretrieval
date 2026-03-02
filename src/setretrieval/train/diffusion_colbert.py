"""
Diffusion-style multi-forward-pass ColBERT training.

Each document is encoded with M forward passes. In each subsequent pass the
hidden states at the "pad" (extra [EMB]) token positions from the previous
pass are injected back as word-level input embeddings for those same
positions, while the context tokens remain unchanged.

Queries are encoded with a single forward pass as normal.

Two training modes
------------------
per_step_loss=False (default)
    Standard GradCache contrastive loss using only the final-pass document
    embeddings.  Intermediate passes still receive gradients through the
    backprop chain when backprop_all_passes=True.

per_step_loss=True
    Contrastive loss is accumulated at every forward pass step and summed.
    GradCache is NOT used in this mode (simpler but higher memory).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from pylate.losses.cached_contrastive import CachedContrastive, RandContext
from pylate.losses.contrastive import extract_skiplist_mask


# ---------------------------------------------------------------------------
# Backward hook (doc-aware version of pylate's _backward_hook)
# ---------------------------------------------------------------------------

def _diffusion_backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: "DiffusionCachedContrastive",
) -> None:
    """Re-run the forward pass with gradients for each mini-batch and connect
    the cached partial derivatives back into the computation graph.

    For query features (feat_idx == 0) we use the standard single-pass
    embed_minibatch_iter.  For document features we use the multi-pass
    variant so that gradients flow correctly through all M passes.
    """
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None

    with torch.enable_grad():
        for feat_idx, (sentence_feature, grad, random_states) in enumerate(
            zip(sentence_features, loss_obj.cache, loss_obj.random_states)
        ):
            is_doc = feat_idx > 0
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                    is_doc=is_doc,
                ),
                grad,
            ):
                surrogate = (
                    torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                )
                surrogate.backward()


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------

class DiffusionCachedContrastive(CachedContrastive):
    """Multi-forward-pass (diffusion-style) contrastive loss for ColBERT.

    Parameters
    ----------
    model
        A PyLate ColBERT model (already monkey-patched with padded_tokenize so
        that documents contain extra [EMB] pad tokens).
    num_passes
        Number of forward passes for document encoding (M).  A value of 1
        is equivalent to the base CachedContrastive.
    backprop_all_passes
        If True, gradients flow through ALL passes (the hidden state passed
        between passes stays in the computation graph).  If False, the
        injected hidden states are detached between passes so only the last
        pass receives gradient from the final loss.
    per_step_loss
        If True, compute contrastive loss at EVERY forward-pass step and
        return the sum.  GradCache is NOT used in this mode.  Use a smaller
        batch size to compensate for the higher memory usage.
    **kwargs
        Additional keyword arguments forwarded to CachedContrastive
        (e.g. mini_batch_size, temperature, score_metric, gather_across_devices).
    """

    def __init__(
        self,
        model,
        num_passes: int = 3,
        backprop_all_passes: bool = True,
        per_step_loss: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        self.num_passes = num_passes
        self.backprop_all_passes = backprop_all_passes
        self.per_step_loss = per_step_loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Unwrap DDP if necessary."""
        return self.model if hasattr(self.model, "_first_module") else self.model.module

    def _get_cls_token_id(self) -> int:
        return self._get_model()._first_module().tokenizer.cls_token_id

    def _get_doc_pad_mask(self, input_ids: Tensor) -> Tensor:
        """Boolean mask [B, T] that is True at extra [EMB] pad-token positions.

        Position 0 is always excluded because it holds the document-prefix
        token (which is also a [EMB]/CLS token in pylate's scheme).
        """
        cls_id = self._get_cls_token_id()
        mask = input_ids == cls_id  # [B, T]
        mask[:, 0] = False          # position 0 is the doc-prefix token
        return mask

    def _get_module_parts(self):
        """Return (auto_model, downstream_modules).

        auto_model      : the HuggingFace transformer backbone
        downstream_modules : list of sentence-transformer modules that follow
                             the Transformer (typically just the Dense
                             projection layer).
        """
        model = self._get_model()
        modules = list(model._modules.values())
        auto_model = modules[0].auto_model   # sentence_transformers Transformer wrapper → HF model
        downstream = modules[1:]             # Dense, etc.
        return auto_model, downstream

    def _get_word_emb_fn(self, auto_model):
        """Return the raw word/token embedding sub-layer of auto_model.embeddings.

        Tries ModernBERT's `tok_embeddings` then BERT's `word_embeddings`.
        """
        emb = auto_model.embeddings
        if hasattr(emb, "tok_embeddings"):
            return emb.tok_embeddings   # ModernBERT
        if hasattr(emb, "word_embeddings"):
            return emb.word_embeddings  # BERT / RoBERTa
        raise AttributeError(
            f"Cannot find word embedding layer in {type(emb)}. "
            "Expected 'tok_embeddings' (ModernBERT) or 'word_embeddings' (BERT)."
        )

    def _run_backbone(
        self,
        auto_model,
        downstream_modules,
        attention_mask: Tensor,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the HuggingFace backbone and all downstream (projection) modules.

        Exactly one of input_ids / inputs_embeds must be provided.

        Returns
        -------
        token_embeds : Tensor [B, T, D]
            Projected embeddings (output of Dense layer).
        last_hidden  : Tensor [B, T, H]
            Raw last hidden state from the transformer (H = hidden_dim).
            Used as feedback for the next diffusion pass.
        """
        assert (input_ids is None) != (
            inputs_embeds is None
        ), "Provide exactly one of input_ids or inputs_embeds."

        hf_kwargs = {"attention_mask": attention_mask}
        if input_ids is not None:
            hf_kwargs["input_ids"] = input_ids
        else:
            hf_kwargs["inputs_embeds"] = inputs_embeds

        hf_out = auto_model(**hf_kwargs)
        last_hidden = hf_out.last_hidden_state  # [B, T, H]

        # Run Dense projection (and any other downstream ST modules)
        features = {"token_embeddings": last_hidden, "attention_mask": attention_mask}
        for module in downstream_modules:
            features = module(features)

        return features["token_embeddings"], last_hidden  # [B, T, D], [B, T, H]

    def _extract_doc_embeddings(self, token_embeds: Tensor, pad_mask: Tensor) -> Tensor:
        """Extract and reshape pad-token embeddings from the full sequence.

        When all examples in the batch have the same number of pad tokens
        (the usual static-dvecs case) the result is a dense tensor of shape
        [B, n_pads, D].  When pad counts differ (dratio case) the full
        sequence with zeros at non-pad positions is returned [B, T, D].
        """
        n_pads = pad_mask.sum(dim=1)
        if n_pads.unique().numel() == 1:
            B, T, D = token_embeds.shape
            n = int(n_pads[0].item())
            return token_embeds[pad_mask].view(B, n, D)
        # Variable pad counts: keep full sequence, zero out non-pad positions.
        return token_embeds * pad_mask.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Multi-pass document embedding
    # ------------------------------------------------------------------

    def _embed_doc_multipass(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> tuple[Tensor, list[RandContext] | None]:
        """Embed a document mini-batch with num_passes forward passes.

        Pass j+1 replaces the word-level embeddings at pad-token positions
        with the last hidden state values at those positions from pass j,
        then runs through the full transformer + projection again.

        Parameters
        ----------
        sentence_feature
            Tokenised document features dict (full batch, not yet sliced).
        begin, end
            Slice indices for the current mini-batch.
        with_grad
            Whether the returned embeddings require grad (True during the
            GradCache backward hook, False during the no-grad forward pass).
        copy_random_state
            True during the initial no-grad pass; creates a RandContext for
            each forward pass so the backward hook can exactly reproduce them.
        random_states
            List of M RandContext objects (one per diffusion pass) to restore.
            None on the initial forward; populated for the backward hook replay.

        Returns
        -------
        embeddings : Tensor [mb, n_pads, D], normalised
        random_states_out : list[RandContext] | None
            M RandContexts (one per pass) if copy_random_state else None.
        """
        minibatch = {k: v[begin:end] for k, v in sentence_feature.items()}
        input_ids = minibatch["input_ids"]
        attention_mask = minibatch["attention_mask"]
        pad_mask = self._get_doc_pad_mask(input_ids)   # [mb, T]

        auto_model, downstream = self._get_module_parts()
        word_emb_fn = self._get_word_emb_fn(auto_model)

        random_states_out: list[RandContext] | None = [] if copy_random_state else None
        current_pad_hidden: Tensor | None = None  # [n_pads_total, H]
        final_token_embeds: Tensor | None = None

        for pass_idx in range(self.num_passes):
            # Gradient context: enable grad only for passes that need it.
            pass_with_grad = with_grad and (
                self.backprop_all_passes or pass_idx == self.num_passes - 1
            )
            pass_grad_ctx = nullcontext if pass_with_grad else torch.no_grad

            # Random-state context for this pass.
            rs_ctx: RandContext | type(nullcontext) = (
                random_states[pass_idx]
                if (random_states is not None and pass_idx < len(random_states))
                else nullcontext()
            )

            with rs_ctx:
                with pass_grad_ctx():
                    if copy_random_state:
                        random_states_out.append(RandContext(*minibatch.values()))

                    if pass_idx == 0 or current_pad_hidden is None:
                        # First pass: standard forward using input_ids.
                        token_embeds, last_hidden = self._run_backbone(
                            auto_model, downstream, attention_mask,
                            input_ids=input_ids,
                        )
                    else:
                        # Subsequent passes: replace pad-position inputs with
                        # the previous pass's last hidden state at those positions.
                        # We work at the *raw word-embedding* level so that the
                        # transformer's embedding layer (norm + dropout) processes
                        # the injected hidden states consistently.
                        raw_tok = word_emb_fn(input_ids).clone()  # [mb, T, H]

                        pad_hidden = (
                            current_pad_hidden
                            if self.backprop_all_passes
                            else current_pad_hidden.detach()
                        )
                        raw_tok[pad_mask] = pad_hidden

                        # Passing inputs_embeds causes auto_model.embeddings to
                        # apply its norm + dropout, which is intentional.
                        token_embeds, last_hidden = self._run_backbone(
                            auto_model, downstream, attention_mask,
                            inputs_embeds=raw_tok,
                        )

                    # Save hidden states at pad positions for the next pass.
                    current_pad_hidden = last_hidden[pad_mask]  # [n_pads, H]
                    final_token_embeds = token_embeds

        pad_embeds = self._extract_doc_embeddings(final_token_embeds, pad_mask)
        embeddings = F.normalize(pad_embeds, p=2, dim=-1)
        return embeddings, random_states_out

    # ------------------------------------------------------------------
    # Override embed_minibatch_iter to route docs through multi-pass
    # ------------------------------------------------------------------

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list | None = None,
        is_doc: bool = False,
    ) -> Iterator[tuple[Tensor, list | None]]:
        """Yield (embeddings, random_state) for each mini-batch chunk.

        For query features (is_doc=False) delegates to the base-class
        embed_minibatch (single forward pass, GradCache compatible).
        For document features (is_doc=True) with num_passes > 1, uses
        _embed_doc_multipass.

        The random_states argument differs in type depending on is_doc:
          * is_doc=False : list[RandContext]          (one per mini-batch chunk)
          * is_doc=True  : list[list[RandContext]]   (one list per chunk,
                           each inner list has num_passes entries)
        """
        input_ids = sentence_feature["input_ids"]
        bsz = input_ids.size(0)

        for i, b in enumerate(range(0, bsz, self.mini_batch_size)):
            e = b + self.mini_batch_size
            rs = None if random_states is None else random_states[i]

            if is_doc and self.num_passes > 1:
                reps, new_rs = self._embed_doc_multipass(
                    sentence_feature, b, e,
                    with_grad=with_grad,
                    copy_random_state=copy_random_state,
                    random_states=rs,
                )
            else:
                reps, new_rs = self.embed_minibatch(
                    sentence_feature, b, e,
                    with_grad=with_grad,
                    copy_random_state=copy_random_state,
                    random_state=rs,
                )

            yield reps, new_rs

    # ------------------------------------------------------------------
    # Forward: GradCache path (per_step_loss=False)
    # ------------------------------------------------------------------

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the contrastive loss with multi-pass document encoding.

        sentence_features[0]  : query (single forward pass)
        sentence_features[1:] : documents (num_passes forward passes each)
        """
        if self.per_step_loss:
            return self._forward_per_step(sentence_features)

        # ---- Standard GradCache path (loss at final step only) ----------
        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )

        reps: list[list[Tensor]] = []
        self.random_states: list[list] = []

        for feat_idx, sentence_feature in enumerate(sentence_features):
            is_doc = feat_idx > 0
            reps_mbs: list[Tensor] = []
            random_state_mbs: list = []

            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
                is_doc=is_doc,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)

            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        # Fix doc masks: our multi-pass embeddings have shape [B, dvecs, D] but
        # extract_skiplist_mask returns masks of shape [B, T_full].  Replace
        # doc masks with all-True tensors of the correct shape so the score
        # metric can apply them without a dimension mismatch.
        # (Pad tokens are [EMB] which are not in the skiplist, so all-True is
        # semantically correct.)
        for i in range(1, len(reps)):
            sample_chunk = reps[i][0]                          # [mb, n_vecs, D]
            n_vecs = sample_chunk.shape[1]
            full_B = sum(chunk.shape[0] for chunk in reps[i])
            masks[i] = torch.ones(
                full_B, n_vecs, dtype=torch.bool, device=sample_chunk.device
            )

        if torch.is_grad_enabled():
            # Step 2: backward through embeddings, cache gradients.
            loss = self.calculate_loss_and_cache_gradients(reps, masks)
            # Step 3: register hook to replay with-grad forward passes.
            loss.register_hook(
                partial(
                    _diffusion_backward_hook,
                    sentence_features=sentence_features,
                    loss_obj=self,
                )
            )
        else:
            loss = self.calculate_loss(reps, masks)

        return loss

    # ------------------------------------------------------------------
    # Forward: per-step loss path (per_step_loss=True, no GradCache)
    # ------------------------------------------------------------------

    def _forward_per_step(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
    ) -> Tensor:
        """Compute contrastive loss at every diffusion step and return the sum.

        No GradCache is used here.  Gradients flow directly through all M
        forward passes (subject to backprop_all_passes).

        The cross-entropy scores at each step use the same query embeddings
        (encoded once) and the document embeddings produced at that step.
        """
        sentence_features = list(sentence_features)
        query_feature = sentence_features[0]
        doc_features = sentence_features[1:]

        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )

        auto_model, downstream = self._get_module_parts()
        word_emb_fn = self._get_word_emb_fn(auto_model)

        # ---- Encode queries (single pass through the full model) ---------
        # Uses the monkey-patched model.forward (newforward), which handles
        # query-vector extraction if qvecs > 0.
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            q_out = self.model(query_feature)
            q_embeds = F.normalize(q_out["token_embeddings"], p=2, dim=-1)

        batch_size = q_embeds.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=q_embeds.device)

        # ---- Prepare per-document-feature state --------------------------
        # Each entry tracks the pad mask and the running hidden-state buffer.
        doc_states = []
        for df in doc_features:
            iids = df["input_ids"]
            amask = df["attention_mask"]
            pm = self._get_doc_pad_mask(iids)
            doc_states.append(
                dict(
                    input_ids=iids,
                    attention_mask=amask,
                    pad_mask=pm,
                    current_pad_hidden=None,
                )
            )

        total_loss = torch.zeros(1, device=q_embeds.device)

        # ---- M forward passes -------------------------------------------
        for pass_idx in range(self.num_passes):
            pass_with_grad = torch.is_grad_enabled() and (
                self.backprop_all_passes or pass_idx == self.num_passes - 1
            )

            step_doc_embeds: list[Tensor] = []

            for doc_idx, state in enumerate(doc_states):
                input_ids = state["input_ids"]
                attention_mask = state["attention_mask"]
                pad_mask = state["pad_mask"]
                current_pad_hidden = state["current_pad_hidden"]

                with torch.set_grad_enabled(pass_with_grad):
                    if pass_idx == 0 or current_pad_hidden is None:
                        token_embeds, last_hidden = self._run_backbone(
                            auto_model, downstream, attention_mask,
                            input_ids=input_ids,
                        )
                    else:
                        raw_tok = word_emb_fn(input_ids).clone()
                        pad_h = (
                            current_pad_hidden
                            if self.backprop_all_passes
                            else current_pad_hidden.detach()
                        )
                        raw_tok[pad_mask] = pad_h
                        token_embeds, last_hidden = self._run_backbone(
                            auto_model, downstream, attention_mask,
                            inputs_embeds=raw_tok,
                        )

                    # Update state for next pass.
                    state["current_pad_hidden"] = last_hidden[pad_mask]

                pad_embeds = self._extract_doc_embeddings(token_embeds, pad_mask)
                doc_step_embeds = F.normalize(pad_embeds, p=2, dim=-1)

                step_doc_embeds.append(doc_step_embeds)

            # Score queries against all doc groups at this step.
            # Pass documents_mask=None: our extracted pad-token embeddings
            # [B, dvecs, D] are not punctuation, so all positions are valid.
            step_scores = torch.cat(
                [
                    self.score_metric(
                        q_embeds,
                        d_embeds,
                        queries_mask=masks[0] if not do_query_expansion else None,
                        documents_mask=None,
                    )
                    for d_embeds in step_doc_embeds
                ],
                dim=1,
            )

            step_loss = F.cross_entropy(
                step_scores / self.temperature,
                labels,
                reduction="sum",
            )
            if self.size_average:
                step_loss = step_loss / batch_size

            total_loss = total_loss + step_loss

        return total_loss.squeeze()
