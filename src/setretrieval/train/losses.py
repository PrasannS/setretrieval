import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from torch import Tensor
import wandb
from pylate import models
from pylate.losses.contrastive import extract_skiplist_mask
from pylate.utils import all_gather, all_gather_with_gradients, get_rank, get_world_size
from __future__ import annotations

from typing import Callable, Iterable

import torch

from pylate.models import ColBERT
from pylate.scores import colbert_kd_scores
from pylate.losses.contrastive import extract_skiplist_mask


class SetContrastive(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        score_metric=None,
        size_average: bool = True, 
        gather_across_devices: bool = False, 
        temperature: float = 1.0, 
        div_coeff: float = 0.0, 
        divq_coeff: float = 0.0, 
        othermod: str = "neither", # can be "neither", "document", or "query"
        othermodel_name: str = "google-bert/bert-large-uncased"
    ) -> None:
        super(SetContrastive, self).__init__()
        self.score_metric = score_metric
        self.model = model
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.temperature = temperature
        self.div_coeff = div_coeff
        self.divq_coeff = divq_coeff
        if othermod != "neither":
            # hopefully this isn't differentiable
            self.other_model = models.ColBERT(othermodel_name)
            self.other_model.eval()
        self.othermod = othermod

    
    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Constrastive loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the anchor and the rest are the positive and negative examples.
        labels
            The labels for the contrastive loss. Not used in this implementation, but kept for compatibility with Trainer.
        """
        # breakpoint()
        batch_size = sentence_features[0]['input_ids'].size(0) 
        if 'splitquant' in sentence_features[0] and sentence_features[0]['splitquant'] > 0:
            batch_size = batch_size // sentence_features[0]['splitquant']
        # TODO set up model isolation
        if self.othermod == "query":
            with torch.no_grad():
                query_embeddings = torch.nn.functional.normalize(self.other_model(sentence_features[0])["token_embeddings"], p=2, dim=-1)
        else:
            query_embeddings = torch.nn.functional.normalize(self.model(sentence_features[0])["token_embeddings"], p=2, dim=-1)
        
        if self.othermod == "document":
            with torch.no_grad():
                document_embeddings = [torch.nn.functional.normalize(self.other_model(sentence_feature)["token_embeddings"], p=2, dim=-1) for sentence_feature in sentence_features[1:]]
        else:
            document_embeddings = [torch.nn.functional.normalize(self.model(sentence_feature)["token_embeddings"], p=2, dim=-1) for sentence_feature in sentence_features[1:]]

        # TODO query embedding split quant thing
        embeddings = [query_embeddings, *document_embeddings]
        # breakpoint()
        # breakpoint()
        # handle the model being wrapped in (D)DP and so require to access module first
        skiplist = self.model.skiplist if hasattr(self.model, "skiplist") else self.model.module.skiplist
        
        # HACK TODO double check old contrastive loss code
        # masks = [torch.ones_like(sentence_feature["input_ids"], dtype=torch.bool) for sentence_feature in sentence_features]
        masks = extract_skiplist_mask(sentence_features=sentence_features, skiplist=skiplist) # seems kinda unnecessary...
        for i in range(len(embeddings)):
            if 'splitquant' in sentence_features[i] and sentence_features[i]['splitquant'] > 0:
                # breakpoint()
                print(embeddings[i].shape, sentence_features[i]['splitquant'])
                embeddings[i] = embeddings[i].reshape(batch_size, sentence_features[i]['splitquant'], embeddings[i].shape[-2], embeddings[i].shape[-1])
                masks[i] = masks[i].reshape(batch_size, sentence_features[i]['splitquant'], masks[i].shape[-1])

        # create corresponding labels
        labels = torch.arange(0, batch_size, device=embeddings[0].device)

        # breakpoint()
        # Possibly gather the embeddings across devices to have more in-batch negatives.
        if self.gather_across_devices:
            # Note that we only gather the documents embeddings and not the queries embeddings (embeddings[0]), but are keeping gradients. This is to lower the memory usage, see https://github.com/mlfoundations/open_clip/issues/616
            embeddings = [
                embeddings[0],
                *[
                    torch.cat(all_gather_with_gradients(embedding))
                    for embedding in embeddings[1:]
                ],
            ]
            # Masks [0] is the anchor mask so we do not need to gather it (even though we are not using it for now anyways)
            # Also, we do gather without gradients for the masks as we do not backpropagate through them
            masks = [
                masks[0],
                *[torch.cat(all_gather(mask)) for mask in masks[1:]],
            ]
            rank = get_rank()
            # Adjust the labels to match the gathered embeddings positions
            labels = labels + rank * batch_size

        # ignore embeddings that are all zero
        def compute_avg_pairwise_sim(inpembeddings):
            if len(inpembeddings.shape) == 4:
                # if input is [B, N, T, D], reshape to [B, T, D] (take first thing of N)
                embeddings = inpembeddings[:, 0, :, :]
                # breakpoint()
            else:
                embeddings = inpembeddings
            B, T, D = embeddings.shape
            if T == 1:
                return torch.tensor([0.0])
            
            # Identify non-zero vectors (any non-zero element in last dim)
            non_zero_mask = (embeddings.abs().sum(dim=-1) > 0)  # (B, T)
            
            # Compute similarity matrix
            sim = embeddings @ embeddings.transpose(-1, -2)  # (B, T, T)
            
            # Create mask for valid pairs (both vectors non-zero, excluding diagonal)
            valid_pairs = non_zero_mask.unsqueeze(-1) & non_zero_mask.unsqueeze(-2)  # (B, T, T)
            valid_pairs = valid_pairs & ~torch.eye(T, dtype=bool, device=embeddings.device)
            
            # Count valid pairs per batch
            num_valid = valid_pairs.sum(dim=(1, 2))  # (B,)
            
            # Sum similarities for valid pairs
            masked_sim = torch.where(valid_pairs, sim, torch.zeros_like(sim))
            sum_sim = masked_sim.sum(dim=(1, 2))  # (B,)
            
            # Compute average, handling case where no valid pairs exist
            avg_sim = torch.where(num_valid > 0, sum_sim / num_valid, torch.zeros_like(sum_sim))
            
            return avg_sim

        avg_pairwise_simq = compute_avg_pairwise_sim(embeddings[0])
        avg_pairwise_simd = compute_avg_pairwise_sim(embeddings[1])
        
        # breakpoint()
        # Note: the queries mask is not used, if added, take care that the expansion tokens are not masked from scoring (because they might be masked during encoding).
        # We might not need to compute the mask for queries but I let the logic there for now
        # HACK no in-batch negatives
        # eyemat = torch.eye(n=batch_size, device=embeddings[0].device)
        scores = torch.cat(
            [
                self.score_metric(
                    embeddings[0],
                    group_embeddings,
                    # queries_mask=masks[0] if not do_query_expansion else None, HACK no longer need masking since we use every token
                    # documents_mask=documents_masks,
                ) 
                for group_embeddings, documents_masks in zip(embeddings[1:], masks[1:])
            ],
            dim=1,
        )
        # breakpoint()

        # compute constrastive loss using cross-entropy over the scores
        loss = F.cross_entropy(
            input=scores / self.temperature,
            target=labels,
            reduction="mean" if self.size_average else "sum",
        )

        divloss = self.divq_coeff * (avg_pairwise_simq.mean()) + self.div_coeff * (avg_pairwise_simd.mean())
        loss += divloss

        # breakpoint()


        if wandb.run is not None and (not self.gather_across_devices or get_rank() == 0):  # Check if wandb is initialized
            wandb.log({
                "train/avg_pairwise_simq": avg_pairwise_simq.mean().item(),
                "train/avg_pairwise_simd": avg_pairwise_simd.mean().item(),
                "train/diversity_loss": divloss.item(),
            })

        # add loss for average pairwise cosine similarity of embeddings between themselves

        # TODO log this in wandb

        # breakpoint()

        # Scale by world size when gathering across device
        if self.gather_across_devices:
            loss *= get_world_size()
        return loss




class DistillationST(torch.nn.Module):
    """Distillation loss for ColBERT model. The loss is computed with respect to the format of SentenceTransformer library.

    Parameters
    ----------
    model
        SentenceTransformer model.
    score_metric
        Function that returns a score between two sequences of embeddings.
    size_average
        Average by the size of the mini-batch or perform sum.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> distillation = losses.Distillation(model=model)

    >>> query = model.tokenize([
    ...     "fruits are healthy.",
    ... ], is_query=True)

    >>> documents = model.tokenize([
    ...     "fruits are good for health.",
    ...     "fruits are bad for health."
    ... ], is_query=False)

    >>> sentence_features = [query, documents]

    >>> labels = torch.tensor([
    ...     [0.7, 0.3],
    ... ], dtype=torch.float32)

    >>> loss = distillation(sentence_features=sentence_features, labels=labels)

    >>> assert isinstance(loss.item(), float)
    """

    def __init__(
        self,
        model: ColBERT,
        score_metric: Callable = colbert_kd_scores,
        size_average: bool = True,
        normalize_scores: bool = True,
    ) -> None:
        super(DistillationST, self).__init__()
        self.score_metric = score_metric
        self.model = model
        self.loss_function = torch.nn.KLDivLoss(
            reduction="batchmean" if size_average else "sum", log_target=True
        )
        self.normalize_scores = normalize_scores

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes the distillation loss with respect to SentenceTransformer.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the query and the rest are documents.
        labels
            The logits for the distillation loss.

        """
        queries_embeddings = torch.nn.functional.normalize(
            [self.model(sentence_features[0])["sentence_embedding"]], p=2, dim=-1
        )
        # Compute the bs * n_ways embeddings
        documents_embeddings = torch.nn.functional.normalize(
            [self.model(sentence_features[1])["sentence_embedding"]], p=2, dim=-1
        )

        # Reshape them to (bs, n_ways)
        documents_embeddings = documents_embeddings.view(
            queries_embeddings.size(0), -1, *documents_embeddings.shape[1:]
        )

        # handle the model being wrapped in (D)DP and so require to access module first
        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )

        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )

        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )

        documents_embeddings_mask = masks[1].view(
            queries_embeddings.size(0), -1, *masks[1].shape[1:]
        )
        scores = self.score_metric(
            queries_embeddings,
            documents_embeddings,
            queries_mask=masks[0] if not do_query_expansion else None,
            documents_mask=documents_embeddings_mask,
        )
        if self.normalize_scores:
            # Compute max and min along the num_scores dimension (dim=1)
            max_scores, _ = torch.max(scores, dim=1, keepdim=True)
            min_scores, _ = torch.min(scores, dim=1, keepdim=True)

            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-8

            # Normalize the scores
            scores = (scores - min_scores) / (max_scores - min_scores + epsilon)
        return self.loss_function(
            torch.nn.functional.log_softmax(scores, dim=-1),
            torch.nn.functional.log_softmax(labels, dim=-1),
        )
