from __future__ import annotations

import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import sentence_transformers.losses as slosses
import sentence_transformers.evaluation as sevaluation
from pylate import evaluation, losses, models, utils
from pylate.losses.contrastive import extract_skiplist_mask, Contrastive
import os
import torch.nn as nn
from typing import Iterable
from torch import Tensor
import torch.nn.functional as F

from pylate.models import ColBERT
from pylate.scores import colbert_scores
from pylate.utils import all_gather, all_gather_with_gradients, get_rank, get_world_size


import numpy as np
import torch

from pylate.utils.tensor import convert_to_tensor


def maxmax_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
) -> torch.Tensor:
    print("Using maxmax_scores!")

    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)
    # breakpoint()
    scores = scores.max(axis=-1).values.max(axis=-1).values
    return scores

class SetContrastive(nn.Module):

    def __init__(self, model: ColBERT, score_metric=maxmax_scores, size_average: bool = True, gather_across_devices: bool = False, temperature: float = 1.0, div_coeff: float = 1.0) -> None:
        super(SetContrastive, self).__init__()
        self.score_metric = score_metric
        self.model = model
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.temperature = temperature
        self.div_coeff = div_coeff

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
        embeddings = [torch.nn.functional.normalize(self.model(sentence_feature)["token_embeddings"], p=2, dim=-1) for sentence_feature in sentence_features]
        # handle the model being wrapped in (D)DP and so require to access module first
        skiplist = self.model.skiplist if hasattr(self.model, "skiplist") else self.model.module.skiplist
        do_query_expansion = self.model.do_query_expansion if hasattr(self.model, "do_query_expansion") else self.model.module.do_query_expansion
        masks = extract_skiplist_mask(sentence_features=sentence_features, skiplist=skiplist)
        batch_size = embeddings[0].size(0)
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

        normq = F.normalize(embeddings[0], dim=-1)
        simqueries = normq @ normq.transpose(-1, -2)
        # breakpoint()
        avg_pairwise_simq = simqueries[:, ~torch.eye(simqueries.shape[1], dtype=bool)].mean(dim=1)

        normd = F.normalize(embeddings[1], dim=-1)
        simdocs = normd @ normd.transpose(-1, -2)
        avg_pairwise_simd = simdocs[:, ~torch.eye(simdocs.shape[1], dtype=bool)].mean(dim=1)
        
        # Note: the queries mask is not used, if added, take care that the expansion tokens are not masked from scoring (because they might be masked during encoding).
        # We might not need to compute the mask for queries but I let the logic there for now
        scores = torch.cat(
            [
                self.score_metric(
                    embeddings[0],
                    group_embeddings,
                    queries_mask=masks[0] if not do_query_expansion else None,
                    documents_mask=documents_masks,
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

        # add loss for average pairwise cosine similarity of embeddings between themselves
        loss += self.div_coeff * (avg_pairwise_simq.mean() + avg_pairwise_simd.mean())

        # TODO log this in wandb

        # breakpoint()

        # Scale by world size when gathering across device
        if self.gather_across_devices:
            loss *= get_world_size()
        return loss


# dataset should be loaded in first, should have format of [query, positive, negative]
def train_colbert(train_dataset, eval_dataset, base_model="google-bert/bert-base-uncased", mini_batch_size=32, per_device_batch_size=1000, num_train_epochs=3, learning_rate=3e-6, dsetname="gemini_datav1", div_coeff=1.0, colscore="maxmax", querylen=256, save_strat="epoch", schedtype="constant", maxchars=5000):
    """Train set retrieval models."""
    train_dataset = train_dataset.map(lambda x: {"positive": x["positive"][:maxchars], "negative": x["negative"][:maxchars]})
    eval_dataset = eval_dataset.map(lambda x: {"positive": x["positive"][:maxchars], "negative": x["negative"][:maxchars]})


    # Set the run name for logging and output directory
    run_name = f"contrastive-{base_model.replace('/', '_')}-bs{per_device_batch_size}-e{num_train_epochs}-lr{learning_rate}-{dsetname}-{colscore}-div{div_coeff}-qlen{querylen}-{schedtype}"
    output_dir = f"propercache/cache/colbert_training/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # breakpoint()

    # 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
    model = models.ColBERT(model_name_or_path=base_model, query_length=querylen)

    # Compiling the model makes the training faster
    model = torch.compile(model)

    # Define the loss function, there's some gather option as well
    if False:
        train_loss = losses.CachedContrastive(model=model, mini_batch_size=mini_batch_size, temperature=0.02, show_progress_bar=True)
    else:
        metric = maxmax_scores if colscore == "maxmax" else colbert_scores
        print(f"Using colscore: {metric.__name__} {colscore} as the score metric")
        # train_loss = losses.Contrastive(model=model, temperature=0.02)
        train_loss = SetContrastive(model=model, temperature=0.02, score_metric=metric, div_coeff=div_coeff)

    # Initialize the evaluator
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    print(f"Let's train on {len(train_dataset)} examples and evaluate on {len(eval_dataset)} examples")
     
    # breakpoint()
    # Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=mini_batch_size,
        bf16=True,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        learning_rate=learning_rate,
        eval_strategy="epoch",
        lr_scheduler_type=schedtype,
        save_only_model=True,
        save_strategy=save_strat,
        # deepspeed="dsconfig.json",
    )

    # Initialize the trainer for the contrastive training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    # Start the training process
    trainer.train()

    trainer.save_model(output_dir)


# dataset should be loaded in first, should have format of [query, positive, negative]
def train_sbert(train_dataset, eval_dataset, base_model="google-bert/bert-base-uncased", per_device_batch_size=32, num_train_epochs=3, learning_rate=3e-6, dsetname="gemini_datav1"):
    """Train single-vector retrieval models."""

    # Set the run name for logging and output directory
    run_name = f"contrastive-{base_model.replace('/', '_')}-bs{per_device_batch_size}-e{num_train_epochs}-lr{learning_rate}-sbert-{dsetname}"
    output_dir = f"propercache/cache/sbert_training/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Define the SentenceTransformer model
    model = SentenceTransformer(base_model)

    # # Compiling the model makes the training faster
    # model = torch.compile(model)

    # Define the loss function - MultipleNegativesRankingLoss is the standard contrastive loss for SBERT
    train_loss = slosses.MultipleNegativesRankingLoss(model=model, scale=50)

    # Initialize the evaluator - TripletEvaluator measures accuracy on triplets
    dev_evaluator = sevaluation.TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="dev",
    )

    print(f"Let's train on {len(train_dataset)} examples and evaluate on {len(eval_dataset)} examples")

    # Configure the training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        bf16=True,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        learning_rate=learning_rate,
        eval_strategy="epoch",
        lr_scheduler_type="constant",
        save_only_model=True,
    )

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    
    # Start the training process
    trainer.train()
    trainer.save_model(output_dir)
    
    return model