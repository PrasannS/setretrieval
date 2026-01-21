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
import wandb
import requests
from peft import LoraConfig
from typing import Callable
import itertools
from setretrieval.eval.maxmax_evaluator import MaxMaxTripletEvaluator

from pylate.scores import colbert_scores
import pylate.scores as pscores
from pylate.utils import all_gather, all_gather_with_gradients, get_rank, get_world_size
import numpy as np
import torch
from sentence_transformers.util import batch_to_device
from torch import nn
from tqdm.autonotebook import trange
from typing import Literal
from numpy import ndarray
import copy
from pylate.utils.tensor import convert_to_tensor
from pylate.models.colbert import ColBERT
from setretrieval.train.scores import route_colbscores_multiquery, route_colbscores_multipos, maxmax_scores, maxmax_scores_pairwise

# given normal output with vector shape [B, T, D], zero out all vectors that aren't in the first token for each item in batch
def newforward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
    outs = self.old_forward(input, **kwargs)
    mask = (input["input_ids"] == self._first_module().tokenizer.cls_token_id) & (input["attention_mask"] == 1)  # Shape: [batch_size, seq_len]
    # ignore cls token
    mask[:, 0] = False
    # breakpoint()
    mask = mask.unsqueeze(-1)
    # mask = torch.zeros_like(outs['token_embeddings'])
    # mask[:, 1, :] = 1   

    outs['token_embeddings'] = outs['token_embeddings'] * mask

    return outs

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

# use to monkey patch ColBERT to tokenize with pad tokens
def padded_tokenize(
        self,
        texts: list[str] | list[dict] | list[tuple[str, str]],
        is_query: bool = True,
        pad: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenizes the input texts.

        Args:
            texts (Union[list[str], list[dict], list[tuple[str, str]]]): A list of texts to be tokenized.
            is_query (bool): Flag to indicate if the texts are queries. Defaults to True.
            pad (bool): Flag to indicate if elements should be padded to max length. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: A dictionary of tensors with the tokenized texts, including "input_ids",
                "attention_mask", and optionally "token_type_ids".
        """
        # Set max sequence length based on whether the input is a query or document
        # max_length = self.query_length if is_query else self.document_length
        # # TODO this should be tokenizer max length
        if self._first_module().max_seq_length in [512, 32768]: # this is to handle weirdness from prefix token
            self._first_module().max_seq_length = self._first_module().max_seq_length - 1
        # print(f"WARNING: HARD-CODED MAX SEQ LENGTH TO 511")
        # HACK ok I think this was usually padding to max length which was killing memory for qwen I'm assuming
        # need to fix this up a bit
        splitquant=0
        if "||||" in texts[0]:

            # for the set case (to circumvent annoying dataloader stuff)
            spltexts = [text.split("||||") for text in texts]
            splitquant = len(spltexts[0])
            texts = [item for sublist in spltexts for item in sublist]

        # breakpoint()

        if self._first_module().tokenizer.cls_token is None:
            cls_token = '[EMB]'
            # breakpoint()
            # check if [EMB] is in added tokens
            if self._first_module().tokenizer.convert_tokens_to_ids(cls_token) is None:
                # in this case, add a special '[CLS]' token to the tokenizer, set it to be the cls_token_id
                self._first_module().tokenizer.cls_token_id = self._first_module().tokenizer.add_tokens(cls_token)
                self._first_module().tokenizer.cls_token = cls_token
                self._first_module().auto_model.resize_token_embeddings(len(self._first_module().tokenizer))
            else:
                self._first_module().tokenizer.cls_token_id = self._first_module().tokenizer.convert_tokens_to_ids(cls_token)
                self._first_module().tokenizer.cls_token = cls_token

        # breakpoint()
        addvecs = self._first_module().tokenizer.query_vectors if is_query else self._first_module().tokenizer.doc_vectors

        if type(texts[0]) == dict:
            texts = [text['text'] for text in texts]


        ct = self._first_module().tokenizer.cls_token
        ct = ct if ct == "[EMB]" else " "+ct
        texts = [text + ct * addvecs for text in texts]


        # Tokenize the texts, let's generally not pad to max length here
        tokenized_outputs = self._first_module().tokenize(texts, padding="longest")

        # breakpoint()

        tokenized_outputs['input_ids'] = tokenized_outputs['input_ids'][:, :self._first_module().max_seq_length]
        tokenized_outputs['attention_mask'] = tokenized_outputs['attention_mask'][:, :self._first_module().max_seq_length]
        if "token_type_ids" in tokenized_outputs:
            tokenized_outputs['token_type_ids'] = tokenized_outputs['token_type_ids'][:, :self._first_module().max_seq_length]

        # in rows that contain no padding, then replace last addvecs tokens with cls_tokens 
        if "bert-large-uncased" in self.tokenizer.name_or_path:
            nprows = [row[-1] != self._first_module().tokenizer.pad_token_id and row[-1] != self._first_module().tokenizer.sep_token_id for row in tokenized_outputs['input_ids']]
            for i in range(len(nprows)):
                if nprows[i]:
                    tokenized_outputs['input_ids'][i, -addvecs:] = self._first_module().tokenizer.cls_token_id
            if sum(nprows) > 0:
                print(f"{sum(nprows)} rows were truncated, adjusted their cls tokens")

        # Determine prefix ID based on input type
        prefix_id = self.query_prefix_id if is_query else self.document_prefix_id

        # add prefix id to every tokenized tensor in tokenized_outputs
        tokenized_outputs['input_ids'] = self.insert_prefix_token(tokenized_outputs['input_ids'], prefix_id)
        tokenized_outputs['attention_mask'] = self.insert_prefix_token(tokenized_outputs['attention_mask'], 1)

        # Update token type IDs if they exist
        if "token_type_ids" in tokenized_outputs:
            tokenized_outputs["token_type_ids"] = self.insert_prefix_token(tokenized_outputs['token_type_ids'], 0)

        # use this in loss function when needed
        tokenized_outputs['splitquant'] = torch.tensor(splitquant)
        return tokenized_outputs

# for model-specific monkey patching
def modadj_tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], is_query: bool = True, pad: bool = False) -> dict[str, torch.Tensor]:
    tokenized_outputs = self.old_tokenize(texts, is_query=is_query, pad=pad)
    if "qwen" in self.tokenizer.name_or_path.lower():
        # swap indices 0 and 1 in all sequences
        tmp = tokenized_outputs['input_ids'][:, 0].clone()
        tokenized_outputs['input_ids'][:, 0] = tokenized_outputs['input_ids'][:, 1].clone()
        tokenized_outputs['input_ids'][:, 1] = tmp
    return tokenized_outputs

def mod_encode(self, sentences: str | list[str], prompt_name: str | None = None, prompt: str | None = None, batch_size: int = 32, show_progress_bar: bool = None, precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32", convert_to_numpy: bool = True, convert_to_tensor: bool = False, padding: bool = False, device: str = None, normalize_embeddings: bool = True, is_query: bool = True, pool_factor: int = 1, protected_tokens: int = 1) -> list[torch.Tensor] | ndarray | torch.Tensor:
    embeddings = self.old_encode(sentences, prompt_name, prompt, batch_size, show_progress_bar, precision, convert_to_numpy, convert_to_tensor, padding, device, normalize_embeddings, is_query, pool_factor, protected_tokens)
    # identify tokens with zero vectors and get rid of them to get a new list of vectors for each item
    new_embeddings = []
    for embedding in embeddings:
        temb = torch.tensor(embedding)
        # identify tokens with zero vectors (all numbers in last dimension are zeros)
        zero_mask = temb.abs().sum(dim=-1) == 0
        # breakpoint()
        # get rid of them
        temb = temb[~zero_mask]
        new_embeddings.append(temb)
    # breakpoint()
    print(f"Embedding count: {len(new_embeddings[0])}")
    return new_embeddings

# dataset should be loaded in first, should have format of [query, positive, negative]
def train_colbert(train_dataset, eval_dataset, base_model="google-bert/bert-base-uncased", mini_batch_size=32, per_device_batch_size=1000, num_train_epochs=3, learning_rate=3e-6, dsetname="gemini_datav1", div_coeff=0.0, colscore="maxmax", save_strat="epoch", schedtype="constant", maxchars=5000, divq_coeff=0.0, temp=0.02, othermodel_name="google-bert/bert-large-uncased", othermod="neither", qvecs=5, dvecs=5, dodefaulttrain="no", compile="yes", lora_rank=-1, embsize=128):
    """Train set retrieval models."""
    train_dataset = train_dataset.map(lambda x: {"positive": x["positive"][:maxchars], "negative": x["negative"][:maxchars]})
    eval_dataset = eval_dataset.map(lambda x: {"positive": x["positive"][:maxchars], "negative": x["negative"][:maxchars]})

    # torch.autograd.set_detect_anomaly(True)
    # Set the run name for logging and output directory
    run_name = f"contrastive-{base_model.replace('/', '_')}-bs{per_device_batch_size}-e{num_train_epochs}-lr{learning_rate}-{dsetname}-{colscore}-divd{div_coeff}-divq{divq_coeff}-qv{qvecs}-dv{dvecs}-{schedtype}-temp{temp}-omod{othermod}-dodefaulttrain{dodefaulttrain}-embsize{embsize}"
    
    # breakpoint()
    if dodefaulttrain == "no":
        models.ColBERT.tokenize = padded_tokenize
        models.ColBERT.old_forward = models.ColBERT.forward
        models.ColBERT.forward = newforward
        models.ColBERT.old_encode = models.ColBERT.encode
        models.ColBERT.encode = mod_encode

    # we need to do this in all cases
    models.ColBERT.old_tokenize = models.ColBERT.tokenize
    models.ColBERT.tokenize = modadj_tokenize

    torch.set_float32_matmul_precision('high')

    if lora_rank > 0:
        lora_config = LoraConfig(r=lora_rank,
            lora_alpha=lora_rank,lora_dropout=0.05,
            target_modules=["q_proj", "v_proj",  "k_proj", "up_proj", "down_proj", "gate_proj"], # "o_proj",
            bias="none",
            # task_type="CAUSAL_LM",
        )
        run_name += f"-lora{lora_rank}"

    output_dir = f"propercache/cache/colbert_training/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder
    model = models.ColBERT(
        model_name_or_path=base_model, query_length=qvecs, document_length=dvecs, 
        do_query_expansion=dodefaulttrain == "yes", embedding_size=embsize
    ) 
    
    print(model.tokenizer)
    if lora_rank > 0:
        model.add_adapter(lora_config)
    # there's an off-by-one here? Maybe tokenizer adds a vector by default?
    # HACK might need to modify this for other models...
    if dodefaulttrain == "no":
        model.tokenizer.query_vectors = qvecs 
        model.tokenizer.doc_vectors = dvecs 

    if compile == "yes":
        print("Compiling model")
        model = torch.compile(model)
    else:
        print("Not compiling model")

    # Define the loss function, there's some gather option as well
    if False:
        train_loss = losses.CachedContrastive(model=model, mini_batch_size=mini_batch_size, temperature=0.02, show_progress_bar=True)
    else:
        metric = maxmax_scores if colscore == "maxmax" else colbert_scores
        metric = route_colbscores_multipos if colscore == "multipos" else metric
        metric = route_colbscores_multiquery if colscore == "multiquery" else metric
        
        print(f"Using colscore: {metric.__name__} {colscore} as the score metric")
        # train_loss = losses.Contrastive(model=model, temperature=0.02)
        # if dodefaulttrain == "no":
        train_loss = SetContrastive(model=model, temperature=temp, score_metric=metric, div_coeff=div_coeff, divq_coeff=divq_coeff, othermodel_name=othermodel_name, othermod=othermod)
        # train_loss = Contrastive(model=model, temperature=temp)
        if div_coeff == 0.0 and divq_coeff == 0.0 and False:
            print("Using Contrastive loss")
            train_loss = Contrastive(model=model, temperature=temp)
        # else: 
        #     print("DOING DEFAULT TRAINING")
        #     train_loss = Contrastive(model=model, temperature=temp)

    if colscore in ["maxmax", "multipos"]:
        print("USING CUSTOM EVALUATOR")
        dev_evaluator = MaxMaxTripletEvaluator(
            anchors=eval_dataset["query"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
        )
    else:
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
        logging_steps=1,
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

    # breakpoint()


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