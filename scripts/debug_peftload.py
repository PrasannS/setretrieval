# try to load colbert model, then add existing peft adapter, see if output matches vectmp.pkl
from pylate import models, retrieve
from setretrieval.utils.utils import pickload

import torch
import os
import numpy as np
import pickle
import argparse
import sys
import time
import random
import pylate.evaluation as evaluation
from datasets import DatasetDict

from setretrieval.train.colbert_train import padded_tokenize, newforward, mod_encode, modadj_tokenize
if __name__ == "__main__":

    models.ColBERT.tokenize = padded_tokenize
    models.ColBERT.old_forward = models.ColBERT.forward
    models.ColBERT.forward = newforward
    models.ColBERT.old_encode = models.ColBERT.encode
    models.ColBERT.encode = mod_encode

    # we need to do this in all cases
    models.ColBERT.old_tokenize = models.ColBERT.tokenize
    models.ColBERT.tokenize = modadj_tokenize


    mod = models.ColBERT(model_name_or_path="Qwen/Qwen3-Embedding-0.6B")
    mod = mod.eval()

    cls_token = '[EMB]'
    mod.tokenizer.cls_token_id = mod.tokenizer.add_tokens(cls_token)
    mod.tokenizer.cls_token = cls_token
    mod._first_module().auto_model.resize_token_embeddings(len(mod.tokenizer))

    mod.tokenizer.query_vectors = 1
    mod.tokenizer.doc_vectors = 10

    breakpoint()
    mod.load_adapter("propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv10-cosine-temp0.02-omodneither-dodefaulttrainno-lora16")
    evset = DatasetDict.load_from_disk("propercache/data/colbert_training/gutenberg_gmini_30k_nosame")['test']

    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=evset["query"],
        positives=evset["positive"],
        negatives=evset["negative"],
        show_progress_bar=True,
    )

    res = dev_evaluator(mod)
    breakpoint()


    ref = pickload("vectmp.pkl")

    res = mod.old_encode("hi there")

    breakpoint()
