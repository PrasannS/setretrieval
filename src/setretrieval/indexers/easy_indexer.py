# Class EasyIndexer, which takes a list of documents (strings) and an index id, and then either loads from cache or embeds / indexes (with faiss)
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import os
from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
import faiss
from pylate import indexes, models, retrieve
from ..utils.utils import pickdump, pickload
import bm25s
import Stemmer
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import random
from vllm import LLM
import pylate.scores as pylatescores
from setretrieval.train.colbert_train import maxmax_scores
from pylate.indexes import PLAID, Voyager
from pylate.rank import RerankResult, rerank
from pylate.utils import iter_batch
import logging
from pylate.rank.rank import reshape_embeddings, func_convert_to_tensor
from tqdm import tqdm
from transformers import AutoTokenizer
from setretrieval.train.colbert_train import padded_tokenize, newforward, modadj_tokenize, mod_encode
import json

logger = logging.getLogger(__name__)

class EasyIndexerBase:
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='propercache/cache/easy_indices', num_gpus=4):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.model = None
        self.index_base_path = os.path.join(index_base_path, model_name.replace("/", "_"))
        self.available_gpus = num_gpus
        self.indices = {}
        self.documents = {}
    
    def load_model(self):
        # force child class to implement this
        raise NotImplementedError("Child class must implement load_model")

    def index_exists(self, index_id):
        # force child class to implement this
        raise NotImplementedError("Child class must implement index_exists")

    def composed_search(self, queries, index_ids, k=10, doembed=True, reverse=False):
        allres = []
        for index_id in index_ids:
            allres.extend(self.search(queries, index_id, k=k, doembed=doembed))
        # now sort by distance (lower is better?)
        allres.sort(key=lambda x: x['score'], reverse=reverse)
        return allres

    def index_dataset(self, dset_path, redo=False):
        # index datasets based on path to a dataset (with a text column)
        dset = Dataset.load_from_disk(dset_path)
        self.index_documents(list(dset["text"]), dset_path.replace("/", "_"), redo=redo)
        return dset_path.replace("/", "_")


