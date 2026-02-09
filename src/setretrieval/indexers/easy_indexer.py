"""Base class for all indexers."""

from datasets import Dataset
import os
from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


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
        raise NotImplementedError("Child class must implement load_model")

    def index_exists(self, index_id):
        raise NotImplementedError("Child class must implement index_exists")

    def composed_search(self, queries, index_ids, k=10, doembed=True, reverse=False):
        allres = []
        for index_id in index_ids:
            allres.extend(self.search(queries, index_id, k=k, doembed=doembed))
        allres.sort(key=lambda x: x['score'], reverse=reverse)
        return allres

    def index_dataset(self, dset_path, redo=False):
        """Index a dataset from disk (must have a 'text' column)."""
        dset = Dataset.load_from_disk(dset_path)
        self.index_documents(list(dset["text"]), dset_path.replace("/", "_"), redo=redo)
        return dset_path.replace("/", "_")
