"""ColBERT indexer using PLAID index."""

import os
import numpy as np
from datasets import Dataset
from pylate import indexes, retrieve

from setretrieval.indexers.easy_indexer import EasyIndexerBase
from setretrieval.indexers.colbert_model import ColBERTModelMixin


class ColBERTEasyIndexer(ColBERTModelMixin, EasyIndexerBase):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='propercache/cache/colbert_indices',
                 qmod_name=None, qvecs=-1, dvecs=-1, passiveqvecs=0, passivedvecs=0, use_bsize=128, usefast=True):
        self._init_colbert_model(model_name, qmod_name, qvecs, dvecs, passiveqvecs, passivedvecs, use_bsize, usefast)
        EasyIndexerBase.__init__(self, model_name, index_base_path, self.num_workers)

    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, f"{index_id}"))

    def index_documents(self, documents, index_id, pre_embeds=None, redo=False):
        os.makedirs(self.index_base_path, exist_ok=True)
        if self.index_exists(index_id) and not redo:
            print(f"Index {index_id} already exists")
            return

        embeds = pre_embeds if pre_embeds is not None else self.embed_with_multi_gpu(documents, qtype="document")

        print("Adding documents to index (new)")
        self.indices[index_id] = indexes.PLAID(
            index_folder=self.index_base_path, index_name=index_id,
            override=True, use_fast=self.usefast, embedding_size=self.embsize
        )
        self.indices[index_id].add_documents(
            documents_ids=[str(i) for i in range(len(documents))],
            documents_embeddings=embeds
        )
        self.documents[index_id] = Dataset.from_dict({"text": documents})
        self.documents[index_id].save_to_disk(os.path.join(self.index_base_path, f"{index_id}"))
        print(f"Indexed {len(documents)} documents for index {index_id}")

    def try_load_cached(self, index_id):
        if index_id not in self.indices:
            if not self.index_exists(index_id):
                raise ValueError(f"Index {index_id} does not exist")
            self.indices[index_id] = indexes.PLAID(
                index_folder=self.index_base_path, index_name=index_id,
                override=True, use_fast=self.usefast, embedding_size=self.embsize
            )
            self.documents[index_id] = Dataset.load_from_disk(os.path.join(self.index_base_path, f"{index_id}"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        return self.indices[index_id]

    def search(self, queries, index_id, k=10, doembed=True):
        self.try_load_cached(index_id)
        assert isinstance(queries, list), "Queries must be a list"
        query_embedding = self.embed_with_multi_gpu(queries, qtype="query") if doembed else np.array(queries)
        retriever = retrieve.ColBERT(index=self.indices[index_id])
        results = retriever.retrieve(queries_embeddings=query_embedding, k=min(k, len(self.documents[index_id])))
        return [[{'score': entry['score'], 'index': int(entry['id']), 'index_id': index_id} for entry in result] for result in results]
