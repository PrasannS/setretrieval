# Class EasyIndexer, which takes a list of documents (strings) and an index id, and then either loads from cache or embeds / indexes (with faiss)
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import argparse
import os
from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
import torch
from multiprocessing import Pool, cpu_count
import time
import faiss
import pandas as pd
from rank_bm25 import BM25Okapi
from pylate import indexes, models, retrieve
from ..utils.utils import pickdump, pickload


class EasyIndexerBase:
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='cache/easy_indices', num_gpus=4):
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

    def index_dataset(self, dset_path):
        # index datasets based on path to a dataset (with a text column)
        dset = Dataset.load_from_disk(dset_path)
        self.index_documents(dset["text"], dset_path.replace("/", "_"))
        

# Indexer for single vector retrieval
class SingleEasyIndexer(EasyIndexerBase):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='cache/single_indices', num_gpus=4):
        super().__init__(model_name, index_base_path, num_gpus)
        
    # By not loading model initially, we can allow inference without GPU (speedup in certain cases)
    def load_model(self):
        if self.model is None:
            print("Loading vanilla sentence transformer model")
            self.model = SentenceTransformer(self.model_name, device='cuda:0' if self.available_gpus > 0 else 'cpu', trust_remote_code=True)

    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, f"{index_id}.faiss"))

    # given documents and an id, either load or search index
    def index_documents(self, documents, index_id, emb_bsize=8, pre_embeds=None):
        os.makedirs(self.index_base_path, exist_ok=True)
        if index_id in self.indices:
            print(f"Index {index_id} already exists")
            return
        assert documents is not None

        if pre_embeds is None:
            embeds = self.embed_with_multi_gpu(documents, batch_size=emb_bsize)
        else: 
            embeds = pre_embeds
        embeds = np.array(embeds)

        print("Now indexing documents...")
        index = faiss.IndexFlatL2(embeds.shape[1])
        index.add(embeds)
        faiss.write_index(index, os.path.join(self.index_base_path, f"{index_id}.faiss"))
        self.documents[index_id] = pd.DataFrame(documents, columns=["text"])
        self.documents[index_id].to_csv(os.path.join(self.index_base_path, f"{index_id}.csv"), index=False) # save documents
        self.indices[index_id] = index
        print(f"Indexed {len(self.documents[index_id])} documents for index {index_id}")

    def embed_with_multi_gpu(self, documents, qtype="document", batch_size=8):

        self.load_model()
        assert qtype in ['document', 'query'] and type(documents) == list

        # This handles GPU distribution automatically
        if len(documents) > self.available_gpus:
            pool = self.model.start_multi_process_pool(target_devices=[f'cuda:{i}' for i in range(self.available_gpus)])
        else: 
            pool = None
        
        if self.model_name == "nomic-ai/nomic-embed-text-v1":
            # preprocess documents with prefix search_document: 
            documents = [f"search_{qtype}: {doc}" for doc in documents]
        
        print("Embedding documents, remember to format documents correctly!")
        print("EXAMPLE: "+documents[0])
        # TODO this should do multi-GPU I think?
        if "Qwen" in self.model_name and qtype == "query":
            embeddings = self.model.encode(documents, pool=pool, batch_size=batch_size, show_progress_bar=True, prompt_name="query")
        else:
            embeddings = self.model.encode(documents, pool=pool, batch_size=batch_size, show_progress_bar=True)
        if pool is not None:
            self.model.stop_multi_process_pool(pool)
        return embeddings

    def try_load_cached(self, index_id):
        if index_id not in self.indices:
            if self.index_exists(index_id) == False:
                raise ValueError(f"Index {index_id} does not exist")
            self.indices[index_id] = faiss.read_index(os.path.join(self.index_base_path, f"{index_id}.faiss"))
            self.documents[index_id] = pd.read_csv(os.path.join(self.index_base_path, f"{index_id}.csv"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        
    # given query, return list of dictionaries with document, index, and score for top k
    def search(self, queries, index_id, k=10, doembed=True):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"

        query_embedding = self.embed_with_multi_gpu(queries, qtype="query") if doembed else np.array(queries)

        # HACK, not sure why this happens, may want to clean up higher up code
        if len(query_embedding.shape) == 3:
            query_embedding = query_embedding.squeeze()

        distances, indices = self.indices[index_id].search(query_embedding, min(k, len(self.documents[index_id])))

        fullresults = [[{'score': distances[j][i],'index': idx, 'index_id': index_id} for i, idx in enumerate(indices[j])] for j in range(len(indices))]
        
        return fullresults

    # given id of document, return top k closest documents to that document
    def document_closest_others(self, doc_index, index_id, k=50):
        self.index_documents(None, index_id)
        relevantindex = self.indices[index_id]
        relembed = relevantindex.reconstruct(doc_index).reshape(1, -1)
        distances, indices = relevantindex.search(relembed, k)
        return distances[0], indices[0]


# ColBERT-style
class ColBERTEasyIndexer(EasyIndexerBase):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='cache/colbert_indices', num_gpus=4):
        super().__init__(model_name, index_base_path, num_gpus)

    def load_model(self):
        if self.model is None:
            print("Loading ColBERT model")
            self.model = models.ColBERT(model_name_or_path=self.model_name)

    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, f"fast_plaid_index"))

    def embed_with_multi_gpu(self, documents, qtype="document", batch_size=8):
        assert qtype in ['document', 'query'] and type(documents) == list
        self.load_model()

        return self.model.encode(documents, batch_size=batch_size, is_query=qtype=="query", show_progress_bar=True)

    def index_documents(self, documents, index_id, emb_bsize=8, pre_embeds=None):
        os.makedirs(self.index_base_path, exist_ok=True)
        if index_id in self.indices:
            print(f"Index {index_id} already exists")
            return

        if pre_embeds is None:
            embeds = self.embed_with_multi_gpu(documents, qtype="document")
        else:
            embeds = pre_embeds
        embeds = np.array(embeds)
        self.indices[index_id] = indexes.PLAID(index_folder=self.index_base_path, index_name=index_id, override=False)
        self.indices.add_documents(documents_ids=[str(i) for i in range(len(documents))], documents_embeddings=embeds)
        self.documents[index_id] = pd.DataFrame(documents, columns=["text"])
        self.documents[index_id].to_csv(os.path.join(self.index_base_path+"/"+index_id+"/", f"documents.csv"), index=False) # save documents
        print(f"Indexed {len(documents)} documents for index {index_id}")

    def try_load_cached(self, index_id):
        if index_id not in self.indices:
            if self.index_exists(index_id) == False:
                raise ValueError(f"Index {index_id} does not exist")
            self.indices[index_id] = indexes.PLAID(index_folder=self.index_base_path, index_name=index_id, override=False)
            self.documents[index_id] = pd.read_csv(os.path.join(self.index_base_path+"/"+index_id+"/", f"documents.csv"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        return self.indices[index_id]

    def search(self, queries, index_id, k=10, doembed=True):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"
        query_embedding = self.embed_with_multi_gpu(queries, qtype="query") if doembed else np.array(queries)
        retriever = retrieve.ColBERT(index=self.indices[index_id])
        results = retriever.retrieve(query_embeddings=query_embedding, k=min(k, len(self.documents[index_id])))
        return [[{'score': entry['score'], 'index': int(entry['id']), 'index_id': index_id} for entry in result] for result in results]

    def composed_search(self, queries, index_ids, k=10, doembed=True, reverse=False):
        # just use super but ensure that reverse is True
        return super().composed_search(queries, index_ids, k=k, doembed=doembed, reverse=True)

# BM25 version of things
class BM25EasyIndexer(EasyIndexerBase):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='cache/bm25_indices', num_gpus=4):
        super().__init__(model_name, index_base_path, num_gpus)

    def load_model(self):
        raise NotImplementedError("BM25EasyIndexer does not need to load a model")

    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, index_id+".bm25"))
        
    def index_documents(self, documents, index_id):
        os.makedirs(self.index_base_path, exist_ok=True)
        if index_id in self.indices:
            print(f"Index {index_id} already exists")
            return
        self.indices[index_id] = BM25Okapi([document.split(" ") for document in documents])
        pickdump(self.indices[index_id], os.path.join(self.index_base_path, index_id+".bm25"))
        self.documents[index_id] = pd.DataFrame(documents, columns=["text"])
        self.documents[index_id].to_csv(os.path.join(self.index_base_path+"/"+index_id+"/", f"documents.csv"), index=False) # save documents
        print(f"Indexed {len(documents)} documents for index {index_id}")
    
    def try_load_cached(self, index_id):
        if index_id not in self.indices:
            if self.index_exists(index_id) == False:
                raise ValueError(f"Index {index_id} does not exist")
            self.indices[index_id] = pickload(os.path.join(self.index_base_path, index_id+".bm25"))
            self.documents[index_id] = pd.read_csv(os.path.join(self.index_base_path+"/"+index_id+"/", f"documents.csv"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        return self.indices[index_id]

    def search(self, queries, index_id, k=10):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"
        queries = [query.split(" ") for query in queries]
        def singleresult(query): 
            tops = np.argsort(self.indices[index_id].get_scores(query))[::-1][:k]
            return [{'score': self.indices[index_id].get_scores(query)[i], 'index': i, 'index_id': index_id} for i in tops]
        return [singleresult(query) for query in queries]

    def composed_search(self, queries, index_ids, k=10, reverse=False):
        # just use super but ensure that reverse is True
        return super().composed_search(queries, index_ids, k=k, reverse=True)