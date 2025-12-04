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


class EasyIndexer:
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='cache/easy_indices', num_gpus=4):
        self.model_name = model_name
        self.available_gpus = num_gpus
        print(f"Available GPUs: {self.available_gpus}")
        self.model = None # SentenceTransformer(model_name, device='cuda:0' if self.available_gpus > 0 else 'cpu', trust_remote_code=True)
        self.indices = {}
        self.bm25_indices = {}
        self.documents = {}
        # add folder specific to model and make that the base path
        self.index_base_path = os.path.join(index_base_path, model_name.replace("/", "_"))

    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device='cuda:0' if self.available_gpus > 0 else 'cpu', trust_remote_code=True)

    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, f"{index_id}.faiss"))

    # given documents and an id, either load or search index
    def index_documents(self, documents, index_id, emb_bsize=8, pre_embeds=None):
        os.makedirs(self.index_base_path, exist_ok=True)
        if index_id in self.indices:
            return

        # read documents and index if they exist
        if self.index_exists(index_id):
            index = faiss.read_index(os.path.join(self.index_base_path, f"{index_id}.faiss"))
            self.documents[index_id] = pd.read_csv(os.path.join(self.index_base_path, f"{index_id}.csv"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        else:
            assert documents is not None
            if pre_embeds is None:
                embeds =self.embed_with_multi_gpu(documents, batch_size=emb_bsize)
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
        assert qtype in ['document', 'query']
        assert type(documents) == list
        # This handles GPU distribution automatically
        if len(documents) > self.available_gpus:
            print(f"Using {self.available_gpus} GPUs for embedding")
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

    # given query, return list of dictionaries with document, index, and score for top k
    def search(self, queries, index_id, k=10, doembed=True):
        if index_id not in self.indices:
            self.index_documents(None, index_id)

        index = self.indices[index_id]
        if doembed:
            query_embedding = self.embed_with_multi_gpu(queries, qtype="query")
        else:
            if isinstance(queries, list)==False:
                query_embedding = np.array([queries])
            else:
                query_embedding = np.array(queries)
        if len(query_embedding.shape) == 3:
            query_embedding = query_embedding.squeeze()
        # breakpoint()
        # breakpoint()
        distances, indices = index.search(query_embedding, min(k, len(self.documents[index_id])))
        print("ok done with that")
        # breakpoint()
        fullresults = []
        for j in tqdm(range(len(indices))):
            results = []
            for i, idx in enumerate(indices[j]):
                if idx < len(self.documents[index_id]):
                    results.append({
                        'score': distances[j][i],
                        'index': idx, 
                        'index_id': index_id
                    })
                    # results.append(self.documents[index_id].iloc[idx])
                    # results[-1]["score"] = distances[j][i]
                    # results[-1]["index"] = idx
            fullresults.append(results)
        if len(fullresults) == 1:
            return fullresults[0]
        return fullresults

    def composed_search(self, queries, index_ids, k=10, doembed=True):
        allres = []
        for index_id in index_ids:
            allres.extend(self.search(queries, index_id, k=k, doembed=doembed))
        # now sort by distance (lower is better?)
        allres.sort(key=lambda x: x['score'])
        return allres

    def index_dataset(self, dset_path):
        # index datasets based on path to a dataset (with a text column)
        dset = Dataset.load_from_disk(dset_path)
        self.index_documents(dset["text"], dset_path.replace("/", "_"))

    def bm25_search(self, query, index_id, k=10):
        if index_id not in self.indices:
            self.index_documents(None, index_id)
        if index_id not in self.bm25_indices:
            print(self.documents[index_id].iloc[0])
            assert isinstance(self.documents[index_id].iloc[0]['text'], str)
            self.bm25_indices[index_id] = BM25Okapi([self.documents[index_id].iloc[i]['text'].split(" ") for i in range(len(self.documents[index_id]))])
        query = query.split(" ")
        scores = self.bm25_indices[index_id].get_scores(query)
        top_indices = np.argsort(scores)[::-1][:k]
        results = [{'text': self.documents[index_id].iloc[i]['text'], 'index': i, 'score': scores[i]} for i in top_indices]
        return results

    # given id of document, return top k closest documents to that document
    def document_closest_others(self, doc_index, index_id, k=50):
        self.index_documents(None, index_id)
        relevantindex = self.indices[index_id]
        relembed = relevantindex.reconstruct(doc_index).reshape(1, -1)
        distances, indices = relevantindex.search(relembed, k)
        return distances[0], indices[0]


