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

    def index_dataset(self, dset_path):
        # index datasets based on path to a dataset (with a text column)
        dset = Dataset.load_from_disk(dset_path)
        self.index_documents(list(dset["text"]), dset_path.replace("/", "_"))
        return dset_path.replace("/", "_")



# Indexer for single vector retrieval
class SingleEasyIndexer(EasyIndexerBase):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='propercache/cache/single_indices', num_gpus=8):
        super().__init__(model_name, index_base_path, num_gpus)

    # By not loading model initially, we can allow inference without GPU (speedup in certain cases)
    def load_model(self):
        if self.model is None:
            print("Loading vanilla sentence transformer model")
            if self.num_gpus == 0:
                self.model = SentenceTransformer(self.model_name, device='cpu', trust_remote_code=True)
                self.toker = AutoTokenizer.from_pretrained(self.model_name)
            else:
                self.model = LLM(model=self.model_name, tensor_parallel_size=torch.cuda.device_count(), runner="pooling")
                # self.model = SentenceTransformer(self.model_name, device='cuda:0' if self.available_gpus > 0 else 'cpu', trust_remote_code=True)
                self.toker = self.model.get_tokenizer()
    
    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, f"{index_id}.faiss"))

    # given documents and an id, either load or search index
    def index_documents(self, documents, index_id, emb_bsize=64, pre_embeds=None):
        os.makedirs(self.index_base_path, exist_ok=True)
        if self.index_exists(index_id):
            print(f"Index {index_id} already exists")
            return
        assert documents is not None
        
        dlim = 5000
        print(f"Truncating {sum([len(d)>dlim for d in documents])} documents to {dlim} characters")
        documents = [d[:dlim] if len(d) > dlim else d for d in documents]

        if pre_embeds is None:
            embeds = self.embed_with_multi_gpu(documents, batch_size=emb_bsize)
        else: 
            embeds = pre_embeds
        embeds = np.array(embeds)

        print("Now indexing documents...")
        index = faiss.IndexFlatL2(embeds.shape[1])
        index.add(embeds)
        faiss.write_index(index, os.path.join(self.index_base_path, f"{index_id}.faiss"))
        self.documents[index_id] = Dataset.from_dict({"text": documents})
        self.documents[index_id].save_to_disk(os.path.join(self.index_base_path, f"{index_id}")) # save documents
        self.indices[index_id] = index
        print(f"Indexed {len(self.documents[index_id])} documents for index {index_id}")

    def embed_with_multi_gpu(self, documents, qtype="document", batch_size=64):

        self.load_model()
        assert qtype in ['document', 'query'] and (type(documents) == list)

        if "google-bert" in self.model_name:
            # tokenize / detokenize to get things less than 510 tokens long
            docs = [self.toker.encode(doc) for doc in documents]
            docs = [self.toker.decode(doc[:505], skip_special_tokens=True) for doc in docs]
            documents = docs
            # breakpoint()
        
        if self.model_name == "nomic-ai/nomic-embed-text-v1":
            # preprocess documents with prefix search_document: 
            documents = [f"search_{qtype}: {doc}" for doc in documents]
        elif "Qwen" in self.model_name and qtype == "query":
            documents = [f"Instruct: Find documents, both normal and unexpected, that are relevant to the query.\nQuery: {doc}" for doc in documents]
        
        if self.num_gpus == 0:
            embeddings = self.model.encode(documents, batch_size=1, show_progress_bar=True)
        else:
            embeddings = self.model.embed(documents)
            embeddings = [e.outputs.embedding for e in embeddings]
        # breakpoint()
        
        return embeddings

    def try_load_cached(self, index_id):
        if index_id not in self.indices:
            if self.index_exists(index_id) == False:
                raise ValueError(f"Index {index_id} does not exist")
            self.indices[index_id] = faiss.read_index(os.path.join(self.index_base_path, f"{index_id}.faiss"))
            self.documents[index_id] = Dataset.load_from_disk(os.path.join(self.index_base_path, f"{index_id}"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        
    # given query, return list of dictionaries with document, index, and score for top k
    def search(self, queries, index_id, k=10, doembed=True):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"

        query_embedding = self.embed_with_multi_gpu(queries, qtype="query") if doembed else queries
        query_embedding = np.array(query_embedding)

        # breakpoint()

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



class ColBERTEasyIndexer(EasyIndexerBase):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='propercache/cache/colbert_indices', gpu_list=None, div_colbert=False):
        # If gpu_list is None, default to single GPU
        if gpu_list is None:
            gpu_list = [0]
        self.div_colbert = div_colbert
        self.gpu_list = list(range(torch.cuda.device_count()))
        self.num_workers = len(self.gpu_list)
        # breakpoint()
        # Initialize base class (you may need to adjust this depending on EasyIndexerBase)
        super().__init__(model_name, index_base_path, self.num_workers)
        self.models = []  # Store multiple model instances

    def load_model(self):
        if self.model is None:
            print("Loading ColBERT model")
            self.model = models.ColBERT(model_name_or_path=self.model_name)

    def load_models_for_workers(self):
        """Load a separate model instance for each worker (can have multiple workers per GPU)"""
        if len(self.models) == 0:
            print(f"Loading {self.num_workers} model instances across GPUs: {set(self.gpu_list)}")
            for worker_id, gpu_id in enumerate(tqdm(self.gpu_list)):
                model = models.ColBERT(model_name_or_path=self.model_name, device=f'cuda:{gpu_id}')
                self.models.append((model, gpu_id, worker_id))
                # print(f"  Worker {worker_id} -> GPU {gpu_id}")

    def embed_with_multi_gpu(self, documents, qtype="document", batch_size=128):
        assert qtype in ['document', 'query'] and type(documents) == list
        
        # Use single GPU if only one worker or small dataset
        if self.num_workers <= 1 or len(documents) < batch_size * 2:
            self.load_model()
            return self.model.encode(documents, batch_size=batch_size, is_query=qtype=="query", show_progress_bar=True)
        
        # Multi-worker encoding
        self.load_models_for_workers()
        
        # Split documents across workers
        chunk_size = (len(documents) + self.num_workers - 1) // self.num_workers
        document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
        
        def encode_on_worker(model_gpu_worker, docs):
            model, gpu_id, worker_id = model_gpu_worker
            # Ensure model is on correct GPU
            return model.encode(docs, batch_size=batch_size, is_query=qtype=="query", show_progress_bar=True)
        
        # Process chunks in parallel across workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(encode_on_worker, self.models[i], chunk)
                for i, chunk in enumerate(document_chunks) if len(chunk) > 0
            ]
            results = [future.result() for future in futures]
        
        # breakpoint()
        allresults = []
        for result in results:
            allresults.extend(result)
        # Concatenate results
        return allresults

    def index_exists(self, index_id):
        # breakpoint()
        return os.path.exists(os.path.join(self.index_base_path, index_id, f"fast_plaid_index"))

    def index_documents(self, documents, index_id, pre_embeds=None):
        os.makedirs(self.index_base_path, exist_ok=True)
        if self.index_exists(index_id):
            print(f"Index {index_id} already exists")
            return

        if pre_embeds is None:
            embeds = self.embed_with_multi_gpu(documents, qtype="document")
        else:
            embeds = pre_embeds
        print("Adding documents to index")
        self.indices[index_id] = indexes.PLAID(index_folder=self.index_base_path, index_name=index_id, override=False)
        self.indices[index_id].add_documents(documents_ids=[str(i) for i in range(len(documents))], documents_embeddings=embeds)
        self.documents[index_id] = Dataset.from_dict({"text": documents})
        self.documents[index_id].save_to_disk(os.path.join(self.index_base_path, f"{index_id}"))
        print(f"Indexed {len(documents)} documents for index {index_id}")

    def try_load_cached(self, index_id):
        # breakpoint()
        if index_id not in self.indices:
            if self.index_exists(index_id) == False:
                raise ValueError(f"Index {index_id} does not exist")
            self.indices[index_id] = indexes.PLAID(index_folder=self.index_base_path, index_name=index_id, override=False)
            self.documents[index_id] = Dataset.load_from_disk(os.path.join(self.index_base_path, f"{index_id}"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        return self.indices[index_id]

    def search(self, queries, index_id, k=10, doembed=True):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"
        query_embedding = self.embed_with_multi_gpu(queries, qtype="query") if doembed else np.array(queries)
        retriever = retrieve.ColBERT(index=self.indices[index_id])
        results = retriever.retrieve(queries_embeddings=query_embedding, k=min(k, len(self.documents[index_id])))
        return [[{'score': entry['score'], 'index': int(entry['id']), 'index_id': index_id} for entry in result] for result in results]


class TokenColBERTEasyIndexer(ColBERTEasyIndexer):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='propercache/cache/faiss_colbert_indices', gpu_list=[0,1,2,3,4,5,6,7]):
        super().__init__(model_name, index_base_path, gpu_list)
        self.token_to_doc_map = {}  # Maps token index to (doc_id, token_position)

    def index_exists(self, index_id):
        # based on single vector index exists code
        return os.path.exists(os.path.join(self.index_base_path, f"{index_id}.faiss"))
    
    def index_documents(self, documents, index_id, emb_bsize=64, pre_embeds=None):
        os.makedirs(self.index_base_path, exist_ok=True)
        if self.index_exists(index_id):
            print(f"Index {index_id} already exists")
            return

        if pre_embeds is None:
            embeds = self.embed_with_multi_gpu(documents, qtype="document")
        else:
            embeds = pre_embeds
        
        # Flatten all token vectors and create mapping
        all_token_vectors = []
        token_map = []  # List of (doc_id, token_idx) tuples
        
        print("Flattening token vectors...")
        for doc_id, doc_embed in enumerate(tqdm(embeds)):
            # doc_embed shape: (num_tokens, embedding_dim)
            num_tokens = doc_embed.shape[0]
            for token_idx in range(num_tokens):
                all_token_vectors.append(doc_embed[token_idx])
                token_map.append((doc_id, token_idx))
        
        # Convert to numpy array
        all_token_vectors = np.array(all_token_vectors, dtype=np.float32)
        print(f"Total token vectors: {len(all_token_vectors)}, dimension: {all_token_vectors.shape[1]}")
        
        # Create FAISS index
        print("Building FAISS index...")
        embedding_dim = all_token_vectors.shape[1]
        print(f"Number of token vectors: {len(all_token_vectors)} for {len(documents)} documents")
        index = faiss.IndexFlatIP(embedding_dim)  # Inner product for ColBERT
        index.add(all_token_vectors)
        
        # Save index and mappings
        faiss.write_index(index, os.path.join(self.index_base_path, f"{index_id}.faiss"))
        
        pickdump(token_map, os.path.join(self.index_base_path, f"{index_id}_token_map.pkl"))
        # Save documents
        self.documents[index_id] = Dataset.from_dict({"text": documents})
        self.documents[index_id].save_to_disk(os.path.join(self.index_base_path, f"{index_id}"))
        
        print(f"Indexed {len(documents)} documents ({len(all_token_vectors)} tokens) for index {index_id}")
    
    def try_load_cached(self, index_id):
        if index_id not in self.indices:
            if not self.index_exists(index_id):
                raise ValueError(f"Index {index_id} does not exist")
            
            self.indices[index_id] = faiss.read_index(os.path.join(self.index_base_path, f"{index_id}.faiss"))
            self.token_to_doc_map[index_id] = pickload(os.path.join(self.index_base_path, f"{index_id}_token_map.pkl"))

            self.documents[index_id] = Dataset.load_from_disk(os.path.join(self.index_base_path, f"{index_id}"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
    

    # TODO deduplication makes things weird
    def search(self, queries, index_id, k=100):
        
        self.try_load_cached(index_id)
        self.load_model()
        
        assert type(queries) == list, "Queries must be a list"
        
        # Encode queries
        print("Encoding queries...")
        query_embeds = self.model.encode(queries, batch_size=32, is_query=True, show_progress_bar=True)
        
        results = []
        token_map = self.token_to_doc_map[index_id]
        
        for query_embed in tqdm(query_embeds, desc="Searching"):
            # query_embed shape: (num_query_tokens, embedding_dim)
            num_query_tokens = query_embed.shape[0]
            # Retrieve more tokens per query token to ensure we get k unique documents after deduplication
            k_per_token = k  # Retrieve k tokens per query token to handle deduplication
            
            # Batch search all query tokens at once
            scores, indices = self.indices[index_id].search(query_embed, k_per_token)
            
            # Collect all document matches with their query token index
            doc_matches = {}  # doc_id -> best match info
            
            for query_token_idx in range(num_query_tokens):
                for j in range(k_per_token):
                    doc_token_idx = indices[query_token_idx][j]
                    doc_id = int(token_map[doc_token_idx][0])
                    token_idx = int(token_map[doc_token_idx][1])
                    score = float(scores[query_token_idx][j])
                    
                    # Keep best score per document
                    if doc_id not in doc_matches or score > doc_matches[doc_id]['score']:
                        doc_matches[doc_id] = {
                            'score': score,
                            'index': doc_id,
                            'token_idx': token_idx,
                            'query_token_idx': query_token_idx,
                            'index_id': index_id
                        }
            
            # Sort by score and take top k
            sorted_matches = sorted(doc_matches.values(), key=lambda x: x['score'], reverse=True)[:k]
            results.append(sorted_matches)
        # breakpoint()
        
        return results
        
# BM25 version of things
class BM25EasyIndexer(EasyIndexerBase):
    def __init__(self, model_name='bm25', index_base_path='propercache/cache/bm25_indices', num_gpus=4):
        super().__init__(model_name, index_base_path, num_gpus)
        self.stemmer = Stemmer.Stemmer("english")

    def load_model(self):
        raise NotImplementedError("BM25EasyIndexer does not need to load a model")

    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, index_id+".bm25"))
        
    def index_documents(self, documents, index_id):
        os.makedirs(self.index_base_path, exist_ok=True)
        if self.index_exists(index_id):
            print(f"Index {index_id} already exists")
            return
        tokenized = bm25s.tokenize(documents, stopwords="en", stemmer=self.stemmer)
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        self.indices[index_id] = retriever
        pickdump(self.indices[index_id], os.path.join(self.index_base_path, index_id+".bm25"))
        self.documents[index_id] = Dataset.from_dict({"text": documents})
        self.documents[index_id].save_to_disk(os.path.join(self.index_base_path, f"{index_id}")) # save documents
        print(f"Indexed {len(documents)} documents for index {index_id}")
    
    def try_load_cached(self, index_id):
        if index_id not in self.indices:
            if self.index_exists(index_id) == False:
                raise ValueError(f"Index {index_id} does not exist")
            self.indices[index_id] = pickload(os.path.join(self.index_base_path, index_id+".bm25"))
            self.documents[index_id] = Dataset.load_from_disk(os.path.join(self.index_base_path, f"{index_id}"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        return self.indices[index_id]

    def search(self, queries, index_id, k=10):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"
        queries = [bm25s.tokenize(query, stemmer=self.stemmer) for query in queries]
        bm25 = self.indices[index_id]
        allresults = []
        print("started search")
        for query in tqdm(queries):
            # Get scores once (not twice like before)
            results, scores = bm25.retrieve(query, k=min(k, len(self.documents[index_id])))
            # Extract scores for top k indices
            allresults.append([{'score': scores[0][i], 'index': results[0][i], 'index_id': index_id} for i in range(len(results[0]))])
        return allresults

    def composed_search(self, queries, index_ids, k=10, reverse=False):
        # just use super but ensure that reverse is True
        return super().composed_search(queries, index_ids, k=k, reverse=True)


class RandomEasyIndexer(BM25EasyIndexer):

    # make the search random
    def search(self, queries, index_id, k=10):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"
        return [[{'score': random.random(), 'index': random.randint(0, len(self.documents[index_id])-1), 'index_id': index_id} for _ in range(k)] for _ in queries]


### OLD CODE FOR UNDERSTANDING PYLATE CODE

# def divrerank(
#     documents_ids: list[list[int | str]],
#     queries_embeddings: list[list[float | int] | np.ndarray | torch.Tensor],
#     documents_embeddings: list[list[float | int] | np.ndarray | torch.Tensor],
#     device: str = None,
# ) -> list[list[RerankResult]]:
    
#     results = []

#     queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
#     documents_embeddings = reshape_embeddings(embeddings=documents_embeddings)

#     for query_embeddings, query_documents_ids, query_documents_embeddings in tqdm(zip(
#         queries_embeddings, documents_ids, documents_embeddings
#     ), desc="Reranking documents"):
#         query_embeddings = func_convert_to_tensor(query_embeddings)

#         query_documents_embeddings = [
#             func_convert_to_tensor(query_document_embeddings)
#             for query_document_embeddings in query_documents_embeddings
#         ]

#         # Pad the documents embeddings
#         query_documents_embeddings = torch.nn.utils.rnn.pad_sequence(
#             query_documents_embeddings, batch_first=True, padding_value=0
#         )

#         if device is not None:
#             query_embeddings = query_embeddings.to(device)
#             query_documents_embeddings = query_documents_embeddings.to(device)
#         else:
#             query_documents_embeddings = query_documents_embeddings.to(
#                 query_embeddings.device
#             )

#         query_scores = maxmax_scores(
#             queries_embeddings=query_embeddings.unsqueeze(0),
#             documents_embeddings=query_documents_embeddings,
#         )[0]

#         scores, sorted_indices = torch.sort(input=query_scores, descending=True)
#         scores = scores.cpu().tolist()

#         query_documents = [query_documents_ids[idx] for idx in sorted_indices.tolist()]

#         results.append(
#             [
#                 RerankResult(id=doc_id, score=score)
#                 for doc_id, score in zip(query_documents, scores)
#             ]
#         )

#     return results


# class DivColBERTRetriever:

#     def __init__(self, index: Voyager | PLAID) -> None:
#         self.index = index

#     def retrieve(
#         self,
#         queries_embeddings: list[list | np.ndarray | torch.Tensor],
#         k: int = 10,
#         k_token: int = 100,
#         device: str | None = None,
#         batch_size: int = 50,
#         subset: list[list[str]] | list[str] | None = None,
#     ) -> list[list[RerankResult]]:
#         # PLAID index directly retrieves the documents
#         if isinstance(self.index, PLAID) or not isinstance(self.index, Voyager):
#             return self.index(
#                 queries_embeddings=queries_embeddings,
#                 k=k,
#                 subset=subset,
#             )

#         # Other indexes first generate candidates by calling the index and then rerank them
#         if k > k_token:
#             logger.warning(
#                 f"k ({k}) is greater than k_token ({k_token}), setting k_token to k."
#             )
#             k_token = k
#         print("We're here now")
#         reranking_results = []
#         for queries_embeddings_batch in iter_batch(
#             queries_embeddings,
#             batch_size=batch_size,
#             desc=f"Retrieving documents (bs={batch_size})",
#         ):
#             print("Using index retrieval")
#             retrieved_elements = self.index(
#                 queries_embeddings=queries_embeddings_batch,
#                 k=k_token,
#             )
#             print("Got retrieved elements")
#             documents_ids = [
#                 list(
#                     set(
#                         [
#                             document_id
#                             for query_token_document_ids in query_documents_ids
#                             for document_id in query_token_document_ids
#                         ]
#                     )
#                 )
#                 for query_documents_ids in retrieved_elements["documents_ids"]
#             ]
#             print("getting embeddings again")
#             documents_embeddings = self.index.get_documents_embeddings(documents_ids)

#             print("reranking")
#             reranking_results.extend(
#                 divrerank(
#                     documents_ids=documents_ids,
#                     queries_embeddings=queries_embeddings_batch,
#                     documents_embeddings=documents_embeddings,
#                     device=device,
#                 )
#             )
#             breakpoint()

#         return [query_results[:k] for query_results in reranking_results]
