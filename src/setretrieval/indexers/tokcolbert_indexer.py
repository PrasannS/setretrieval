
from setretrieval.inference.colbert_indexer import ColBERTEasyIndexer

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm import LLM
import torch
import os
from datasets import Dataset
import numpy as np
import faiss
from setretrieval.utils.utils import pickdump, pickload


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