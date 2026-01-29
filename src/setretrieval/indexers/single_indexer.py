from setretrieval.inference.easy_indexer import EasyIndexerBase

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm import LLM
import torch
import os
from datasets import Dataset
import numpy as np
import faiss


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
    def index_documents(self, documents, index_id, emb_bsize=64, pre_embeds=None, redo=False):
        os.makedirs(self.index_base_path, exist_ok=True)
        if self.index_exists(index_id) and redo == False:
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

        if "google-bert" in self.model_name or "e5-large" in self.model_name or "stella" in self.model_name:
            print("doing truncation where necessary")
            # tokenize / detokenize to get things less than 510 tokens long
            docs = [self.toker.encode(doc) for doc in documents]
            # print(max([len(d) for d in docs]))
            # breakpoint()
            docs = [self.toker.decode(doc[:480], skip_special_tokens=True) for doc in docs]
            documents = docs
            sanlens = [len(self.toker.encode(doc)) for doc in documents]
            print("Max num of tokens is ", max(sanlens))
            # breakpoint()
            # breakpoint()
        insmods = ['Qwen', 'intfloat/multilingual-e5-large-instruct', 'NovaSearch/stella_en_1.5B_v5']
        isinmod = any(ins in self.model_name for ins in insmods)
        # breakpoint()
        if self.model_name == "nomic-ai/nomic-embed-text-v1":
            # preprocess documents with prefix search_document: 
            documents = [f"search_{qtype}: {doc}" for doc in documents]
        elif isinmod and qtype == "query": # TODO I think chat template might be the issue
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
    def search(self, queries, index_id, k=10, doembed=True, qtype="query"):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"

        query_embedding = self.embed_with_multi_gpu(queries, qtype=qtype) if doembed else queries
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