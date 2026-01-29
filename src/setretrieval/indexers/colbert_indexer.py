from setretrieval.inference.easy_indexer import EasyIndexerBase
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm import LLM
import torch
import os
from datasets import Dataset
import numpy as np
import faiss
from pylate import indexes, models, retrieve
from setretrieval.utils.utils import pickdump, pickload

class ColBERTEasyIndexer(EasyIndexerBase):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='propercache/cache/colbert_indices', gpu_list=None, div_colbert=False, qmod_name=None, qvecs=-1, dvecs=-1, use_bsize=128, usefast=True):
        # If gpu_list is None, default to single GPU
        if gpu_list is None:
            gpu_list = [0]
        self.div_colbert = div_colbert
        self.gpu_list = list(range(torch.cuda.device_count()))
        self.num_workers = len(self.gpu_list)
        self.qmodel = None
        self.qmod_name = qmod_name
        self.qvecs = qvecs
        self.dvecs = dvecs
        self.use_bsize = use_bsize

        self.embsize = 512 if "embsize512" in model_name else 128
        self.embsize = 8 if "embsize8" in model_name else self.embsize
        self.usefast = usefast == "yes"

        # TODO might be able to do this automatically
        if self.qvecs > 0 and self.dvecs > 0: 
            # do monkey patching
            print("Monkey patching ColBERT model")
            models.ColBERT.tokenize = padded_tokenize
            models.ColBERT.old_forward = models.ColBERT.forward
            models.ColBERT.forward = newforward
        
        models.ColBERT.old_encode = models.ColBERT.encode
        # get rid of zero vectors, should be harmless anyways
        models.ColBERT.encode = mod_encode

        models.ColBERT.old_tokenize = models.ColBERT.tokenize
        models.ColBERT.tokenize = modadj_tokenize

        self.islora = "adapter_config.json" in os.listdir(model_name)
        
        # TODO this will break for multiple model case
        # breakpoint()
        # Initialize base class (you may need to adjust this depending on EasyIndexerBase)
        super().__init__(model_name, index_base_path, self.num_workers)
        self.models = []  # Store multiple model instances
        # breakpoint()

    def evmload(self, name, device=None):
        if device is None:
            device = "cuda"
        if self.islora:
            print("Loading LoRA model")
            # figure out base model name from adapter_config.json
            with open(os.path.join(name, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config["base_model_name_or_path"]
            adapter_name = name
            name = base_model_name
        mod = models.ColBERT(model_name_or_path=name, device=device)
        mod.eval()

        if self.islora:
            if mod.tokenizer.cls_token is None:
                cls_token = '[EMB]'
                mod.tokenizer.cls_token_id = mod.tokenizer.add_tokens(cls_token)
                mod.tokenizer.cls_token = cls_token
                mod._first_module().auto_model.resize_token_embeddings(len(mod.tokenizer))
            print("Loading adapter")
            mod.load_adapter(adapter_name)
        
        if self.qvecs > 0 and self.dvecs > 0:
            mod.tokenizer.query_vectors = self.qvecs
            mod.tokenizer.doc_vectors = self.dvecs
    
        return mod

    def load_model(self):
        if self.model is None:
            print("Loading ColBERT model")
            self.model = self.evmload(self.model_name)
        if self.qmodel is None and self.qmod_name is not None:
            self.qmodel = self.evmload(self.qmod_name)
            print(f"Loaded query model {self.qmod_name}")

    def load_models_for_workers(self):
        """Load a separate model instance for each worker (can have multiple workers per GPU)"""
        if len(self.models) == 0:
            print(f"Loading {self.num_workers} model instances across GPUs: {set(self.gpu_list)}")
            for worker_id, gpu_id in enumerate(tqdm(self.gpu_list)):
                model = self.evmload(self.model_name, device=f'cuda:{gpu_id}')
                self.models.append((model, gpu_id, worker_id))

    def embed_with_multi_gpu(self, documents, qtype="document"):
        assert qtype in ['document', 'query'] and type(documents) == list
        
        # Use single GPU if only one worker or small dataset
        if self.num_workers <= 1 or len(documents) < self.use_bsize * 2 * self.num_workers or (self.qmod_name is not None and qtype == "query"):
            self.load_model()
            if qtype == "query":
                return self.model.old_encode(documents, batch_size=self.use_bsize, is_query=qtype=="query", show_progress_bar=True)
            else:
                return self.model.old_encode(documents, batch_size=self.use_bsize, is_query=qtype=="query", show_progress_bar=True)
        
        # Multi-worker encoding
        self.load_models_for_workers()
        # breakpoint()
        
        # Split documents across workers
        chunk_size = (len(documents) + self.num_workers - 1) // self.num_workers
        document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
        
        def encode_on_worker(model_gpu_worker, docs):
            model, gpu_id, worker_id = model_gpu_worker
            # Ensure model is on correct GPU
            if qtype == "query":
                return model.old_encode(docs, batch_size=self.use_bsize, is_query=qtype=="query", show_progress_bar=True)
            else:
                return model.old_encode(docs, batch_size=self.use_bsize, is_query=qtype=="query", show_progress_bar=True)
        
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
        # HACK to avoid having to use fast-plaid
        return os.path.exists(os.path.join(self.index_base_path, f"{index_id}"))

    def index_documents(self, documents, index_id, pre_embeds=None, redo=False):
        os.makedirs(self.index_base_path, exist_ok=True)
        if self.index_exists(index_id) and redo == False:
            print(f"Index {index_id} already exists")
            return

        if pre_embeds is None:
            embeds = self.embed_with_multi_gpu(documents, qtype="document")
        else:
            embeds = pre_embeds
        print("Adding documents to index (new)")
        self.indices[index_id] = indexes.PLAID(index_folder=self.index_base_path, index_name=index_id, override=True, use_fast=self.usefast, embedding_size=self.embsize)
        self.indices[index_id].add_documents(documents_ids=[str(i) for i in range(len(documents))], documents_embeddings=embeds)
        self.documents[index_id] = Dataset.from_dict({"text": documents})
        self.documents[index_id].save_to_disk(os.path.join(self.index_base_path, f"{index_id}"))
        print(f"Indexed {len(documents)} documents for index {index_id}")

    def try_load_cached(self, index_id):
        # breakpoint()
        if index_id not in self.indices:
            if self.index_exists(index_id) == False:
                raise ValueError(f"Index {index_id} does not exist")
            # TODO set up some hardcoded stuff here
            print('New edit')
            self.indices[index_id] = indexes.PLAID(index_folder=self.index_base_path, index_name=index_id, override=True, use_fast=self.usefast, embedding_size=self.embsize)
            self.documents[index_id] = Dataset.load_from_disk(os.path.join(self.index_base_path, f"{index_id}"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        return self.indices[index_id]

    def search(self, queries, index_id, k=10, doembed=True):
        self.try_load_cached(index_id)
        assert type(queries) == list, "Queries must be a list"
        query_embedding = self.embed_with_multi_gpu(queries, qtype="query") if doembed else np.array(queries)
        # breakpoint()
        retriever = retrieve.ColBERT(index=self.indices[index_id])
        results = retriever.retrieve(queries_embeddings=query_embedding, k=min(k, len(self.documents[index_id])))
        # breakpoint()
        return [[{'score': entry['score'], 'index': int(entry['id']), 'index_id': index_id} for entry in result] for result in results]

