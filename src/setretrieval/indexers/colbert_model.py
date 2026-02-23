"""Mixin providing ColBERT model loading, LoRA support, and multi-GPU embedding.

Extracted from ColBERTEasyIndexer and SingColBERTEasyIndexer to eliminate
~100 lines of duplicated code. Classes using this mixin should call
`_init_colbert_model()` in their `__init__`.
"""

from pylate import models
from setretrieval.train.pylate_monkeypatch import padded_tokenize, newforward, modadj_tokenize, mod_encode

import torch
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


import re

import re

def extract_all_vals(path: str):
    # Match any of the four keys followed by digits,
    # requiring a non-word boundary before them
    pattern = r'\b(p?qv|p?dv)(\d+)'
    
    matches = re.findall(pattern, path)
    
    # Default values
    result = {"qv": 0, "dv": 0, "pqv": 0, "pdv": 0}
    
    for key, value in matches:
        result[key] = int(value)
    
    return result["qv"], result["dv"], result["pqv"], result["pdv"]

class ColBERTModelMixin:
    """Mixin that provides ColBERT model loading, LoRA support, and multi-GPU embedding.

    Classes using this mixin must also inherit from EasyIndexerBase (or a subclass),
    which provides `self.model_name`, `self.model`, `self.indices`, etc.
    """

    

    # get rid of manually needing to specify qvecs, dvecs, passiveqvecs, passivedvecs
    def _init_colbert_model(self, model_name, qmod_name=None, qvecs=-1, dvecs=-1, passiveqvecs=0, passivedvecs=0, use_bsize=128, usefast=True):
        """Initialize ColBERT model configuration. Call this in subclass __init__."""
        self.gpu_list = list(range(torch.cuda.device_count()))
        self.num_workers = len(self.gpu_list)
        self.qmodel = None
        self.qmod_name = qmod_name

        qv, dv, pqv, pdv = extract_all_vals(model_name)
        print(f"Extracted values: qv={qv}, dv={dv}, pqv={pqv}, pdv={pdv}")
        self.qvecs = qv
        self.dvecs = dv
        self.use_bsize = use_bsize
        self.passiveqvecs = pqv
        self.passivedvecs = pdv
        self.models = []

        # breakpoint()

        # Parse embedding size from model name (e.g. "model-embsize128")
        if "embsize" in model_name:
            es = model_name.split("embsize")[-1]
            for sep in "/", "-":
                if sep in es:
                    es = es.split(sep)[0]
            self.embsize = int(es)
        else:
            print("No embsize found in model name, defaulting to 128")
            self.embsize = 128

        self.usefast = usefast == "yes"

        # Monkey patching for fixed-length query/doc vectors
        if self.qvecs > -1 and self.dvecs > -1:
            print("Monkey patching ColBERT model")
            models.ColBERT.tokenize = padded_tokenize
            models.ColBERT.old_forward = models.ColBERT.forward
            models.ColBERT.forward = newforward

        # Patch encode to remove zero vectors
        models.ColBERT.old_encode = models.ColBERT.encode
        models.ColBERT.encode = mod_encode
        models.ColBERT.old_tokenize = models.ColBERT.tokenize
        models.ColBERT.tokenize = modadj_tokenize

        self.islora = os.path.exists(os.path.join(model_name, "adapter_config.json"))

    def evmload(self, name, device=None):
        """Load a ColBERT model instance, with optional LoRA adapter support."""
        if device is None:
            device = "cuda"

        adapter_name = None
        if self.islora:
            print("Loading LoRA model")
            with open(os.path.join(name, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            adapter_name = name
            name = adapter_config["base_model_name_or_path"]

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

        if self.qvecs > -1 and self.dvecs > -1:
            mod.tokenizer.query_vectors = self.qvecs
            mod.tokenizer.doc_vectors = self.dvecs
            mod.tokenizer.qpass_vecs = self.passiveqvecs
            mod.tokenizer.dpass_vecs = self.passivedvecs

        return mod

    def load_model(self):
        """Load the primary ColBERT model and optional separate query model."""
        if self.model is None:
            print("Loading ColBERT model")
            self.model = self.evmload(self.model_name)
        if self.qmodel is None and self.qmod_name is not None:
            self.qmodel = self.evmload(self.qmod_name)
            print(f"Loaded query model {self.qmod_name}")

    def load_models_for_workers(self):
        """Load a separate model instance for each GPU worker."""
        if len(self.models) == 0:
            print(f"Loading {self.num_workers} model instances across GPUs: {set(self.gpu_list)}")
            for worker_id, gpu_id in enumerate(tqdm(self.gpu_list)):
                model = self.evmload(self.model_name, device=f'cuda:{gpu_id}')
                self.models.append((model, gpu_id, worker_id))

    def embed_with_multi_gpu(self, documents, qtype="document", batch_size=None):
        """Encode documents/queries using ColBERT, with multi-GPU parallelism for large batches."""
        assert qtype in ['document', 'query'] and isinstance(documents, list)

        use_bsize = batch_size if batch_size is not None else self.use_bsize

        # Use single GPU for small batches or when using a separate query model
        if (self.num_workers <= 1 or
                len(documents) < use_bsize * 2 * self.num_workers or
                (self.qmod_name is not None and qtype == "query")):
            self.load_model()
            return self.model.encode(
                documents, batch_size=use_bsize,
                is_query=(qtype == "query"), show_progress_bar=True
            )

        # Multi-worker encoding
        self.load_models_for_workers()

        chunk_size = (len(documents) + self.num_workers - 1) // self.num_workers
        document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]

        def encode_on_worker(model_gpu_worker, docs):
            model, gpu_id, worker_id = model_gpu_worker
            return model.encode(
                docs, batch_size=use_bsize,
                is_query=(qtype == "query"), show_progress_bar=True
            )

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(encode_on_worker, self.models[i], chunk)
                for i, chunk in enumerate(document_chunks) if len(chunk) > 0
            ]
            results = [future.result() for future in futures]

        allresults = []
        for result in results:
            allresults.extend(result)
        return allresults
