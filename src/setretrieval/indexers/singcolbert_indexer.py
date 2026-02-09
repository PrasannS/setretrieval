"""ColBERT indexer that flattens multi-vector embeddings to single vectors for FAISS search.

Uses ColBERT for encoding but flattens token-level embeddings into a single vector
per document, then indexes/searches with standard FAISS (inherited from SingleEasyIndexer).
"""

import torch

from setretrieval.indexers.single_indexer import SingleEasyIndexer
from setretrieval.indexers.colbert_model import ColBERTModelMixin


class SingColBERTEasyIndexer(ColBERTModelMixin, SingleEasyIndexer):
    """ColBERT model with single-vector FAISS retrieval (flattened token embeddings)."""

    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1', index_base_path='propercache/cache/colbert_indices',
                 qmod_name=None, qvecs=-1, dvecs=-1, use_bsize=128, usefast=True):
        self._init_colbert_model(model_name, qmod_name, qvecs, dvecs, use_bsize, usefast)
        SingleEasyIndexer.__init__(self, model_name, index_base_path, self.num_workers)

    def embed_with_multi_gpu(self, documents, qtype="document", batch_size=None):
        """Encode with ColBERT and flatten multi-vector embeddings to single vectors."""
        embeds = ColBERTModelMixin.embed_with_multi_gpu(self, documents, qtype=qtype, batch_size=batch_size)
        return [torch.tensor(emb).view(-1) for emb in embeds]
