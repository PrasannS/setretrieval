"""MultiChunkIndexer: wraps any EasyIndexerBase to support multi-variant documents.

Each document is a list of text strings (chunks / paraphrases / variants).
All chunks are indexed in the underlying base indexer; search aggregates
chunk-level scores back to document level (max-score-over-chunks wins),
so retrieving *any* chunk counts as retrieving that document.

The chunk→document mapping is persisted alongside the base index files so
that re-loading works transparently across runs.
"""

import os

from setretrieval.utils.utils import pickdump, pickload


class MultiChunkIndexer:
    """
    Wraps any EasyIndexerBase subclass to support multi-chunk documents.

    Each "document" is indexed as a list of text variants; the first entry
    is treated as the canonical representation, but all variants are stored
    in the underlying FAISS/BM25/etc. index.  Search results are returned at
    the document level (max score over all chunks of that document).

    Args:
        base_indexer: An instantiated EasyIndexerBase subclass
                      (e.g. ColBERTMaxSimIndexer, SingleDenseIndexer, …).

    Example::

        base    = ColBERTMaxSimIndexer(model_name="…")
        indexer = MultiChunkIndexer(base)

        # Each document is [original, paraphrase_1, paraphrase_2, …]
        documents = [
            ["The cat sat on the mat.", "A feline was resting on the rug."],
            ["Paris is the capital of France."],      # single-chunk is fine too
        ]
        index_id = indexer.index_documents(documents, "my_index")
        results  = indexer.search(queries, "my_index", k=10)
        # results[i][j]["index"] → document index (not chunk index)
    """

    def __init__(self, base_indexer):
        self.base = base_indexer
        self._chunk_to_doc: dict[str, list[int]] = {}  # index_id → list[doc_id]
        self._num_docs: dict[str, int] = {}             # index_id → n_docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_path(self, index_id: str) -> str:
        """Path where the chunk→doc mapping pickle is stored."""
        return os.path.join(self.base.index_base_path, f"{index_id}_multichunk_map.pkl")

    def _save_map(self, index_id: str) -> None:
        pickdump(
            (self._chunk_to_doc[index_id], self._num_docs[index_id]),
            self._map_path(index_id),
        )

    def _ensure_map_loaded(self, index_id: str) -> None:
        if index_id in self._chunk_to_doc:
            return
        path = self._map_path(index_id)
        if not os.path.exists(path):
            raise ValueError(
                f"No chunk map found for index '{index_id}' at {path}. "
                "Call index_documents() first."
            )
        chunk_to_doc, num_docs = pickload(path)
        self._chunk_to_doc[index_id] = chunk_to_doc
        self._num_docs[index_id] = num_docs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_exists(self, index_id: str) -> bool:
        """True iff both the base index and the chunk map exist on disk."""
        return (
            self.base.index_exists(index_id)
            and os.path.exists(self._map_path(index_id))
        )

    def index_documents(self, documents: list, index_id: str, **kwargs) -> str:
        """
        Index multi-chunk documents.

        Args:
            documents: List of documents. Each document is a non-empty list
                       of strings (text variants). The first string is the
                       canonical representation; all are indexed.
            index_id:  Identifier for this index (forwarded to the base indexer).
            **kwargs:  Forwarded verbatim to base_indexer.index_documents().

        Returns:
            index_id (for convenience, mirrors EasyIndexerBase.index_dataset).
        """
        flat_texts: list[str] = []
        chunk_to_doc: list[int] = []
        for doc_id, chunks in enumerate(documents):
            for chunk in chunks:
                flat_texts.append(chunk)
                chunk_to_doc.append(doc_id)

        avg = len(flat_texts) / max(len(documents), 1)
        print(
            f"MultiChunkIndexer: {len(documents)} docs → "
            f"{len(flat_texts)} chunks ({avg:.2f} avg/doc)"
        )

        self.base.index_documents(flat_texts, index_id, **kwargs)

        self._chunk_to_doc[index_id] = chunk_to_doc
        self._num_docs[index_id] = len(documents)
        self._save_map(index_id)
        return index_id

    def index_dataset(self, dset_path: str, redo: bool = False) -> str:
        """
        Index a HuggingFace Dataset from disk.

        Expects either:
          - a ``chunks`` column (``list[list[str]]``) — multi-chunk mode
          - a ``text``   column (``list[str]``)        — treated as single-chunk

        Returns the index_id (``dset_path.replace("/", "_")``), matching the
        convention of EasyIndexerBase.index_dataset.
        """
        from datasets import Dataset

        dset = Dataset.load_from_disk(dset_path)
        index_id = dset_path.replace("/", "_")

        if "chunks" in dset.column_names:
            documents = list(dset["chunks"])
        else:
            documents = [[text] for text in dset["text"]]

        self.index_documents(documents, index_id, redo=redo)
        return index_id

    def search(
        self,
        queries: list,
        index_id: str,
        k: int = 100,
        oversample_factor: int = 5,
        **kwargs,
    ) -> list:
        """
        Search queries and return document-level results.

        Internally retrieves ``k * oversample_factor`` chunks from the base
        indexer, maps each chunk back to its parent document, and keeps the
        maximum score over all chunks of each document (max-over-chunks rule).

        Args:
            queries:           List of query strings.
            index_id:          Index to search.
            k:                 Number of top documents to return per query.
            oversample_factor: Multiplier for chunk-level k to ensure we see
                               enough unique documents after deduplication.
            **kwargs:          Forwarded to base_indexer.search().

        Returns:
            ``List[List[dict]]`` — one inner list per query. Each dict::

                {"index": int,   # document index (not chunk index)
                 "score": float,
                 "index_id": str}
        """
        self._ensure_map_loaded(index_id)
        chunk_to_doc = self._chunk_to_doc[index_id]
        num_chunks = len(chunk_to_doc)

        chunk_k = min(k * oversample_factor, num_chunks)
        chunk_results = self.base.search(queries, index_id, k=chunk_k, **kwargs)

        results = []
        for query_chunk_results in chunk_results:
            doc_scores: dict[int, float] = {}
            for pred in query_chunk_results:
                chunk_idx = int(pred["index"])
                if chunk_idx >= num_chunks:
                    continue  # stale FAISS entry after partial updates
                doc_id = chunk_to_doc[chunk_idx]
                score = float(pred["score"])
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = score

            top_k = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            results.append(
                [{"index": doc_id, "score": score, "index_id": index_id} for doc_id, score in top_k]
            )

        return results
