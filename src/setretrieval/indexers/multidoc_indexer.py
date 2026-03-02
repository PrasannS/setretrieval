"""Multi-document indexer that groups documents into concatenated chunks before indexing.

Creates a corpus of N-document chunks (each original doc appears in J chunks),
indexes those chunks with a provided base indexer, and aggregates chunk results
back to original documents at query time.
"""
import os
import random
import numpy as np
import bm25s
import Stemmer
from tqdm import tqdm
from setretrieval.utils.utils import pickdump, pickload
from datasets import Dataset
from setretrieval.indexers.easy_indexer import EasyIndexerBase


class MultiDocIndexer(EasyIndexerBase):
    """Wraps a base indexer by creating a corpus of concatenated document chunks.

    Each original document appears in J chunks (j_repeats), each chunk
    contains N concatenated documents (n_per_chunk).

    Two grouping strategies:
    - 'random': randomly shuffles doc indices J times and partitions into n_per_chunk groups.
    - 'bm25': runs J rounds of greedy BM25 grouping — each doc is paired with its
              most BM25-similar unassigned neighbors in that round.

    At search time, top k_chunks are retrieved from the base indexer and
    aggregated back to original documents by occurrence count (score as tie-breaker).
    """

    def __init__(self, base_indexer, n_per_chunk=5, j_repeats=3,
                 grouping='random', index_base_path='propercache/cache/multidoc_indices',
                 chunk_sep='\n\n'):
        """
        Args:
            base_indexer: Instantiated indexer (e.g. ColBERTMaxSimIndexer) for chunk retrieval.
            n_per_chunk: Number of documents concatenated per chunk (N).
            j_repeats: Number of chunks each document appears in (J).
            grouping: 'random' or 'bm25'.
            index_base_path: Directory for chunk metadata (separate from base indexer cache).
            chunk_sep: Separator string inserted between documents within a chunk.
        """
        self.base_indexer = base_indexer
        self.n_per_chunk = n_per_chunk
        self.j_repeats = j_repeats
        self.grouping = grouping
        self.index_base_path = index_base_path
        self.chunk_sep = chunk_sep
        self.chunk_to_doc_map = {}  # index_id -> list[list[int]]  (chunk_idx -> [doc_ids])
        self.doc_counts = {}         # index_id -> int (number of original docs)
        self.indexaffix = f"multidoc_{n_per_chunk}_{j_repeats}_{grouping}"
        self.documents = {}
        super().__init__(self.base_indexer.model_name, self.index_base_path)


    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, f"{index_id}_{self.indexaffix}_chunk_map.pkl"))

    def _create_random_groups(self, n_docs):
        """Each doc appears j_repeats times; randomly shuffle and partition into n_per_chunk groups."""
        doc_indices = list(range(n_docs)) * self.j_repeats
        random.shuffle(doc_indices)
        # Pad to a multiple of n_per_chunk
        remainder = len(doc_indices) % self.n_per_chunk
        if remainder:
            doc_indices.extend(random.choices(range(n_docs), k=self.n_per_chunk - remainder))
        return [doc_indices[i:i + self.n_per_chunk]
                for i in range(0, len(doc_indices), self.n_per_chunk)]

    def _create_bm25_groups(self, documents):
        """
        Same functionality as original, but significantly faster.
        """

        n_docs = len(documents)
        stemmer = Stemmer.Stemmer("english")

        print("Building BM25 index for grouping...")

        # --- Tokenize ONCE ---
        tokenized = bm25s.tokenize(documents, stopwords="en", stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(tokenized)

        # --- Precompute full ranking ONCE ---
        print("Precomputing BM25 rankings...")
        all_results, _ = retriever.retrieve(tokenized, k=n_docs, show_progress=True, n_threads=0)
        # all_results shape: [n_docs, n_docs]

        all_groups = []

        for _ in tqdm(range(self.j_repeats), desc="BM25 grouping rounds"):

            # numpy bool array is MUCH faster than python list
            assigned = np.zeros(n_docs, dtype=bool)

            groups_this_round = []

            order = np.random.permutation(n_docs)

            for seed_id in order:

                if assigned[seed_id]:
                    continue

                assigned[seed_id] = True
                group = [int(seed_id)]

                if self.n_per_chunk > 1:

                    # Use precomputed ranking
                    candidates = all_results[seed_id]

                    for cand_id in candidates:
                        if len(group) >= self.n_per_chunk:
                            break

                        cand_id = int(cand_id)

                        if not assigned[cand_id]:
                            assigned[cand_id] = True
                            group.append(cand_id)

                # Padding (unchanged semantics)
                if len(group) < self.n_per_chunk:
                    pad_size = self.n_per_chunk - len(group)
                    group.extend(
                        np.random.randint(0, n_docs, size=pad_size).tolist()
                    )

                groups_this_round.append(group)

            all_groups.extend(groups_this_round)

        return all_groups

    def index_documents(self, documents, index_id, redo=False, **kwargs):
        """Build chunk corpus and index it with the base indexer.

        Args:
            documents: List of document strings to index.
            index_id: Unique identifier for this index.
            redo: If True, re-index even if a cached index exists.
            **kwargs: Passed through to base_indexer.index_documents.
        """
        os.makedirs(self.index_base_path, exist_ok=True)

        self.documents[index_id] = Dataset.from_dict({"text": documents})
        self.documents[index_id].save_to_disk(os.path.join(self.index_base_path, f"{index_id}_{self.indexaffix}"))

        if self.index_exists(index_id) and not redo:
            print(f"MultiDocIndex {index_id} already exists")
            return

        n_docs = len(documents)
        print(f"Creating {self.grouping} groups: {n_docs} docs, "
              f"n_per_chunk={self.n_per_chunk}, j_repeats={self.j_repeats}")

        if self.grouping == 'random':
            groups = self._create_random_groups(n_docs)
        elif self.grouping == 'bm25':
            groups = self._create_bm25_groups(documents)
        else:
            raise ValueError(f"Unknown grouping: '{self.grouping}'. Choose 'random' or 'bm25'.")

        chunk_texts = [
            self.chunk_sep.join(documents[doc_id] for doc_id in group)
            for group in groups
        ]

        print(f"Created {len(chunk_texts)} chunks from {n_docs} documents "
              f"(expected ~{n_docs * self.j_repeats // self.n_per_chunk})")
        self.base_indexer.index_documents(chunk_texts, index_id+f"_{self.indexaffix}", redo=redo, **kwargs)


        pickdump(groups, os.path.join(self.index_base_path, f"{index_id}_{self.indexaffix}_chunk_map.pkl"))
        pickdump(n_docs, os.path.join(self.index_base_path, f"{index_id}_{self.indexaffix}_ndocs.pkl"))

        self.chunk_to_doc_map[index_id] = groups
        self.doc_counts[index_id] = n_docs


    def try_load_cached(self, index_id):
        self.base_indexer.try_load_cached(index_id+f"_{self.indexaffix}")
        if index_id not in self.chunk_to_doc_map:
            if not self.index_exists(index_id):
                raise ValueError(f"MultiDocIndex {index_id} does not exist")
            self.chunk_to_doc_map[index_id] = pickload(
                os.path.join(self.index_base_path, f"{index_id}_{self.indexaffix}_chunk_map.pkl"))
            self.doc_counts[index_id] = pickload(
                os.path.join(self.index_base_path, f"{index_id}_{self.indexaffix}_ndocs.pkl"))
            print(f"Loaded MultiDocIndex {index_id}: "
                  f"{len(self.chunk_to_doc_map[index_id])} chunks, "
                  f"{self.doc_counts[index_id]} original docs")

    def search(self, queries, index_id, k=100, k_chunks=100):
        """Retrieve top-k original documents by aggregating over chunk results.

        Fetches k_chunks from the base indexer, then for each original document
        counts how many retrieved chunks it appears in. Final ranking is by
        occurrence count (descending), with max chunk score as tie-breaker.

        Args:
            queries: List of query strings.
            index_id: Index identifier.
            k: Number of original documents to return per query.
            k_chunks: Number of chunks to retrieve from base indexer before aggregation.

        Returns:
            List (one per query) of dicts with keys: index, score, count, index_id.
        """
        self.try_load_cached(index_id)
        assert isinstance(queries, list), "Queries must be a list"

        chunk_results_all = self.base_indexer.search(queries, index_id+f"_{self.indexaffix}", k=k_chunks)

        groups = self.chunk_to_doc_map[index_id]
        n_docs = self.doc_counts[index_id]

        all_results = []
        for chunk_results in chunk_results_all:
            doc_occurrence_counts = np.zeros(n_docs, dtype=np.int32)
            doc_max_scores = np.full(n_docs, -np.inf, dtype=np.float64)

            for cr in chunk_results:
                chunk_idx = cr['index']
                score = cr['score']
                for doc_id in groups[chunk_idx]:
                    doc_occurrence_counts[doc_id] += 1
                    if score > doc_max_scores[doc_id]:
                        doc_max_scores[doc_id] = score

            # Rank by (occurrence count desc, max score desc)
            seen_doc_ids = np.where(doc_occurrence_counts > 0)[0]
            sorted_doc_ids = sorted(
                seen_doc_ids,
                key=lambda d: (doc_occurrence_counts[d], doc_max_scores[d]),
                reverse=True
            )[:k]

            all_results.append([{
                'index': int(doc_id),
                'score': float(doc_max_scores[doc_id]),
                'count': int(doc_occurrence_counts[doc_id]),
                'index_id': index_id,
            } for doc_id in sorted_doc_ids])

        return all_results
