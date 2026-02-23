"""ColBERT indexers using FAISS for token-level retrieval.

Contains a shared base class (ColBERTFaissTokenIndexer) that handles the common
token-level FAISS indexing logic, plus two search strategies:
  - TokenColBERTEasyIndexer: best-score-per-document
  - ColBERTMaxSimIndexer: MaxSim (sum of max per query token)
"""

import os
import numpy as np
import faiss
from datasets import Dataset
from tqdm import tqdm

from setretrieval.indexers.easy_indexer import EasyIndexerBase
from setretrieval.indexers.colbert_model import ColBERTModelMixin
from setretrieval.utils.utils import pickdump, pickload


class ColBERTFaissTokenIndexer(ColBERTModelMixin, EasyIndexerBase):
    """Base class for ColBERT indexers that use FAISS with per-token vectors.

    Subclasses only need to implement `search()`.
    """

    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1',
                 index_base_path='propercache/cache/faiss_colbert_indices',
                 qmod_name=None, qvecs=-1, dvecs=-1, passiveqvecs=0, passivedvecs=0, use_bsize=128, usefast=True):
        self._init_colbert_model(model_name, qmod_name, qvecs, dvecs, passiveqvecs, passivedvecs, use_bsize, usefast)
        EasyIndexerBase.__init__(self, model_name, index_base_path, self.num_workers)
        self.token_to_doc_map = {}

    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, f"{index_id}.faiss"))

    def index_documents(self, documents, index_id, emb_bsize=64, pre_embeds=None, redo=False):
        os.makedirs(self.index_base_path, exist_ok=True)
        if self.index_exists(index_id) and not redo:
            print(f"Index {index_id} already exists")
            return

        embeds = pre_embeds if pre_embeds is not None else self.embed_with_multi_gpu(documents, qtype="document")

        # Flatten all token vectors and create mapping
        all_token_vectors = []
        token_map = []  # List of (doc_id, token_idx) tuples

        print("Flattening token vectors...")
        for doc_id, doc_embed in enumerate(tqdm(embeds)):
            num_tokens = doc_embed.shape[0]
            for token_idx in range(num_tokens):
                all_token_vectors.append(doc_embed[token_idx])
                token_map.append((doc_id, token_idx))

        all_token_vectors = np.array(all_token_vectors, dtype=np.float32)
        print(f"Total token vectors: {len(all_token_vectors)}, dimension: {all_token_vectors.shape[1]}")

        # Create FAISS index
        print("Building FAISS index...")
        embedding_dim = all_token_vectors.shape[1]
        print(f"Number of token vectors: {len(all_token_vectors)} for {len(documents)} documents")
        base_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for ColBERT
        index = faiss.IndexIDMap(base_index)

        ids = np.arange(len(all_token_vectors)).astype(np.int64)
        index.add_with_ids(all_token_vectors, ids)

        # Save index and mappings
        faiss.write_index(index, os.path.join(self.index_base_path, f"{index_id}.faiss"))
        pickdump(token_map, os.path.join(self.index_base_path, f"{index_id}_token_map.pkl"))
        pickdump(list(embeds), os.path.join(self.index_base_path, f"{index_id}_doc_embeds.pkl"))

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


class TokenColBERTEasyIndexer(ColBERTFaissTokenIndexer):
    """ColBERT FAISS indexer with best-score-per-document search."""

    def search(self, queries, index_id, k=100):
        self.try_load_cached(index_id)
        self.load_model()
        assert isinstance(queries, list), "Queries must be a list"

        print("Encoding queries...")
        query_embeds = self.model.encode(queries, batch_size=32, is_query=True, show_progress_bar=True)

        results = []
        token_map = self.token_to_doc_map[index_id]

        for query_embed in tqdm(query_embeds, desc="Searching"):
            num_query_tokens = query_embed.shape[0]
            k_per_token = k

            scores, indices = self.indices[index_id].search(query_embed, k_per_token)

            doc_matches = {}
            for query_token_idx in range(num_query_tokens):
                for j in range(k_per_token):
                    doc_token_idx = indices[query_token_idx][j]
                    doc_id = int(token_map[doc_token_idx][0])
                    token_idx = int(token_map[doc_token_idx][1])
                    score = float(scores[query_token_idx][j])

                    if doc_id not in doc_matches or score > doc_matches[doc_id]['score']:
                        doc_matches[doc_id] = {
                            'score': score,
                            'index': doc_id,
                            'token_idx': token_idx,
                            'query_token_idx': query_token_idx,
                            'index_id': index_id
                        }

            sorted_matches = sorted(doc_matches.values(), key=lambda x: x['score'], reverse=True)[:k]
            results.append(sorted_matches)

        return results


class ColBERTMaxSimIndexer(ColBERTFaissTokenIndexer):
    """ColBERT FAISS indexer with MaxSim (sum of max per query token) scoring."""

    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1',
                 index_base_path='propercache/cache/colbert_indices',
                 qmod_name=None, qvecs=-1, dvecs=-1, passiveqvecs=0, passivedvecs=0, use_bsize=128, usefast=True, detailed_save="no"):
        super().__init__(model_name, index_base_path, qmod_name, qvecs, dvecs, passiveqvecs, passivedvecs, use_bsize, usefast)
        self.detailed_save = detailed_save

    def search(self, queries, index_id, k=100, k_tokens=1000, indbsize=16):
        """
        Standard ColBERT search using MaxSim scoring.

        Args:
            queries: List of query strings
            index_id: Index identifier
            k: Number of top documents to return
            k_tokens: Number of top token matches to retrieve per query token
        """
        self.try_load_cached(index_id)
        self.load_model()
        assert isinstance(queries, list), "Queries must be a list"

        print("Encoding queries...")
        query_embeds = self.model.encode(queries, batch_size=32, is_query=True, show_progress_bar=True)

        results = []
        token_map = self.token_to_doc_map[index_id]
        num_docs = len(self.documents[index_id])
        detailed_preds = []

        for batch_start in tqdm(range(0, len(query_embeds), indbsize), desc="Searching with MaxSim"):
            batch_query_embeds = query_embeds[batch_start:batch_start + indbsize]

            # Keep track of how many tokens each query has
            token_counts = [qe.shape[0] for qe in batch_query_embeds]

            # Concatenate into one big matrix
            concat_query_embed = np.concatenate(batch_query_embeds, axis=0)

            # 🔹 SINGLE batched FAISS call
            scores_all, indices_all = self.indices[index_id].search(concat_query_embed, k_tokens)

            detailed_preds.append((scores_all, indices_all))

            # Now split results back per query
            offset = 0
            for qi, num_query_tokens in enumerate(token_counts):
                scores = scores_all[offset:offset + num_query_tokens]
                indices = indices_all[offset:offset + num_query_tokens]
                offset += num_query_tokens

                # ---- FROM HERE DOWN YOUR ORIGINAL CODE IS IDENTICAL ----
                doc_scores = np.zeros(num_docs, dtype=np.float32)

                for query_token_idx in range(num_query_tokens):
                    token_scores = scores[query_token_idx]
                    token_indices = indices[query_token_idx]

                    doc_max_for_query_token = {}
                    for j in range(k_tokens):
                        doc_token_idx = token_indices[j]
                        doc_id = int(token_map[doc_token_idx][0])
                        score = float(token_scores[j])

                        if doc_id not in doc_max_for_query_token or score > doc_max_for_query_token[doc_id]:
                            doc_max_for_query_token[doc_id] = score

                    for doc_id, max_score in doc_max_for_query_token.items():
                        doc_scores[doc_id] += max_score

                top_k_doc_ids = np.argsort(doc_scores)[::-1][:k]
                query_results = [{
                    'index': int(doc_id),
                    'score': float(doc_scores[doc_id]),
                    'index_id': index_id
                } for doc_id in top_k_doc_ids]

                results.append(query_results)

        if self.detailed_save is not "no":
            pickdump(detailed_preds, os.path.join("propercache/cache/detailed_preds", f"{index_id}_{self.detailed_save}.pkl"))
        return results


    # TODO this was too complicated, maybe do this later or vibe code it later when it's more important.
    # # update the index with a small number of new documents ()
    def smallupdate_index(self, index_id, new_documents, new_id=None):

        self.try_load_cached(index_id)
        self.load_model()

        old_dataset = self.documents[index_id]
        old_texts = [old_dataset[i]["text"] for i in range(len(old_dataset))]

        assert len(old_texts) == len(new_documents), "Currently requires same number of documents"

        # Identify changed documents
        replace_inds = [i for i in range(len(old_texts)) if old_texts[i] != new_documents[i]['text']]

        print(f"Updating {len(replace_inds)} documents")

        # Remove all token ids belonging to changed docs
        token_map = self.token_to_doc_map[index_id]
        repinds = set(replace_inds)

        remove_token_ids = [token_id for token_id, (doc_id, _) in enumerate(token_map) if doc_id in repinds]
        remove_token_ids = np.array(remove_token_ids, dtype=np.int64)

        # breakpoint()
        self.indices[index_id].remove_ids(remove_token_ids)

        # Re-embed changed documents
        changed_texts = [new_documents[i]['text'] for i in replace_inds]

        new_embeds = self.model.encode(
            changed_texts,
            batch_size=32,
            is_query=False,
            show_progress_bar=True
        )

        #  Add new token vectors with NEW unique ids
        current_max_id = len(token_map)
        new_token_ids, new_token_vectors, new_token_entries = [], [], []

        next_id = current_max_id

        for doc_local_idx, doc_id in enumerate(replace_inds):
            doc_embed = new_embeds[doc_local_idx]

            for token_idx in range(doc_embed.shape[0]):
                new_token_vectors.append(doc_embed[token_idx])
                new_token_ids.append(next_id)
                new_token_entries.append((doc_id, token_idx))
                next_id += 1

        # assume that we will have new token vectors
        new_token_vectors = np.array(new_token_vectors, dtype=np.float32)
        new_token_ids = np.array(new_token_ids, dtype=np.int64)

        self.indices[index_id].add_with_ids(new_token_vectors, new_token_ids)

        # Update token_to_doc_map
        # We DO NOT delete old entries — we mark them invalid by setting to None
        for token_id in remove_token_ids:
            token_map[token_id] = None

        # Extend map to accommodate new ids
        if len(token_map) < next_id:
            token_map.extend([None] * (next_id - len(token_map)))

        for token_id, entry in zip(new_token_ids, new_token_entries):
            token_map[token_id] = entry

        self.token_to_doc_map[index_id] = token_map

        # -------------------------------------------------
        # Update document dataset
        # -------------------------------------------------
        self.documents[index_id] = new_documents

        # Persist everything
        if False:
            faiss.write_index(
                self.indices[index_id],
                os.path.join(self.index_base_path, f"{index_id}.faiss")
            )

            pickdump(
                token_map,
                os.path.join(self.index_base_path, f"{index_id}_token_map.pkl")
            )

            self.documents[index_id].save_to_disk(
                os.path.join(self.index_base_path, f"{index_id}")
            )

            print("Index successfully updated.")

