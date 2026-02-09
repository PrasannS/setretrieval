from setretrieval.indexers.easy_indexer import EasyIndexerBase
import os
from datasets import Dataset
from setretrieval.utils.utils import pickdump, pickload
import bm25s
import Stemmer
from tqdm import tqdm
import random

class BM25EasyIndexer(EasyIndexerBase):
    def __init__(self, model_name='bm25', index_base_path='propercache/cache/bm25_indices', num_gpus=4):
        super().__init__(model_name, index_base_path, num_gpus)
        self.stemmer = Stemmer.Stemmer("english")

    def load_model(self):
        raise NotImplementedError("BM25EasyIndexer does not need to load a model")

    def index_exists(self, index_id):
        return os.path.exists(os.path.join(self.index_base_path, index_id+".bm25"))
        
    def index_documents(self, documents, index_id, **kwargs):
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
        self.documents[index_id].save_to_disk(os.path.join(self.index_base_path, f"{index_id}"))
        print(f"Indexed {len(documents)} documents for index {index_id}")
    
    def try_load_cached(self, index_id):
        if index_id not in self.indices:
            if not self.index_exists(index_id):
                raise ValueError(f"Index {index_id} does not exist")
            self.indices[index_id] = pickload(os.path.join(self.index_base_path, index_id+".bm25"))
            self.documents[index_id] = Dataset.load_from_disk(os.path.join(self.index_base_path, f"{index_id}"))
            print(f"Loaded index {index_id} from cache with {len(self.documents[index_id])} documents")
        return self.indices[index_id]

    def search(self, queries, index_id, k=10):
        self.try_load_cached(index_id)
        assert isinstance(queries, list), "Queries must be a list"
        queries = [bm25s.tokenize(query, stemmer=self.stemmer) for query in queries]
        bm25 = self.indices[index_id]
        allresults = []
        print("started search")
        for query in tqdm(queries):
            results, scores = bm25.retrieve(query, k=min(k, len(self.documents[index_id])))
            allresults.append([{'score': scores[0][i], 'index': results[0][i], 'index_id': index_id} for i in range(len(results[0]))])
        return allresults

    def composed_search(self, queries, index_ids, k=10, reverse=False):
        return super().composed_search(queries, index_ids, k=k, reverse=True)


class RandomEasyIndexer(BM25EasyIndexer):
    """Random baseline indexer (returns random documents)."""

    def search(self, queries, index_id, k=10):
        self.try_load_cached(index_id)
        assert isinstance(queries, list), "Queries must be a list"
        return [[{'score': random.random(), 'index': random.randint(0, len(self.documents[index_id])-1), 'index_id': index_id} for _ in range(k)] for _ in queries]
