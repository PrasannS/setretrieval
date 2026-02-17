# print out average number of characters, average number of modernbert tokens per document in 
# normal, long, short fiqa corpora

from datasets import Dataset
from transformers import AutoTokenizer

if __name__ == "__main__":
    toker = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    normal_corpus = Dataset.load_from_disk("propercache/data/datastores/fiqacorpus")
    long_corpus = Dataset.load_from_disk("propercache/data/datastores/fiqacorpus_long")
    short_corpus = Dataset.load_from_disk("propercache/data/datastores/fiqacorpus_short")

    normal_chars = [len(doc["text"]) for doc in normal_corpus]
    long_chars = [len(doc["text"]) for doc in long_corpus]
    short_chars = [len(doc["text"]) for doc in short_corpus]

    normal_tokens = [len(toker.encode(doc["text"])) for doc in normal_corpus]
    long_tokens = [len(toker.encode(doc["text"])) for doc in long_corpus]
    short_tokens = [len(toker.encode(doc["text"])) for doc in short_corpus]

    print(f"Average number of characters in normal corpus: {sum(normal_chars) / len(normal_chars)}")
    print(f"Average number of characters in long corpus: {sum(long_chars) / len(long_chars)}")
    print(f"Average number of characters in short corpus: {sum(short_chars) / len(short_chars)}")

    print(f"Average number of tokens in normal corpus: {sum(normal_tokens) / len(normal_tokens)}")
    print(f"Average number of tokens in long corpus: {sum(long_tokens) / len(long_tokens)}")
    print(f"Average number of tokens in short corpus: {sum(short_tokens) / len(short_tokens)}")