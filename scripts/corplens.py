# print out average number of characters, average number of modernbert tokens per document in 
# normal, long, short fiqa corpora

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from statistics import mean

if __name__ == "__main__":
    toker = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    normal_corpus = Dataset.load_from_disk("propercache/data/datastores/fiqacorpus")
    long_corpus = Dataset.load_from_disk("propercache/data/datastores/fiqacorpus_longv2")
    short_corpus = Dataset.load_from_disk("propercache/data/datastores/fiqacorpus_short")

    normal_chars = [len(doc["text"]) for doc in tqdm(normal_corpus)]
    long_chars = [len(doc["text"]) for doc in tqdm(long_corpus)]
    short_chars = [len(doc["text"]) for doc in tqdm(short_corpus)]

    normal_words = [doc["text"].split() for doc in tqdm(normal_corpus)]
    long_words = [doc["text"].split() for doc in tqdm(long_corpus)]
    short_words = [doc["text"].split() for doc in tqdm(short_corpus)]

    def unique_ngrams(toklist, n=2):
        ngs = set()
        for toks in toklist:
            ngs.update([" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)])
        print(len(ngs))
        return list(ngs)

    # breakpoint()

    normal_tokens = [toker.encode(doc["text"]) for doc in tqdm(normal_corpus)]
    long_tokens = [toker.encode(doc["text"]) for doc in tqdm(long_corpus)]
    short_tokens = [toker.encode(doc["text"]) for doc in tqdm(short_corpus)]

    # breakpoint()
    alltoks_normal = set([item for sublist in normal_tokens for item in sublist])
    alltoks_long = set([item for sublist in long_tokens for item in sublist])
    alltoks_short = set([item for sublist in short_tokens for item in sublist])

    # breakpoint()
    normal_toks_len = [len(toks) for toks in normal_tokens]
    long_toks_len = [len(toks) for toks in long_tokens]
    short_toks_len = [len(toks) for toks in short_tokens]


    print(f"Average number of characters in normal corpus: {sum(normal_chars) / len(normal_chars)}")
    print(f"Average number of tokens in normal corpus: {sum(normal_toks_len) / len(normal_toks_len)}")
    print(f"Average number of characters in long corpus: {sum(long_chars) / len(long_chars)}")
    print(f"Average number of tokens in long corpus: {sum(long_toks_len) / len(long_toks_len)}")
    print(f"Average number of characters in short corpus: {sum(short_chars) / len(short_chars)}")
    print(f"Average number of tokens in short corpus: {sum(short_toks_len) / len(short_toks_len)}")

    print(f"Number of unique tokens in normal corpus: {len(alltoks_normal)}")
    print(f"Number of unique tokens in long corpus: {len(alltoks_long)}")
    print(f"Number of unique tokens in short corpus: {len(alltoks_short)}")