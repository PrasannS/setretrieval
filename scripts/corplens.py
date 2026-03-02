# print out average number of characters, average number of modernbert tokens per document in 
# normal, long, short fiqa corpora

from datasets import Dataset, load_from_disk, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt

if __name__ == "__main__":
    toker = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    normal_corpus = load_from_disk("propercache/data/datastores/fiqacorpus")
    long_corpus = load_from_disk("propercache/data/datastores/fiqacorpus_longv2")
    short_corpus = load_from_disk("propercache/data/datastores/fiqacorpus_short")

    nq_corpus = load_from_disk("propercache/data/datastores/nqcorpus_bm25")

    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(10000))

    td_toks = [toker.encode(doc["positive"]) for doc in tqdm(train_dataset)]

    fiqa_toks = [toker.encode(doc["text"]) for doc in tqdm(normal_corpus.select(range(10000)))]
    nq_toks = [toker.encode(doc["text"]) for doc in tqdm(nq_corpus.select(range(10000)))]
    # show histogram of token length of train, fiqa, nq, save to png
    plt.hist([len(toks) for toks in td_toks], bins=100, label="train", color="blue", alpha=0.5)
    plt.hist([len(toks) for toks in fiqa_toks], bins=100, label="fiqa", color="green", alpha=0.5)
    plt.hist([len(toks) for toks in nq_toks], bins=100, label="nq", color="red", alpha=0.5)
    plt.legend()
    plt.savefig("token_length_hist.png")
    plt.close()

    #     mean([len(toks) for toks in td_toks])
    breakpoint()
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