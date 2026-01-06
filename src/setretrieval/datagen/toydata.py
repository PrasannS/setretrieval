from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from datasets import Dataset

# TASK 0: For each document in wikidocs, split up into sentences. 
# TODO can do more natural splitt
def chunk_ds(ds, nwords=10):
    alldata = []
    for doc in tqdm(ds):
        t = sent_tokenize(doc['text'])
        for sent in t:
            alldata.append({'text': sent})
    return Dataset.from_list(alldata)

# given a dataset of sentences, extract all n-grams in each sentence, add thi
def process_ngrams(row, n=2):
    toks = word_tokenize(row['text'])
    ngrams = [toks[i:i+n] for i in range(len(toks)-n+1)]
    row['pos_chunks'] = [" ".join(n).lower() for n in ngrams]
    return row