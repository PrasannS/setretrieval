from tqdm import tqdm

# filtering function to get rid of weird chunks in project gutenberg data
def filter_guten_docdata(row, col='text'):
    if 'eBook' in row[col] or row[col].count("|") > 5 or "                       " in row[col]: 
        return False
    if " the " not in row[col] or "The " not in row[col] or "Gutenberg" in row[col] or "electronic" in row[col]: 
        return False
    if row[col].count("--") > 3:
        return False
    # if 'id' in row: # HACK used to have > 800 filter
    #     return False
    return True

# function to get indices of dataset which meet some criteria
def get_dsinds(ds, seqs, query_col='query'):
    inds = []
    for i, row in tqdm(enumerate(ds), total=len(ds)):
        if row[query_col] in seqs:
            inds.append(i)
    return inds

def qtosearchlists(index, qs, docs, indid, k=2000):
    indidmap = {'wikipedia': 'propercache_data_datastores_wikipedia_8chunk_150k', 'gutenberg': 'propercache_data_datastores_gutenberg_chunks_800k'}
    indid = indidmap[indid]
    searches = index.search(qs, indid, k=k)
    searchlists = {}
    for i, q in enumerate(qs):
        searchlists[q] = [docs[row['index']] for row in searches[i]]
    return searchlists


def checktoverlap(idlist, searchlist):
    for k in idlist:
        print(k)
        assert k in searchlist
        idtset = set([r['text'] for r in idlist[k]])
        searchtset = set([r['text'] for r in searchlist[k]])
        print("both ", len(idtset & searchtset), "same id ", len(idtset), " search ", len(searchtset))
