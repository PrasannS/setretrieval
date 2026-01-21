# notebook for processing gutenberg, wikipedia, fineweb large scale chunk data
from datasets import Dataset, load_dataset
from setretrieval.datagen.data_utils import filter_guten_docdata
from tqdm import tqdm

def process_gutenberg_chunks(intext, idval=None):
    divided = intext.split("\n\n")
    results = []
    for d in divided:
        if len(d) < 100:
            continue
        if len(d) > 2000:
            # split by sentences
            restmp = d.replace("\n", " ").split('.')
            # join every 5 sentences
            restmp = [restmp[i:i+10] for i in range(0, len(restmp), 5)]
            results.extend([". ".join(r).strip()[:2000] for r in restmp])
        else:
            results.append(d.replace("\n", " ").strip())
    return [{'text': r, 'wc': len(r.split()), 'id': idval} for r in results]


def group_chunks_maxsize(clist, msize=200):
    # group chunks into group of at least msize
    newclist = []
    cur = {'text': '', 'id': '', 'wc': 0}
    for c in clist: 
        if cur['wc'] + c['wc'] <= msize: 
            cur['text'] += c['text']
            cur['wc'] += c['wc']
            cur['id'] = c['id']
        else: 
            newclist.append(cur)
            cur = c
    newclist.append(cur)
    return newclist

def row_chunkproc(row):
    row['chunks'] = process_gutenberg_chunks(row['text'], idval=row['id'])
    return row

def group_chunks_maxsize_row(row):
    row['chunks'] = group_chunks_maxsize(row['chunks'], msize=300)
    return row

if __name__ == "__main__":
    gutenbergdata = Dataset.load_from_disk("propercache/data/datastores/gutenberg_1k")
    fullgutenbergen = load_dataset("manu/project_gutenberg", split="en")

    # remove stuff which might be from our test set
    ignoreids = set(gutenbergdata["id"])
    useids = [x not in ignoreids for x in list(fullgutenbergen["id"])]
    seeset = set()
    
    # remove duplicates
    for i, x in enumerate(list(fullgutenbergen["id"])):
        if x in seeset: 
            useids[i] = False
        else:
            seeset.add(x)

    useinds = [i for i, x in enumerate(useids) if x]
    filtguten = fullgutenbergen.select(useinds)
    filtguten = filtguten.select(range(5000))
    
    # assert no duplicates
    assert len(filtguten) == len(set(filtguten["id"])), "Duplicates found"
    
    # now get rid of duplicates (based on id)
    
    breakpoint()

    filtguten = filtguten.map(row_chunkproc, num_proc=64)

    filtguten = filtguten.map(group_chunks_maxsize_row, num_proc=64)

    breakpoint()

    allchunks = []
    for i in tqdm(filtguten):
        allchunks.extend(i['chunks'])

    fullgutenmega = Dataset.from_list(allchunks)

    breakpoint()

    gfinds = []
    ind = 0
    for row in tqdm(fullgutenmega):
        if filter_guten_docdata(row):
            gfinds.append(ind)
        ind +=1

    fullgutenmega = fullgutenmega.select(gfinds)

    fullgutenmega.save_to_disk("propercache/data/datastores/gutenberg_5kbooks_megachunks")