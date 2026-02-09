from datasets import load_dataset

def get_doc_and_ids(doc_pairs):
    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(str(dp['id']))
        documents.append(dp['content'])
    return documents, doc_ids
    
def process_pos_id2doc(entry, id2doc):
    pos_docs = entry["pos"]
    res = []
    for pos in pos_docs:
        instruction, doc_id = pos[0], pos[1]
        doc = id2doc[doc_id]
        res.append([instruction, doc])
    entry["pos"] = res
    return entry


if __name__ == "__main__":

    hq_dataset = load_dataset("reasonir/reasonir-data", "hq")
    bright_docs = load_dataset("xlangai/BRIGHT", "documents")
    all_docs = []   
    all_ids = []
    for task in bright_docs.keys():
        docs, ids = get_doc_and_ids(bright_docs[task])
        all_docs.extend(docs)
        all_ids.extend(ids)

    id2doc = {}
    for i in range(len(all_docs)):
        id2doc[all_ids[i]] = all_docs[i]

    hq_dataset = hq_dataset.map(lambda x: process_pos_id2doc(x, id2doc))

    breakpoint()

    hq_dataset.save_to_disk("propercache/data/colbert_training/reasonir_hq")