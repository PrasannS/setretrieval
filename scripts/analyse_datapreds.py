# import pickload
from setretrieval.utils.utils import pickload
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mean


data32 = pickload("propercache/cache/detailed_preds/propercache_data_datastores_fiqacorpus_paircolbnormalq32d100embsize128.pkl")

def nd_2_docids_posids(ndata):
    docids, posids = [], []
    for nd in ndata:
        tmp_docids, tmp_posids = [], []
        for n in nd[1]:
            tok_docids, tok_posids = [], []
            for tok in n:
                tok_docids.append(int(tok/100))
                tok_posids.append(int(tok%100))
            tmp_docids.append(tok_docids)
            tmp_posids.append(tok_posids)
        docids.append(tmp_docids)
        posids.append(tmp_posids)
    return docids, posids

docids32, posids32 = nd_2_docids_posids(data32)

unc_dict = {k: [] for k in range(5000)}
for d in docids32:
    cset = set()
    for i in range(5000):
        cset.add(d[0][i])
        unc_dict[i].append(len(cset))

# fig, ax = plt.subplots()
plt.scatter(list(range(5000)), [mean(unc_dict[i]) for i in range(5000)])
plt.xlabel("top k")
plt.ylabel("total count of unique documents in this top k")
# save to file
plt.savefig("unc_dict_32.png")
breakpoint()

pflats = []
for p in posids32:
    pflats.extend(p[0])

plt.hist(pflats, bins=100)
plt.xlabel("position")
plt.ylabel("count")
# save to file
plt.savefig("pflats_32.png")
breakpoint()