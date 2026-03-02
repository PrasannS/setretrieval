# processing of NQ (will use for information expansion experiment as well, so let's try to make this small if possible)

# python scripts/make_nq_evalset.py \
#     --output_dir propercache/data \
#     --top_k 5 \
#     --split test \
#     --corpus_name nqcorpus_bm25_t5 \
#     --evalset_name nq_testset_t5


# python scripts/make_nq_evalset.py \
#     --output_dir propercache/data \
#     --top_k 5 \
#     --split test \
#     --corpus_name nqcorpus_bm25_t5 \
#     --evalset_name nq_testset_t5

python scripts/nq_fullchunk_setup.py --origchunks propercache/data/datastores/nqcorpus_bm25_t5
