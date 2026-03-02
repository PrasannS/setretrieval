

# --- original corpus ---
# python scripts/wikipedia_eval.py --index_type "colbert_faiss" --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final --dataset_path propercache/data/datastores/nqcorpus_bm25_t5 --eval_set_path propercache/data/evalsets/nq_testset_t5 --k 10 --save_preds "paircolbq1d100_orig" --forceredo "no"

python scripts/wikipedia_eval.py --index_type "colbert_faiss" --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv400-pqv0-pdv0-embsize128/final --dataset_path propercache/data/datastores/nqcorpus_bm25_t5 --eval_set_path propercache/data/evalsets/nq_testset_t5 --k 10 --save_preds "paircolbq1d400_orig" --forceredo "yes"

# --- summarized corpus ---
# python scripts/wikipedia_eval.py --index_type "colbert_faiss" --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final --dataset_path propercache/data/datastores/nqcorpus_bm25_t5_summarized --eval_set_path propercache/data/evalsets/nq_testset_t5_summarized --k 10 --save_preds "paircolbq1d100_summ" --forceredo "no"

# python scripts/wikipedia_eval.py --index_type "colbert_faiss" --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv400-pqv0-pdv0-embsize128/final --dataset_path propercache/data/datastores/nqcorpus_bm25_t5_summarized --eval_set_path propercache/data/evalsets/nq_testset_t5_summarized --k 10 --save_preds "paircolbq1d400_summ" --forceredo "no"
