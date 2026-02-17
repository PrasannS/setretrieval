
nanofiqaeval() {
    local MODEL=$1
    local QVECS=$2
    local DVECS=$3
    local EMBSIZE=$4
    local FORCE=$5
    local SAVE_PREDS=$6
    local INDEX_TYPE=$7
    local PASSIVEQVECS=$8
    local PASSIVEDVECS=$9
    # nanomsmarco
    python scripts/wikipedia_eval.py --index_type $INDEX_TYPE --model_name $MODEL --dataset_path propercache/data/datastores/nanomsmarco_corpus --eval_set_path propercache/data/evalsets/nanomsmarco_evalset --k 10 --save_preds $SAVE_PREDS --colbert_qvecs $QVECS --colbert_dvecs $DVECS --forceredo $FORCE --colbert_passiveqvecs $PASSIVEQVECS --colbert_passivedvecs $PASSIVEDVECS
    # fiqa
    python scripts/wikipedia_eval.py --index_type $INDEX_TYPE --model_name $MODEL --dataset_path propercache/data/datastores/fiqacorpus --eval_set_path propercache/data/evalsets/fiqa_testset --k 10 --save_preds $SAVE_PREDS --colbert_qvecs $QVECS --colbert_dvecs $DVECS --forceredo $FORCE --colbert_passiveqvecs $PASSIVEQVECS --colbert_passivedvecs $PASSIVEDVECS
    # nq
    python scripts/wikipedia_eval.py --index_type $INDEX_TYPE --model_name $MODEL --dataset_path propercache/data/datastores/nqcorpus_bm25 --eval_set_path propercache/data/evalsets/nq_testset --k 10 --save_preds $SAVE_PREDS --colbert_qvecs $QVECS --colbert_dvecs $DVECS --forceredo $FORCE --colbert_passiveqvecs $PASSIVEQVECS --colbert_passivedvecs $PASSIVEDVECS
}

# # bm25
# nanofiqaeval "bm25" -1 -1 0 "partial" "bm25" "bm25" 0 0
# 1 1 768
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv1-pqv0-pdv0-embsize768/final" 1 1 768 "partial" "paircolbnormalq1d1embsize768" "colbert" 0 0
# 1 10 768
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv10-embsize768/final" 1 10 768 "partial" "paircolbnormalq1d10embsize768" "colbert" 0 0
# 1 100 768
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize768/final" 1 100 768 "partial" "paircolbnormalq1d100embsize768" "colbert_faiss" 0 0
# 1 1 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv1-pqv0-pdv0-embsize128/final" 1 1 128 "partial" "paircolbnormalq1d1embsize128" "colbert_faiss" 0 0
# 1 6 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv6-embsize128/final" 1 6 128 "partial" "paircolbnormalq1d6embsize128" "colbert_faiss" 0 0
# 1 100 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final" 1 100 128 "partial" "paircolbnormalq1d100embsize128" "colbert_faiss" 0 0
# 32 100 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv32-dv100-embsize128/final" 32 100 128 "partial" "paircolbnormalq32d100embsize128" "colbert_faiss" 0 0
# large model, 1 1 1024
nanofiqaeval "output/ModernBERT-large/ModernBERT-large-pylate-pairwise-msmarco-0.0003-qv1-dv1-pqv0-pdv0-embsize1024/final" 1 1 1024 "partial" "modernbertlargecolbnormalq1d1embsize1024" "colbert_faiss" 0 0
# large model, 1 10 1024
nanofiqaeval "output/ModernBERT-large/ModernBERT-large-pylate-pairwise-msmarco-0.0003-qv1-dv10-pqv0-pdv0-embsize1024/final" 1 10 1024 "partial" "modernbertlargecolbnormalq1d10embsize1024" "colbert_faiss" 0 0
# large model, 1 100 1024
nanofiqaeval "output/ModernBERT-large/ModernBERT-large-pylate-pairwise-msmarco-0.0003-qv1-dv100-pqv0-pdv0-embsize1024/final" 1 100 1024 "partial" "modernbertlargecolbnormalq1d100embsize1024" "colbert_faiss" 0 0





