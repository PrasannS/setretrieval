# run detailed predictions for 1 1, 1 100, 32 100 on fiqa, msmarco. On 128 emb size.
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
    python scripts/wikipedia_eval.py --index_type $INDEX_TYPE --model_name $MODEL --dataset_path propercache/data/datastores/nanomsmarco_corpus --eval_set_path propercache/data/evalsets/nanomsmarco_evalset --k 1 --save_preds $SAVE_PREDS --colbert_qvecs $QVECS --colbert_dvecs $DVECS --forceredo $FORCE --colbert_passiveqvecs $PASSIVEQVECS --colbert_passivedvecs $PASSIVEDVECS --detailed_save $SAVE_PREDS
    # fiqa
    python scripts/wikipedia_eval.py --index_type $INDEX_TYPE --model_name $MODEL --dataset_path propercache/data/datastores/fiqacorpus --eval_set_path propercache/data/evalsets/fiqa_testset --k 1 --save_preds $SAVE_PREDS --colbert_qvecs $QVECS --colbert_dvecs $DVECS --forceredo $FORCE --colbert_passiveqvecs $PASSIVEQVECS --colbert_passivedvecs $PASSIVEDVECS --detailed_save $SAVE_PREDS
}

# 1 1
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv1-pqv0-pdv0-embsize128/final" 1 1 128 "yes" "paircolbnormalq1d1embsize128" "colbert_faiss" 0 0
# 1 100
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final" 1 100 128 "yes" "paircolbnormalq1d100embsize128" "colbert_faiss" 0 0
# 32 100
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv32-dv100-embsize128/final" 32 100 128 "yes" "paircolbnormalq32d100embsize128" "colbert_faiss" 0 0
# 1 400
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv400-embsize128/final" 1 400 128 "yes" "paircolbnormalq1d400embsize128" "colbert_faiss" 0 0
# ssh -t -t prasann@lambda.stat.berkeley.edu -L 8888:localhost:8888 ssh lambda-hyperplane02 -L 8888:localhost:8888
