#!/bin/bash
#SBATCH --job-name=modernbert-pylate
#SBATCH --output=%j-topkeval.log
#SBATCH --error=%j-topkeval.err
#SBATCH --time=12:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=400GB
#SBATCH --nodes=1

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

# nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv1-dv100-pqv0-pdv0-embsize128-topk2-alpha1.0/final" 1 100 128 "yes" "paircolbnormalq1d100embsize128topk2alpha1.0" "colbert_faiss" 0 0

# nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv1-dv100-pqv0-pdv0-embsize128-topk2-alpha1.0/final" 1 100 128 "yes" "paircolbnormalq1d100embsize128topk2alpha2.0" "colbert_faiss" 0 0

# nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv1-dv100-pqv0-pdv0-embsize128-topk4-alpha1.0/final" 1 100 128 "yes" "paircolbnormalq1d100embsize128topk4alpha1.0" "colbert_faiss" 0 0

nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv1-dv100-pqv0-pdv0-embsize128-topk8-alpha1.0/final" 1 100 128 "yes" "paircolbnormalq1d100embsize128topk8alpha1.0" "colbert_faiss" 0 0


