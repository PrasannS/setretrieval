#!/bin/bash
#SBATCH --job-name=modernbert-pylate
#SBATCH --output=%j-eval.log
#SBATCH --error=%j-eval.err
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=600GB
source /system/linux/miniforge-3.12/etc/profile.d/conda.sh
conda activate scaling7

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
    # python scripts/wikipedia_eval.py --index_type $INDEX_TYPE --model_name $MODEL --dataset_path propercache/data/datastores/nqcorpus_bm25 --eval_set_path propercache/data/evalsets/nq_testset --k 10 --save_preds $SAVE_PREDS --colbert_qvecs $QVECS --colbert_dvecs $DVECS --forceredo $FORCE --colbert_passiveqvecs $PASSIVEQVECS --colbert_passivedvecs $PASSIVEDVECS
}

# eval models from qvscalingtrains.sh

# 4 100 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv4-dv100-pqv0-pdv0-embsize128/final" 4 100 128 "no" "paircolbnormalq4d100embsize128" "colbert" 0 0
# 16 100 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv16-dv100-pqv0-pdv0-embsize128/final" 16 100 128 "yes" "paircolbnormalq16d100embsize128" "colbert" 0 0
# 64 100 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv64-dv100-pqv0-pdv0-embsize128/final" 64 100 128 "yes" "paircolbnormalq64d100embsize128" "colbert" 0 0
# 16 200 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv16-dv200-pqv0-pdv0-embsize128/final" 16 200 128 "yes" "paircolbnormalq16d200embsize128" "colbert" 0 0
# 64 200 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv64-dv200-pqv0-pdv0-embsize128/final" 64 200 128 "yes" "paircolbnormalq64d200embsize128" "colbert" 0 0
# 16 400 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv16-dv400-pqv0-pdv0-embsize128/final" 16 400 128 "yes" "paircolbnormalq16d400embsize128" "colbert" 0 0
# 64 400 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv64-dv400-pqv0-pdv0-embsize128/final" 64 400 128 "yes" "paircolbnormalq64d400embsize128" "colbert" 0 0
# 128 400 128
nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv128-dv400-pqv0-pdv0-embsize128/final" 128 400 128 "yes" "paircolbnormalq128d400embsize128" "colbert" 0 0

