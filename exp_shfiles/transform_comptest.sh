#!/bin/bash
#SBATCH --job-name=modernbert-pylate
#SBATCH --output=%j-transform_comptest.log
#SBATCH --error=%j-transform_comptest.err
#SBATCH --time=24:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-task=128
#SBATCH --mem=200GB

source /system/linux/miniforge-3.12/etc/profile.d/conda.sh
conda activate scaling7

# ── Corpus / eval set paths ──────────────────────────────────────────────────
ORIG_CORPUS=propercache/data/datastores/fiqacorpus
ORIG_EVAL=propercache/data/evalsets/fiqa_testset

SHORT_FULL=propercache/data/datastores/fiqacorpus_short
LONG_FULL=propercache/data/datastores/fiqacorpus_long
LONG_FULL_V2=propercache/data/datastores/fiqacorpus_longv2
SHORT_POSONLY=propercache/data/datastores/fiqacorpus_short_posonly
SHORT_NEGONLY=propercache/data/datastores/fiqacorpus_short_negonly
LONG_POSONLY=propercache/data/datastores/fiqacorpus_long_posonly
LONG_POSONLY_V2=propercache/data/datastores/fiqacorpus_longv2_posonly
LONG_NEGONLY=propercache/data/datastores/fiqacorpus_long_negonly
LONG_NEGONLY_V2=propercache/data/datastores/fiqacorpus_longv2_negonly
SHORT_EVAL=propercache/data/evalsets/fiqa_testset_short
LONG_EVAL=propercache/data/evalsets/fiqa_testset_long
LONG_EVAL_V2=propercache/data/evalsets/fiqa_testset_longv2
# ── Helper: run one eval ─────────────────────────────────────────────────────
run_eval() {
    local MODEL=$1
    local QVECS=$2
    local DVECS=$3
    local INDEX_TYPE=$4
    local CORPUS=$5
    local EVAL_SET=$6
    local SAVE_PREDS=$7
    local EMBSIZE=$8
    python -u scripts/wikipedia_eval.py \
        --index_type $INDEX_TYPE \
        --model_name $MODEL \
        --dataset_path $CORPUS \
        --eval_set_path $EVAL_SET \
        --k 10 \
        --save_preds $SAVE_PREDS \
        --colbert_qvecs $QVECS \
        --colbert_dvecs $DVECS \
        --forceredo "no" \
        --colbert_passiveqvecs 0 \
        --colbert_passivedvecs 0 \
        --colbert_ebsize $EMBSIZE
}

# ── Run all transform comparisons for a given model ──────────────────────────
# Combos:
#   corpus:   orig, short_full, long_full, short_posonly, short_negonly, long_posonly, long_negonly
#   evalset:  orig, short, long
# Meaningful pairs:
#   1. orig corpus    + orig eval      (baseline)
#   2. short_full     + short eval     (everything shortened)
#   3. long_full      + long eval      (everything lengthened)
#   4. short_posonly  + short eval     (only positives shortened in both)
#   5. short_negonly  + orig eval      (only negatives shortened, positives unchanged)
#   6. long_posonly   + long eval      (only positives lengthened in both)
#   7. long_negonly   + orig eval      (only negatives lengthened, positives unchanged)
#   8. long_full_v2   + long_eval_v2  (everything lengthened by 2x)
#   9. long_negonly_v2 + orig eval      (only negatives lengthened by 2x, positives unchanged)
#   10. long_posonly_v2 + long_eval_v2  (only positives lengthened by 2x in both)
run_transform_suite() {
    local MODEL=$1
    local QVECS=$2
    local DVECS=$3
    local INDEX_TYPE=$4
    local TAG=$5  # short identifier for save_preds naming
    local EMBSIZE=$6
    echo "=========================================="
    echo "Running transform suite: $TAG (q${QVECS}d${DVECS})"
    echo "=========================================="

    # run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $ORIG_CORPUS     $ORIG_EVAL  "${TAG}_orig" $EMBSIZE
    # run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $SHORT_FULL      $SHORT_EVAL "${TAG}_short_full" $EMBSIZE
    # run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $SHORT_POSONLY   $SHORT_EVAL "${TAG}_short_posonly" $EMBSIZE
    # run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $SHORT_NEGONLY   $ORIG_EVAL  "${TAG}_short_negonly" $EMBSIZE
    # run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $LONG_POSONLY    $LONG_EVAL  "${TAG}_long_posonly" $EMBSIZE
    # initially omitted
    # run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $LONG_FULL       $LONG_EVAL  "${TAG}_long_full" $EMBSIZE
    # run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $LONG_NEGONLY    $ORIG_EVAL  "${TAG}_long_negonly" $EMBSIZE
    # longv2 stuff
    run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $LONG_FULL_V2       $LONG_EVAL_V2  "${TAG}_long_full_v2" $EMBSIZE
    run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $LONG_NEGONLY_V2    $ORIG_EVAL  "${TAG}_long_negonly_v2" $EMBSIZE
    run_eval $MODEL $QVECS $DVECS $INDEX_TYPE $LONG_POSONLY_V2    $LONG_EVAL_V2  "${TAG}_long_posonly_v2" $EMBSIZE
}

# ── Models to test ───────────────────────────────────────────────────────────
Q1D400="output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv400-pqv0-pdv0-embsize128/final"
Q1D100="output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final"

# run_transform_suite "bm25" -1 -1 bm25 "bm25" 16
# q 1 d 1
run_transform_suite "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv1-pqv0-pdv0-embsize128/final" 1 1 colbert "paircolbnormalq1d1embsize128" 16
# 1 100
# run_transform_suite "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final" 1 100 colbert "paircolbnormalq1d100embsize128" 32
# 32 100
run_transform_suite "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv32-dv100-embsize128/final" 32 100 colbert "paircolbnormalq32d100embsize128" 32
# 1 400
# run_transform_suite $Q1D400 1 400 colbert "q1d400" 32
# run_transform_suite $Q1D400 1 400 colbert "q1d400"
# run_transform_suite $Q1D100 1 100 colbert "q1d100"
# bm25
