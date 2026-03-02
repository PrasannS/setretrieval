#!/bin/bash
#SBATCH --job-name=modernbert-pylate
#SBATCH --output=%j-dynatrain.log
#SBATCH --error=%j-dynatrain.err
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=400GB
#SBATCH --nodes=1

source /system/linux/miniforge-3.12/etc/profile.d/conda.sh
conda activate scaling7
# dynamic train, padding tokens are consistent w.r.t. document or query length
# torchrun --nproc_per_node=4 scripts/train_pylate_pairwise.py --qvecs 1 --dratio 0.6 --embdim 128 --lr 3e-4 --mini_batch_size 32
# torchrun --nproc_per_node=4 scripts/train_pylate_pairwise.py --qvecs 1 --dratio 2.4 --embdim 128 --lr 3e-4 --mini_batch_size 32



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

# nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv1-dv-1-pqv0-pdv0-embsize128-qratio0-dratio0.6/final" -1 -1 128 "yes" "paircolbdratio06" "colbert_faiss" 0 0

nanofiqaeval "output/ModernBERT-base/ModernBERT-base-pylate-pairwise-msmarco-0.0003-qv1-dv-1-pqv0-pdv0-embsize128-qratio0-dratio2.4/final" -1 -1 128 "yes" "paircolbdratio24" "colbert_faiss" 0 0
