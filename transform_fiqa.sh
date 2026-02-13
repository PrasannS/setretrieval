#!/bin/bash
#SBATCH --job-name=transform_fiqa
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /system/linux/miniforge-3.12/etc/profile.d/conda.sh
conda activate scaling2
export OMP_NUM_THREADS=128

CORPUS=propercache/data/datastores/fiqacorpus
MODEL=gemini-2.5-flash-lite

# Shorten
python -u scripts/transform_corpus.py \
    --corpus_path $CORPUS \
    --mode shorten \
    --output_path propercache/data/datastores/fiqacorpus_short \
    --model $MODEL

# Lengthen
python -u scripts/transform_corpus.py \
    --corpus_path $CORPUS \
    --mode lengthen \
    --output_path propercache/data/datastores/fiqacorpus_long \
    --model $MODEL
