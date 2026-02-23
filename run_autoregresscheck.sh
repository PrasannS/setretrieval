#!/bin/bash
# Compare ModernBERT base vs large on parallel vs autoregressive prediction (task1)
# N values: 4, 16, 64

set -e

SCRIPT="scripts/autoreg_lim.py"
N_VALUES="4"
EPOCHS=1
TRAIN_SAMPLES=20000
EVAL_SAMPLES=500
BATCH_SIZE=16

# ModernBERT base
echo "=== ModernBERT Base ==="
# python $SCRIPT \
#   --task task1 \
#   --pretrained facebook/opt-350m \
#   --n_values $N_VALUES \
#   --modes both \
#   --epochs $EPOCHS \
#   --train_samples $TRAIN_SAMPLES \
#   --eval_samples $EVAL_SAMPLES \
#   --batch_size $BATCH_SIZE \
#   --output_dir ./results/modernbert_base

# ModernBERT large
echo "=== ModernBERT Large ==="
python $SCRIPT \
  --task task2 \
  --pretrained facebook/opt-125m \
  --n_values $N_VALUES \
  --modes both \
  --epochs $EPOCHS \
  --train_samples $TRAIN_SAMPLES \
  --eval_samples $EVAL_SAMPLES \
  --batch_size $BATCH_SIZE \
  --output_dir ./results/opt-1.3b \
  --lr 2e-5

echo "=== Done. Results in ./results/opt-350m and ./results/opt-1.3b ==="
