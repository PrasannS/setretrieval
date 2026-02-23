#!/bin/bash
#SBATCH --job-name=modernbert-pylate
#SBATCH --output=%j-pltrain.log
#SBATCH --error=%j-pltrain.err
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=800GB
#SBATCH --nodes=1
source /system/linux/miniforge-3.12/etc/profile.d/conda.sh
conda activate scaling7

torchrun --nproc_per_node=4 scripts/train_pylate_pairwise.py --qvecs 1 --dvecs 100 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 16 --topk 2 --alpha 1
torchrun --nproc_per_node=4 scripts/train_pylate_pairwise.py --qvecs 1 --dvecs 100 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 16 --topk 2 --alpha 2
torchrun --nproc_per_node=4 scripts/train_pylate_pairwise.py --qvecs 1 --dvecs 100 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 16 --topk 4 --alpha 1
torchrun --nproc_per_node=4 scripts/train_pylate_pairwise.py --qvecs 1 --dvecs 100 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 16 --topk 8 --alpha 1

