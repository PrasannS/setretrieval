#!/bin/bash
#SBATCH --job-name=modernbert-pylate
#SBATCH --output=%j-pltrain.log
#SBATCH --error=%j-pltrain.err
#SBATCH --time=24:00:00
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=800GB
#SBATCH --nodes=1

source /system/linux/miniforge-3.12/etc/profile.d/conda.sh
conda activate scaling7

# 4 100 128
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py --qvecs 4 --dvecs 100 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 32

# 16 100 128
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py --qvecs 16 --dvecs 100 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 32

# 64 100 128
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py --qvecs 64 --dvecs 100 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 32

# 16 200 128
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py --qvecs 16 --dvecs 200 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 32

# 64 200 128
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py --qvecs 64 --dvecs 200 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 32

# 16 400 128
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py --qvecs 16 --dvecs 400 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 32

# 64 400 128
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py --qvecs 64 --dvecs 400 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 32

# 128 400 128
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py --qvecs 64 --dvecs 400 --passiveqvecs 0 --passivedvecs 0 --embdim 128 --lr 3e-4 --mini_batch_size 16
