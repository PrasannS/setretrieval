#!/bin/bash
#SBATCH --job-name=generate2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8               # Request 8 GPUs on the node
#SBATCH --mem=400G                 # Adjust as needed
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --output=logs/generate_bigdata-%j.out

source /system/linux/miniforge-3.12/etc/profile.d/conda.sh

# conda info --envs
conda activate scaling2
export OMP_NUM_THREADS=128

python scripts/generate_setpositives.py