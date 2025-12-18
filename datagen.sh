#!/bin/bash
#SBATCH --job-name=generate3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8               # Request 8 GPUs on the node
#SBATCH --mem=400G                 # Adjust as needed
#SBATCH --time=72:00:00
#SBATCH --output=propercache/logs/generate_bigdata-%j.out

source /system/linux/miniforge-3.12/etc/profile.d/conda.sh

# conda info --envs
conda activate scaling2
export OMP_NUM_THREADS=128

python scripts/generate_setpositives.py --startindex 0 --endindex 1500
python scripts/generate_setpositives.py --startindex 1500 --endindex 3000
python scripts/generate_setpositives.py --startindex 3000 --endindex 4500
python scripts/generate_setpositives.py --startindex 4500 --endindex 6000
python scripts/generate_setpositives.py --startindex 6000 --endindex 7500
python scripts/generate_setpositives.py --startindex 7500 --endindex 9000
python scripts/generate_setpositives.py --startindex 9000 --endindex 10500