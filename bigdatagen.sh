#!/bin/bash
#SBATCH --job-name=generate_setpositivestk
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


source /system/linux/miniforge-3.12/etc/profile.d/conda.sh

# conda info --envs
conda activate scaling2
export OMP_NUM_THREADS=128

# python scripts/generate_setpositives.py --startindex 0 --endindex 500 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --starter_question_set propercache/data/colbert_training/gemini_ntrain_ptest_3

# python scripts/generate_setpositives.py --startindex 0 --endindex 500 --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --starter_question_set propercache/data/colbert_training/gemini_gutfilteredtrainqs


python scripts/generate_setpositives.py --startindex 0 --endindex 2 --dataset_path propercache/data/datastores/sanitychecks/wikirands100k --starter_question_set propercache/data/colbert_training/wikigemini_tinybigtest/
python scripts/generate_setpositives.py --startindex 0 --endindex 2 --dataset_path propercache/data/datastores/sanitychecks/gutterands100k --starter_question_set propercache/data/colbert_training/gutengemini_tinybigtest/

# python scripts/generate_setpositives.py --startindex 1500 --endindex 3000
# python scripts/generate_setpositives.py --startindex 3000 --endindex 4500
# python scripts/generate_setpositives.py --startindex 4500 --endindex 6000
# python scripts/generate_setpositives.py --startindex 6000 --endindex 7500
# python scripts/generate_setpositives.py --startindex 7500 --endindex 9000
# python scripts/generate_setpositives.py --startindex 9000 --endindex 10500

# python scripts/generate_starterquestions.py --startindex 0 --endindex 300 --model gemini-2.5-flash-lite --dataset_path propercache/data/datastores/wikipedia_docs_1.5M --domain wikipedia