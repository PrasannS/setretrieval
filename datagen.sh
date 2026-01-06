#!/bin/bash
#SBATCH --job-name=absdatagen
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

# FINAL COMMANDS

# python scripts/generate_setpositives.py --startindex 0 --endindex 125 --dataset_path propercache/data/datastores/absdata_domain1 --starter_question_set propercache/data/colbert_training/absdata_domain1_datagenset --comparison yes --modelcnt 3

# python scripts/generate_setpositives.py --startindex 0 --endindex 125 --dataset_path propercache/data/datastores/absdata_domain3 --starter_question_set propercache/data/colbert_training/absdata_domain3_datagenset --comparison yes --modelcnt 3

# python scripts/generate_setpositives.py --startindex 0 --endindex 500 --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --starter_question_set propercache/data/colbert_training/gemini_gutfilteredtrainqs --comparison no --modelcnt 3 --temp_minlimit 300 --max_s1_pps 1000


# python scripts/generate_setpositives.py --startindex 0 --endindex 500 --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --starter_question_set propercache/data/colbert_training/gemini_gutfilteredtrainqs --comparison no --modelcnt 3 --temp_minlimit 500 --max_s1_pps 1000

# python scripts/generate_setpositives.py --startindex 0 --endindex 500 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --starter_question_set propercache/data/colbert_training/gemini_ntrain_ptest_3 --comparison no --modelcnt 3 --temp_minlimit 500 --max_s1_pps 1000


# OLD


# python scripts/generate_setpositives.py --startindex 0 --endindex 500 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --starter_question_set propercache/data/colbert_training/gemini_ntrain_ptest_3

# python scripts/generate_setpositives.py --startindex 0 --endindex 500 --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --starter_question_set propercache/data/colbert_training/gemini_gutfilteredtrainqs


# python scripts/generate_setpositives.py --startindex 0 --endindex 125 --dataset_path propercache/data/datastores/absdata_domain2 --starter_question_set propercache/data/colbert_training/absdata_domain2_datagenset --comparison yes
# python scripts/generate_setpositives.py --startindex 0 --endindex 125 --dataset_path propercache/data/datastores/absdata_domain3 --starter_question_set propercache/data/colbert_training/absdata_domain3_datagenset --comparison yes
# python scripts/generate_setpositives.py --startindex 0 --endindex 125 --dataset_path propercache/data/datastores/absdata_domain6 --starter_question_set propercache/data/colbert_training/absdata_domain6_datagenset --comparison yes

python scripts/generate_setpositives.py --startindex 0 --endindex 2 --dataset_path propercache/data/datastores/sanitychecks/wikitinytop5k --starter_question_set propercache/data/colbert_training/wikigemini_tinybigtest/ --comparison no --modelcnt 3
python scripts/generate_setpositives.py --startindex 0 --endindex 2 --dataset_path propercache/data/datastores/sanitychecks/guttinytop5k --starter_question_set propercache/data/colbert_training/gutengemini_tinybigtest/ --comparison no --modelcnt 3

# python scripts/generate_setpositives.py --startindex 1500 --endindex 3000
# python scripts/generate_setpositives.py --startindex 3000 --endindex 4500
# python scripts/generate_setpositives.py --startindex 4500 --endindex 6000
# python scripts/generate_setpositives.py --startindex 6000 --endindex 7500
# python scripts/generate_setpositives.py --startindex 7500 --endindex 9000
# python scripts/generate_setpositives.py --startindex 9000 --endindex 10500

# python scripts/generate_starterquestions.py --startindex 0 --endindex 300 --model gemini-2.5-flash-lite --dataset_path propercache/data/datastores/wikipedia_docs_1.5M --domain wikipedia