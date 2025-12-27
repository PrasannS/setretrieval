
# source /system/linux/miniforge-3.12/etc/profile.d/conda.sh

# # conda info --envs
# conda activate scaling2
# export OMP_NUM_THREADS=128

# python scripts/generate_setpositives.py --startindex 0 --endindex 1500
# python scripts/generate_setpositives.py --startindex 1500 --endindex 3000
# python scripts/generate_setpositives.py --startindex 3000 --endindex 4500
# python scripts/generate_setpositives.py --startindex 4500 --endindex 6000
# python scripts/generate_setpositives.py --startindex 6000 --endindex 7500
# python scripts/generate_setpositives.py --startindex 7500 --endindex 9000
# python scripts/generate_setpositives.py --startindex 9000 --endindex 10500

# python scripts/generate_starterquestions.py --startindex 0 --endindex 100000 --model gemini-2.5-flash-lite --dataset_path propercache/data/datastores/wikipedia_docs_1.5M --domain wikipedia