#!/bin/bash
#SBATCH --job-name=evals
#SBATCH --time=72:00:00
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-abstract_relevant_train_30k_newfilt-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-abstract_relevant_train_30k_newfilt-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-abstract_relevant_train_30k_newfilt-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-abstract_relevant_train_30k_newfilt-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100

source /system/linux/miniforge-3.12/etc/profile.d/conda.sh

# conda info --envs
conda activate scaling2

python scripts/wikipedia_eval.py --index_type single --model_name "intfloat/multilingual-e5-large-instruct" --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

python scripts/wikipedia_eval.py --index_type single --model_name "intfloat/multilingual-e5-large-instruct" --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

python scripts/wikipedia_eval.py --index_type single --model_name "intfloat/multilingual-e5-large-instruct" --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

python scripts/wikipedia_eval.py --index_type single --model_name "intfloat/multilingual-e5-large-instruct" --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100
