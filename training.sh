# torchrun --nproc_per_node=8 scripts/train.py > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 32 --num_epochs 3 --learning_rate 3e-6 > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# from here trying constant lr
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 2 --num_epochs 11 --learning_rate 3e-6 --model_name "google-bert/bert-base-uncased" --traintype "sbert" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "sbert" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "sbert" --dataset "gemini_ntrain_ptest" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --dataset "gemini_ntrain_ptest" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "sbert" --dataset "gemini_ntrain_ptest" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "sbert" --dataset "gemini_ntrain_ntest" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "sbert" --dataset "gemini_multisingleposmix_ptest" # > propercache/logs/train$(date +%Y%m%d_%H%M%S).log
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "gemini_multisingleposmix_ptest" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "sbert" --dataset "gemini_multipostrain_ptest" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "gemini_multipostrain_ptest" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "sbert" --dataset "nountraining10words" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "nountraining10words" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 


# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "sbert" --dataset "nountraining100words" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 


# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "nountraining100words" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "nountraining100words" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "gemini_abstracttrain" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "gemini_gutenbergtrain" #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "abstract_relevant_train_40k_filtered" --querylen 256 #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "propercache/models/bertbase" --traintype "colbert" --dataset "gutenberg_prox_train_40k_filtered" --querylen 256 #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 20 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "nountraining100words" --querylen 256  #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --dataset "gemini_ntrain_ptest" --querylen 64  #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 1 --num_epochs 5 --learning_rate 3e-6 --model_name "Qwen/Qwen3-Embedding-4B" --traintype "colbert" --dataset "gemini_ntrain_ptest" --querylen 32 --maxchars 100 #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "wiki_gemini_mini_train_10k" --querylen 32 #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 3e-6 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "wiki_gemini_mini_train_full" --querylen 32 #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 3e-6 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining1words100000" --querylen 32 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 3e-6 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining5words100000" --querylen 32 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 3e-6 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining10words100000" --querylen 32 

# torchrun --nproc_per_node=4 scripts/train.py --batch_size 16 --num_epochs 5 --learning_rate 3e-4 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining25words100000" --querylen 32 

# torchrun --nproc_per_node=4 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining25words100000ndps25" --querylen 32 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining25words100000ndps25" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 3e-6 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining25words1000000ndps25" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountestset25" --querylen 32

# torchrun --nproc_per_node=4 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining100words100000ndps1" --querylen 128

# torchrun --nproc_per_node=4 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining100words100000ndps100" --querylen 128

# train on skew sets
# torchrun --nproc_per_node=4 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining100kuniform5_100" --querylen 128

# torchrun --nproc_per_node=4 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountraining100kpower5_100" --querylen 128

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountrain100000rand10dwords" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "nountrain100000minimal10dwords" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 1 --num_epochs 1 --learning_rate 3e-5 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "sbert" --dataset "gemini_ntrain_ptest" --querylen 32  #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "colbert" --dataset "gemini_ntrain_ptest" --querylen 32  #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "gemini_ntrain_ptest" --querylen 32  #> propercache/logs/train$(date +%Y%m%d_%H%M%S).log 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "wiki_gemini_mini_train_10k" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "abstract_relevant_train_30k_newfilt" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "gemini_abstracttrain" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "gemini_gutenbergtrain" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "nountrain100000rand10dwords" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "nountrain100000minimal10dwords" --querylen 128

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "nountrain100000minimal50dwords" --querylen 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountrain100000rand10dwords" --querylen 32 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountrain100000rand50dwords" --querylen 32



# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountrain100000rand50dwords" --querylen 32 --div_coeff 1.0

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountrain100000rand50dwords" --querylen 32 --divq_coeff 1.0

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountrain100000rand50dwords" --querylen 32 --div_coeff 0.05

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountrain100000rand50dwords" --querylen 32 --div_coeff 0.0

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountraining50words100000ndps1" --querylen 50 --div_coeff 0.0

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountraining50words100000ndps1" --querylen 50 --divq_coeff 1.0

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountraining50words100000ndps1" --querylen 50 --divq_coeff 100.0

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountraining50words100000ndps1" --querylen 50 --divq_coeff 10.0 --colscore "maxmax"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountraining50words100000ndps1" --querylen 50 --divq_coeff 0.0 --temp 0.02

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountraining50words100000ndps1" --querylen 50 --divq_coeff 0.0 --temp 0.5

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "gemini_gutenbergtrain" --querylen 32 --divq_coeff 0.0

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "gemini_gutenbergtrain" --querylen 32 --divq_coeff 0.0 --div_coeff 10

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "gemini_gutenbergtrain" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 1

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --dataset "gemini_gutenbergtrain" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 0.05

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "NovaSearch/stella_en_1.5B_v5" --traintype "colbert" --dataset "gemini_gutenbergtrain" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 0.05

# torchrun --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2.1e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "wiki1gramtrain50000samplebalancedcontam" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 0.5
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "wiki2gramtrain50000samples" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 0.05

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "wiki2gramtrain50000samples" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 0.05


# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountraining50words100000ndps1" --querylen 50 --divq_coeff 0.0 --temp 0.5

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 1 --dataset "v2nountraining50words100000ndps1" --div_coeff 0 --temp 0.02 --doclen 2 --querylen 50

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "wiki2gramtrain50000samples" --querylen 1 --divq_coeff 0.0 --div_coeff 0 --temp 0.5


# torchrun --nproc_per_node=4 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "v2nountrain100000rand150dwords" --div_coeff 0 --temp 0.5 --doclen 512 --querylen 8 --dodefaulttrain "yes"

# train on 4 realistic sets (wiki, guten, bio, phys)
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 1 --dataset "abstract_relevant_train_30k_newfilt" --div_coeff 0 --temp 0.02 --doclen 511 --querylen 32 --dodefaulttrain "yes"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 1 --dataset "gemini_gutenbergtrain" --div_coeff 0 --temp 0.02 --doclen 511 --querylen 32 --dodefaulttrain "yes"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 1 --dataset "gemini_ntrain_ptest" --div_coeff 0 --temp 0.02 --doclen 511 --querylen 32 --dodefaulttrain "yes"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 1 --dataset "gemini_abstracttrain" --div_coeff 0 --temp 0.02 --doclen 511 --querylen 32 --dodefaulttrain "yes"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 0 --dataset "fineweb_gmini_300k" --div_coeff 0 --temp 1 --doclen 511 --querylen 48 --dodefaulttrain "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 1e-6 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "gutenberg_gmini_300k" --div_coeff 0 --temp 0.02 --doclen 511 --querylen 32 --dodefaulttrain "yes"


# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 8e-6 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 0 --dataset "gutenberg_gmini_30k" --div_coeff 0 --temp 0.02 --doclen 511 --querylen 32 --dodefaulttrain "yes"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 8e-6 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 0 --dataset "gutenberg_gmini_90k" --div_coeff 0 --temp 0.02 --doclen 511 --querylen 32 --dodefaulttrain "yes"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 8e-6 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 0 --dataset "gutenberg_gmini_30k_nosame" --div_coeff 0 --temp 0.02 --doclen 511 --querylen 32 --dodefaulttrain "yes"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 8e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "gutenberg_gmini_30k_nosame" --div_coeff 0 --temp 0.02 --doclen 1 --querylen 1 --dodefaulttrain "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 8e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "gutenberg_gmini_30k_nosame" --div_coeff 0 --temp 0.02 --doclen 100 --querylen 1 --dodefaulttrain "no"


# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "intfloat/multilingual-e5-large-instruct" --traintype "colbert" --divq_coeff 0 --dataset "gutenbergnoshuff_gmini_300k" --div_coeff 0 --temp 1 --doclen 511 --querylen 48 --dodefaulttrain "no"

# torchrun --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "train1wordsdebug100" --div_coeff 0 --temp 1 --doclen 5 --querylen 5 --dodefaulttrain "no" --compile "no" --save_strat "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "v2nountrain100000rand10dwords" --div_coeff 0 --temp 0.5 --doclen 10 --querylen 1 --dodefaulttrain "no" --compile "yes" --save_strat "no"
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "v2nountrain100000rand10dwords" --div_coeff 0 --temp 0.5 --doclen 10 --querylen 10 --dodefaulttrain "no" --compile "yes" --save_strat "no"
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "v2nountrain100000rand10dwords" --div_coeff 0 --temp 0.5 --doclen 20 --querylen 1 --dodefaulttrain "no" --compile "yes" --save_strat "no"

# torchrun --master_port 12345 --nproc_per_node=1 scripts/train.py --batch_size 4 --num_epochs 10 --learning_rate 2e-5 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "train1wordsdebug100" --div_coeff 0 --temp 1 --doclen 5 --querylen 5 --dodefaulttrain "no" --compile "no" --save_strat "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 5e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "v2nountrain100000rand10dwords" --div_coeff 0 --temp 0.02 --doclen 10 --querylen 32 --dodefaulttrain "no" --compile "yes" --save_strat "no"


# torchrun --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 10 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "train1wordsdebug100" --div_coeff 0 --temp 1 --doclen 5 --querylen 5 --dodefaulttrain "no" --compile "no" --save_strat "no"

# torchrun --master_port 12345 --nproc_per_node=1 scripts/train.py --batch_size 4 --num_epochs 3 --learning_rate 2e-5 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "train1wordsdebug100" --div_coeff 0 --temp 1 --doclen 5 --querylen 5 --dodefaulttrain "no" --compile "no" --save_strat "no" #--lora_rank 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 8e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "gutenberg_gmini_30k_nosame" --div_coeff 0 --temp 0.02 --doclen 1 --querylen 100 --dodefaulttrain "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 1 --dataset "v2nountraining50words100000ndps1" --div_coeff 0 --temp 0.02 --doclen 10 --querylen 100 --dodefaulttrain "yes" --embsize 128 --compile "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 8e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "v2nountrain100000rand50dwords" --div_coeff 1 --temp 0.02 --doclen 100 --querylen 8 --dodefaulttrain "yes" --embsize 128 --compile "no"

# torchrun --master_port 12345 --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 5e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "train50wordsdebug100setversion" --div_coeff 0 --temp 1 --doclen 4 --querylen 50 --dodefaulttrain "no" --compile "no" --save_strat "no" --colscore "multipos" #--lora_rank 32


# torchrun --master_port 12345 --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 5e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "10dworddebug100setversion" --div_coeff 1 --temp 0.02 --doclen 20 --querylen 1 --dodefaulttrain "no" --compile "no" --save_strat "no" --colscore "multiquery" #--lora_rank 32

# torchrun --master_port 12345 --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 5e-5 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "train50wordsdebug100setversion" --div_coeff 0 --temp 0.02 --doclen 1 --querylen 50 --dodefaulttrain "no" --compile "no" --save_strat "no" --colscore "multipos" #--lora_rank 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 5e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "mathtask1_expr2num_min0_max5000_md0_0_hardneg0.0_trainingsize100000" --div_coeff 0 --temp 0.02 --doclen 1 --querylen 1 --dodefaulttrain "no" --embsize 8 --compile "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 5e-6 --model_name "facebook/opt-1.3b" --traintype "sbert" --divq_coeff 0 --dataset "wikipedia_train_impossible_20k_processed" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no"

# train for impossible, possible sets
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 5e-6 --model_name "facebook/opt-1.3b" --traintype "sbert" --divq_coeff 0 --dataset "wikipedia_train_possible_20k_processed" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no"

# torchrun --master_port 12345 --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 5e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "10dworddebug100setversion" --div_coeff 0 --temp 0.02 --doclen 20 --querylen 1 --dodefaulttrain "no" --compile "no" --save_strat "no" --colscore "multiquery" #--lora_rank 32

# torchrun --master_port 12345 --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 5 --learning_rate 5e-5 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --divq_coeff 0 --dataset "train50wordsdebug100setversion_splitup" --div_coeff 0 --temp 0.02 --doclen 1 --querylen 50 --dodefaulttrain "no" --compile "no" --save_strat "no" # --colscore "multipos" #--lora_rank 32

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_500k" --div_coeff 0 --temp 0.02 --dodefaulttrain "yes" --compile "yes" --querylen 32 

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_500k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "yes" --querylen 1 --doclen 1

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_500k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "yes" --querylen 1 --doclen 1 --embsize 1024

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_500k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "yes" --querylen 1 --doclen 1 --embsize 10000

# torchrun --nproc_per_node=1 scripts/train.py --batch_size 1 --num_epochs 1 --learning_rate 3e-6 --model_name "Qwen/Qwen3-Embedding-8B" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_500k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --querylen 1 --doclen 1 --embsize 128


# torchrun --nproc_per_node=1 scripts/train.py --batch_size 1 --num_epochs 1 --learning_rate 3e-6 --model_name "Qwen/Qwen3-Embedding-8B" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_500k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --querylen 1 --doclen 1 --embsize 128

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 3e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --querylen 1 --doclen 1 --embsize 1024 --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 6e-5 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "sbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 3e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --traintype "colbert" --querylen 1 --doclen 1 --embsize 1024 --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 9e-5 --model_name "google-bert/bert-large-uncased" --traintype "sbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no"


# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 9e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 1024 --querylen 1 --doclen 1

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 9e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 8 --querylen 1 --doclen 1


# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 9e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 10 --doclen 10

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 6e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 1 --doclen 10
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 6e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 1 --doclen 100

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 6e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 1 --doclen 1

# torchrun --nproc_per_node=2 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 6e-5 --model_name "Alibaba-NLP/gte-modernbert-base" --traintype "colbert" --divq_coeff 0 --dataset "fiqa_train_retrieval" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 10 --doclen 10 --colscore "extend"

# torchrun --nproc_per_node=2 scripts/reasoncolbert_reprod.py

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 64 --num_epochs 1 --learning_rate 6e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "msmarco_100k" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 40 --doclen 40 --colscore "extend"

torchrun --nproc_per_node=8 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 6e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "fiqa_train_retrieval" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 1 --doclen 1 --colscore "extend"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 6e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "fiqa_train_retrieval" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 10 --doclen 10 --colscore "extend"

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 16 --num_epochs 1 --learning_rate 6e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --divq_coeff 0 --dataset "fiqa_train_retrieval" --div_coeff 0 --temp 0.02 --dodefaulttrain "no" --compile "no" --gcheck "no" --embsize 128 --querylen 40 --doclen 40 --colscore "extend"

