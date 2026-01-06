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

torchrun --nproc_per_node=1 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2.1e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "wiki1gramtrain50000samplebalancedcontam" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 0.5
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "wiki2gramtrain50000samples" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 0.05
