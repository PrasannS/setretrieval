# torchrun --nproc_per_node=8 scripts/train.py > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 32 --num_epochs 3 --learning_rate 3e-6 > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# from here trying constant lr
# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 2 --num_epochs 11 --learning_rate 3e-6 --model_name "google-bert/bert-base-uncased" --traintype "sbert" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "sbert" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --traintype "sbert" --dataset "gemini_ntrain_ptest" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-5 --model_name "google-bert/bert-base-uncased" --dataset "gemini_ntrain_ptest" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "Qwen/Qwen3-Embedding-0.6B" --dataset "gemini_ntrain_ptest" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &

# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 11 --learning_rate 3e-6 --model_name "google-bert/bert-large-uncased" --traintype "sbert" --dataset "gemini_ntrain_ntest" > logs/train$(date +%Y%m%d_%H%M%S).log 2>&1 &