export CUDA_VISIBLE_DEVICES=7
python src/setretrieval/train/train_api.py --checkpoint propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-v2nountraining50words100000ndps1-maxsim-divd0.0-divq0.0-qlen50-cosine-temp0.5 --device cuda --host 0.0.0.0 --port 5000 --debug --batch-size 8

# wait for 5 seconds
# sleep 5
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# # train the model
# torchrun --nproc_per_node=4 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 2.1e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "" --querylen 32 --divq_coeff 0.0 --div_coeff 0 --temp 0.5 --api_model "document"


# torchrun --nproc_per_node=8 scripts/train.py --batch_size 8 --num_epochs 1 --learning_rate 3e-5 --model_name "google-bert/bert-large-uncased" --traintype "colbert" --dataset "v2nountraining50words100000ndps1" --querylen 50 --divq_coeff 0.0 --temp 0.02 --api_model "document"

