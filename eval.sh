# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 10
# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path data/datastores/wikipedia_docs_150k --eval_set_path data/evalsets/settest_v1 --k 10

# now do colbert
# python scripts/wikipedia_eval.py --index_type colbert --model_name cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_150k --eval_set_path data/evalsets/settest_v1 --k 100

# python scripts/wikipedia_eval.py --index_type random --model_name cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-0.6B --dataset_path data/datastores/wikipedia_docs_150k --eval_set_path data/evalsets/settest_v1 --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name cache/sbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06-sbert/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100

python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-0.6B --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100

