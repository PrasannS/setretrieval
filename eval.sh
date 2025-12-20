# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 10
# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path data/datastores/wikipedia_docs_150k --eval_set_path data/evalsets/settest_v1 --k 10

# now do colbert
# python scripts/wikipedia_eval.py --index_type colbert --model_name cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_150k --eval_set_path data/evalsets/settest_v1 --k 100

# python scripts/wikipedia_eval.py --index_type random --model_name cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-0.6B --dataset_path data/datastores/wikipedia_docs_150k --eval_set_path data/evalsets/settest_v1 --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name cache/sbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06-sbert/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-0.6B --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1_paraphrased --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-0.6B --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name cache/sbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06-sbert/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1 --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name cache/sbert_training/contrastive-google-bert_bert-large-uncased-bs8-e11-lr3e-06-sbert/checkpoint-4631 --dataset_path data/datastores/wikipedia_docs_15k --eval_set_path data/evalsets/settest_v1_paraphrased --k 100

# # BM25 results on both sets
# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100
# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path propercache/data/datastores/wikipedia_docs_150k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# # Random results on both sets
# python scripts/wikipedia_eval.py --index_type random --model_name random --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100
# python scripts/wikipedia_eval.py --index_type random --model_name random --dataset_path propercache/data/datastores/wikipedia_docs_150k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# #QWEN NO TRAINED
# python scripts/wikipedia_eval.py --index_type single --model_name propercache/models/qwenemb06B --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name propercache/models/qwenemb06B --dataset_path propercache/data/datastores/wikipedia_docs_150k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# # COLBERT NORMAL
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e11-lr3e-05-gemini_ntrain_ptest/checkpoint-1500 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e11-lr3e-05-gemini_ntrain_ptest/checkpoint-1500 --dataset_path propercache/data/datastores/wikipedia_docs_150k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# # ColBERT mixed results on both sets
# # propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-gemini_multisingleposmix_ptest/checkpoint-4210
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-gemini_multisingleposmix_ptest/checkpoint-4210 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-gemini_multisingleposmix_ptest/checkpoint-4210 --dataset_path propercache/data/datastores/wikipedia_docs_150k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# # SBERT normal trained results on both sets
# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-google-bert_bert-base-uncased-bs8-e11-lr3e-06-sbert-gemini_ntrain_ptest/checkpoint-2000 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-google-bert_bert-base-uncased-bs8-e11-lr3e-06-sbert-gemini_ntrain_ptest/checkpoint-2000 --dataset_path propercache/data/datastores/wikipedia_docs_150k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# # SBERT MIXED
# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-propercache_models_bertbase-bs8-e11-lr3e-06-sbert-gemini_multisingleposmix_ptest/checkpoint-4631 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-propercache_models_bertbase-bs8-e11-lr3e-06-sbert-gemini_multisingleposmix_ptest/checkpoint-4631 --dataset_path propercache/data/datastores/wikipedia_docs_150k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# # eval for noun tasks
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-nountraining10words/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest10 --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-nountraining100words/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# # for single
# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-sbert-nountraining10words/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest10 --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-sbert-nountraining100words/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-sbert-nountraining100words/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100


# python scripts/wikipedia_eval.py --index_type divcolbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-nountraining10words/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-nountraining10words/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100



# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/sbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-sbert-nountraining100words/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

# python scripts/wikipedia_eval.py --index_type divcolbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-nountraining100words-maxmax-div1.0/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-abstract_relevant_train_40k_filtered-maxsim-div1.0-qlen256/checkpoint-4460  --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-gemini_abstracttrain-maxsim-div1.0/checkpoint-4210  --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-gemini_gutenbergtrain-maxsim-div1.0/checkpoint-4210  --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e10-lr3e-06-nountraining100words-maxsim-div1.0-qlen256/checkpoint-1560  --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-propercache_models_bertbase-bs8-e20-lr3e-05-nountraining100words-maxsim-div1.0-qlen256/checkpoint-2500 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e20-lr3e-06-nountraining100words-maxsim-div1.0-qlen256/checkpoint-3120 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

