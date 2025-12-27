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

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e10-lr3e-06-gemini_ntrain_ptest-maxsim-div1.0-qlen64/checkpoint-2500 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e10-lr3e-06-gemini_ntrain_ptest-maxsim-div1.0-qlen64/checkpoint-2500 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e10-lr3e-06-wiki_gemini_mini_train_10k-maxsim-div1.0-qlen32-cosine/checkpoint-4210 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e10-lr3e-06-wiki_gemini_mini_train_full-maxsim-div1.0-qlen32-cosine/checkpoint-46400 --dataset_path propercache/data/datastores/wikipedia_docs_15k --eval_set_path propercache/data/evalsets/settest_v1_paraphrased --k 100


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountraining1words100000-maxsim-div1.0-qlen32-cosine/checkpoint-1562 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest1 --k 1

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountraining1words100000-maxsim-div1.0-qlen32-cosine/checkpoint-7810 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest1 --k 1

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountraining5words100000-maxsim-div1.0-qlen32-cosine/checkpoint-1562 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest5 --k 5

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountraining5words100000-maxsim-div1.0-qlen32-cosine/checkpoint-7810 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest5 --k 5

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountraining10words100000-maxsim-div1.0-qlen32-cosine/checkpoint-7810 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest10 --k 10

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountraining25words100000-maxsim-div1.0-qlen32-cosine/checkpoint-7810 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountraining25words100000ndps5-maxsim-div1.0-qlen32-cosine/checkpoint-7810 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountraining25words100000ndps25-maxsim-div1.0-qlen32-cosine/checkpoint-7810 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-06-nountestset25-maxsim-div1.0-qlen32-cosine/checkpoint-1950 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-05-nountestset25-maxsim-div1.0-qlen32-cosine/checkpoint-1950 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e5-lr3e-05-nountraining25words100000ndps5-maxsim-div1.0-qlen32-cosine/checkpoint-4686 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e5-lr3e-05-nountraining25words100000-maxsim-div1.0-qlen32-cosine/checkpoint-1562 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25



# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-05-nountraining25words100000-maxsim-div1.0-qlen32-cosine/checkpoint-7810 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

## START 25 ablate eval


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e5-lr3e-05-nountraining25words100000-maxsim-div1.0-qlen32-cosine/checkpoint-1562 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e5-lr3e-05-nountraining25words100000ndps5-maxsim-div1.0-qlen32-cosine/checkpoint-1562 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining25words100000ndps25-maxsim-div1.0-qlen32-cosine/checkpoint-1562 --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25


## START 100 ablate eval


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100words100000ndps100-maxsim-div1.0-qlen128-cosine --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100words100000ndps1-maxsim-div1.0-qlen128-cosine --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

# skew set evals, for both 25 and 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100kuniform5_100-maxsim-div1.0-qlen128-cosine --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100kpower5_100-maxsim-div1.0-qlen128-cosine --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest25 --k 25

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100kuniform5_100-maxsim-div1.0-qlen128-cosine --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100kpower5_100-maxsim-div1.0-qlen128-cosine --dataset_path propercache/data/datastores/allnouns --eval_set_path propercache/data/evalsets/nountest100 --k 100

python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e1-lr3e-05-nountrain100000minimal10dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50

python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e1-lr3e-05-nountrain100000rand10dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50