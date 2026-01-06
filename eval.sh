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

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100kpower5_100-maxsim-div1.0-qlen128-cosine --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/nountest100 --k 100 --forceredo "yes"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e1-lr3e-05-nountrain100000minimal10dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e1-lr3e-05-nountrain100000rand10dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-0.6B --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-4B --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-0.6B --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-4B --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs1-e1-lr3e-05-sbert-gemini_ntrain_ptest --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100
# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs1-e1-lr3e-05-sbert-gemini_ntrain_ptest --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-wiki_gemini_mini_train_10k-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-wiki_gemini_mini_train_full-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# # wiki version of the last two
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-wiki_gemini_mini_train_10k-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-wiki_gemini_mini_train_full-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# eval wiki model on all the newpipetest sets

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_ntrain_ptest-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100

# now eval model trained on gutenberg on all eval sets (wiki, gutenberg, science abstract)

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100

# now eval model trained on science abstract on all eval sets (wiki, gutenberg, science abstract)

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-abstract_relevant_train_30k_newfilt-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-abstract_relevant_train_30k_newfilt-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-abstract_relevant_train_30k_newfilt-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-abstract_relevant_train_30k_newfilt-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100

# now eval last science model on all eval sets

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_abstracttrain-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_abstracttrain-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_abstracttrain-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-gemini_abstracttrain-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100

# do evals for bm25 now

# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100

# do evals for random

# python scripts/wikipedia_eval.py --index_type random --model_name random --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type random --model_name random --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type random --model_name random --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type random --model_name random --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100

# do evals for qwen 4B
# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-4B --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-4B --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-4B --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-4B --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100

# noun randset eval
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-nountrain100000rand10dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-nountrain100000minimal10dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-nountrain100000minimal10dwords-maxsim-div1.0-qlen128-cosine --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-nountrain100000minimal50dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/evaltdstore50words50pos100k --eval_set_path propercache/data/evalsets/testset50words50pos --k 50


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand10dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/v2testset10words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand50dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore50words250pos100k --eval_set_path propercache/data/evalsets/v2testset50words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand10dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore10words250pos100k --eval_set_path propercache/data/evalsets/v2testset10words250pos --k 250 --forceredo "yes"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand50dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore50words250pos100k --eval_set_path propercache/data/evalsets/v2testset50words250pos --k 250

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand250dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore250words250pos100k --eval_set_path propercache/data/evalsets/v2testset250words250pos --k 250

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand50dwords-maxsim-div0.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore50words50pos100k --eval_set_path propercache/data/evalsets/v2testset50words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand50dwords-maxsim-divd1.0-divq0.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore50words50pos100k --eval_set_path propercache/data/evalsets/v2testset50words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand50dwords-maxsim-divd0.0-divq1.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore50words50pos100k --eval_set_path propercache/data/evalsets/v2testset50words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand50dwords-maxsim-divd0.05-divq0.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore50words50pos100k --eval_set_path propercache/data/evalsets/v2testset50words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-v2nountraining50words100000ndps1-maxsim-divd0.0-divq0.0-qlen50-cosine --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-v2nountraining50words100000ndps1-maxsim-divd0.0-divq100.0-qlen50-cosine --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-v2nountraining50words100000ndps1-maxsim-divd0.0-divq1.0-qlen50-cosine --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-v2nountraining50words100000ndps1-maxmax-divd0.0-divq10.0-qlen50-cosine --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50
# python scripts/wikipedia_eval.py --index_type divcolbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-v2nountraining50words100000ndps1-maxmax-divd0.0-divq10.0-qlen50-cosine --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50


# do colbert eval for new gutenberg train stuff

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.02 --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq0.0-qlen32-cosine-temp1.0 --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd10.0-divq0.0-qlen32-cosine-temp0.02 --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-v2nountraining50words100000ndps1-maxsim-divd0.0-divq0.0-qlen50-cosine-temp0.5 --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50

# check eval for e5 model (first in single, then colbert trained model)

# python scripts/wikipedia_eval.py --index_type single --model_name intfloat/multilingual-e5-large-instruct --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.05 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# same for stella
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python scripts/wikipedia_eval.py --index_type single --model_name NovaSearch/stella_en_1.5B_v5 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100 --forceredo "yes"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-NovaSearch_stella_en_1.5B_v5-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.05 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100 --forceredo "yes"

# now for new toy task
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-wiki1gramtrain50000samples-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.05 --dataset_path propercache/data/datastores/wiki1gramdstore50000 --eval_set_path propercache/data/evalsets/evalwiki1grameval50000samples --k 50 --forceredo "yes" --save_preds "yes"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-wiki1gramtrain50000samplebalanced-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.5 --dataset_path propercache/data/datastores/wiki1gramdstore50000balanced --eval_set_path propercache/data/evalsets/evalwiki1grameval50000samplesbalanced --k 100 --forceredo "yes" --save_preds "yes"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-wiki1gramtrain50000samplebalancedcontam-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.5 --dataset_path propercache/data/datastores/wiki1gramdstore50000balancedcontam --eval_set_path propercache/data/evalsets/evalwiki1grameval50000samplesbalancedcontam --k 100 --save_preds "yes" --forceredo "yes"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-wiki1gramtrain50000samplebalancedcontam-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.5 --dataset_path propercache/data/datastores/wiki1gramdstore50000balancedcontam --eval_set_path propercache/data/evalsets/evalwiki1grameval50000samplesbalancedcontam --k 100 --save_preds "yes" --forceredo "yes"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.05 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100 --forceredo "yes" --save_preds "guttraine5"

python scripts/wikipedia_eval.py --index_type single --model_name intfloat/multilingual-e5-large-instruct --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100 --forceredo "yes" --save_preds "e5notrainwiki"
