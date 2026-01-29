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

# python scripts/wikipedia_eval.py --index_type single --model_name intfloat/multilingual-e5-large-instruct --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100 --forceredo "yes" --save_preds "e5notrainwiki"

# noun qmod eval
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr3e-05-v2nountraining50words100000ndps1-maxsim-divd0.0-divq0.0-qlen50-cosine-temp0.5 --qmod_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-wiki2gramtrain50000samples-maxsim-divd0.0-divq0.0-qlen32-cosine-temp0.05-omoddocument --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50 --save_preds "bertlargewithqmodsnoun50"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-wiki2gramtrain50000samples-maxsim-divd0.0-divq0.0-qlen4-cosine-temp0.5-omodneither --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50 --save_preds "bertlargewithqmodsnoun50"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand150dwords-maxsim-divd0.0-divq0.0-qv8-dv512-cosine-temp0.5-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/v2evaltdstore150words50pos100k --eval_set_path propercache/data/evalsets/v2testset150words50pos --k 50 --save_preds "bertlargeinvtoy150"


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_gutenbergtrain-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_ntrain_ptest-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_ntrain_ptest-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_ntrain_ptest-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gemini_ntrain_ptest-maxsim-divd0.0-divq1.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100


# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e2-lr2e-05-fineweb_gmini_300k-maxsim-divd0.0-divq1.0-qv48-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes/checkpoint-4640 --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e2-lr2e-05-fineweb_gmini_300k-maxsim-divd0.0-divq1.0-qv48-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes/checkpoint-4640  --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e2-lr2e-05-fineweb_gmini_300k-maxsim-divd0.0-divq1.0-qv48-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes/checkpoint-4640 --dataset_path propercache/data/datastores/absdata_domain1 --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e2-lr2e-05-fineweb_gmini_300k-maxsim-divd0.0-divq1.0-qv48-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes/checkpoint-4640 --dataset_path propercache/data/datastores/absdata_domain3 --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100


# MODEL=propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gutenberg_gmini_300k-maxsim-divd0.0-divq1.0-qv48-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes

# MODEL=propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr2e-05-gutenbergnoshuff_gmini_300k-maxsim-divd0.0-divq1.0-qv48-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes


run_wikipedia_eval() {
    local MODEL=$1
    local SAVE_PREDS=$2
    local FORCE_REDO=$3
    local ETYPE=$4
    
    if [ -z "$MODEL" ]; then
        echo "Error: Model parameter is required"
        echo "Usage: run_wikipedia_eval <model_name>"
        return 1
    fi
    
    echo "Running evaluations for model: $MODEL"
    echo "========================================="
    
    echo "Running Wikipedia docs evaluation..."
    python scripts/wikipedia_eval.py --index_type $ETYPE --model_name "$MODEL" \
        --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont \
        --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100 --save_preds $SAVE_PREDS --forceredo $FORCE_REDO
    
    echo "Running Gutenberg evaluation..."
    python scripts/wikipedia_eval.py --index_type $ETYPE --model_name "$MODEL" \
        --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont \
        --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100 --save_preds $SAVE_PREDS --forceredo $FORCE_REDO
    
    echo "Running bioabs evaluation..."
    python scripts/wikipedia_eval.py --index_type $ETYPE --model_name "$MODEL" \
        --dataset_path propercache/data/datastores/absdata_domain1 \
        --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100 --save_preds $SAVE_PREDS --forceredo $FORCE_REDO
    
    echo "Running physabs evaluation..."
    python scripts/wikipedia_eval.py --index_type $ETYPE --model_name "$MODEL" \
        --dataset_path propercache/data/datastores/absdata_domain3 \
        --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100 --save_preds $SAVE_PREDS --forceredo $FORCE_REDO
    
    echo "========================================="
    echo "All evaluations completed for model: $MODEL"
}


run_wikipedia_eval_vectors() {
    local MODEL=$1
    local QVECS=$2
    local DVECS=$3
    local EBSIZE=$4
    local FORCE_REDO=$5
    local USEFAST=$6
    local SAVE_PREDS=$7

    if [ -z "$MODEL" ]; then
        echo "Error: Model parameter is required"
        echo "Usage: run_wikipedia_eval <model_name>"
        return 1
    fi
    
    echo "Running evaluations for model: $MODEL"
    echo "========================================="
    
    echo "Running Wikipedia docs evaluation..."
    python scripts/wikipedia_eval.py --index_type colbert --model_name "$MODEL" \
        --dataset_path propercache/data/datastores/wikipedia_docs_10k_decont \
        --eval_set_path propercache/data/evalsets/wiki_newpipetest --k 100 --colbert_qvecs $QVECS --colbert_dvecs $DVECS --colbert_ebsize $EBSIZE --forceredo $FORCE_REDO --colbert_usefast $USEFAST --save_preds $SAVE_PREDS
    
    echo "Running Gutenberg evaluation..."
    python scripts/wikipedia_eval.py --index_type colbert --model_name "$MODEL" \
        --dataset_path propercache/data/datastores/gutenberg_chunks_10k_decont \
        --eval_set_path propercache/data/evalsets/gutenberg_newpipetest --k 100 --colbert_qvecs $QVECS --colbert_dvecs $DVECS --colbert_ebsize $EBSIZE --forceredo $FORCE_REDO --colbert_usefast $USEFAST --save_preds $SAVE_PREDS
    
    echo "Running bioabs evaluation..."
    python scripts/wikipedia_eval.py --index_type colbert --model_name "$MODEL" \
        --dataset_path propercache/data/datastores/absdata_domain1 \
        --eval_set_path propercache/data/evalsets/bioabs_newpipetest --k 100 --colbert_qvecs $QVECS --colbert_dvecs $DVECS --colbert_ebsize $EBSIZE --forceredo $FORCE_REDO --colbert_usefast $USEFAST --save_preds $SAVE_PREDS
    
    echo "Running physabs evaluation..."
    python scripts/wikipedia_eval.py --index_type colbert --model_name "$MODEL" \
        --dataset_path propercache/data/datastores/absdata_domain3 \
        --eval_set_path propercache/data/evalsets/physabs_newpipetest --k 100 --colbert_qvecs $QVECS --colbert_dvecs $DVECS --colbert_ebsize $EBSIZE --forceredo $FORCE_REDO --colbert_usefast $USEFAST --save_preds $SAVE_PREDS
    
    echo "========================================="
    echo "All evaluations completed for model: $MODEL"
}

# run_wikipedia_eval propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr1e-06-gutenberg_gmini_300k-maxsim-divd0.0-divq0.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes-embsize128 "bertgutlarge300ktrain" "yes" colbert


# run_wikipedia_eval Qwen/Qwen3-Embedding-4B "qwen4bsingle" "partial" single



# run_wikipedia_eval propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr8e-06-gutenberg_gmini_300k-maxsim-divd0.0-divq0.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes "e5gut300ktrain" "yes" colbert

# run_wikipedia_eval propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr8e-06-gutenberg_gmini_30k-maxsim-divd0.0-divq0.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes "e5gut30ktrain" "yes" colbert

# run_wikipedia_eval propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr8e-06-gutenberg_gmini_90k-maxsim-divd0.0-divq0.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes "gut90ktrain" "yes"

# run_wikipedia_eval propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes
# run_wikipedia_eval propercache/cache/colbert_training/contrastive-intfloat_multilingual-e5-large-instruct-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd10.0-divq0.0-qv32-dv511-cosine-temp0.02-omodneither-dodefaulttrainyes




# run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv10-cosine-temp0.02-omodneither-dodefaulttrainno-lora16 1 10 32 "yes"

# run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno 1 1 16

# run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno 1 1

# run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv100-cosine-temp0.02-omodneither-dodefaulttrainno 1 100

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand10dwords-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50 --colbert_qvecs 1 --colbert_dvecs 1

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr5e-05-v2nountrain100000rand10dwords-maxsim-divd0.0-divq0.0-qv1-dv10-cosine-temp0.02-omodneither-dodefaulttrainno --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50 --colbert_qvecs 1 --colbert_dvecs 10

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr5e-06-v2nountrain100000rand10dwords-maxsim-divd0.0-divq0.0-qv1-dv10-cosine-temp0.02-omodneither-dodefaulttrainno --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50 --colbert_qvecs 1 --colbert_dvecs 10 --forceredo "yes"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr5e-06-v2nountrain100000rand10dwords-maxsim-divd0.0-divq0.0-qv32-dv10-cosine-temp0.02-omodneither-dodefaulttrainno --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50 --colbert_qvecs 32 --colbert_dvecs 10

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr5e-06-v2nountrain100000rand10dwords-maxsim-divd0.0-divq0.0-qv1-dv20-cosine-temp0.02-omodneither-dodefaulttrainno --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50 --colbert_qvecs 1 --colbert_dvecs 20



# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand10dwords-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno --dataset_path propercache/data/datastores/evaltdstore10words50pos100k --eval_set_path propercache/data/evalsets/testset10words50pos --k 50 

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr2e-06-v2nountraining50words100000ndps1-maxsim-divd0.0-divq1.0-qv100-dv10-cosine-temp0.02-omodneither-dodefaulttrainyes-embsize128 --dataset_path propercache/data/datastores/allnounsfixed --eval_set_path propercache/data/evalsets/v2nountest50 --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand50dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore50words250pos100k --eval_set_path propercache/data/evalsets/v2testset50words50pos --k 50

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr2e-05-v2nountrain100000rand50dwords-maxsim-div1.0-qlen32-cosine --dataset_path propercache/data/datastores/v2evaltdstore50words250pos100k --eval_set_path propercache/data/evalsets/v2testset50words250pos --k 250 --forceredo "no" --save_preds "bertlarge50dwordstrain"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-v2nountrain100000rand50dwords-maxsim-divd1.0-divq0.0-qv8-dv100-cosine-temp0.02-omodneither-dodefaulttrainyes-embsize128 --dataset_path propercache/data/datastores/v2evaltdstore50words250pos100k --eval_set_path propercache/data/evalsets/v2testset50words250pos --k 250 --forceredo "yes" --save_preds "qwen06bsingle50dwordstrain"

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-mathtask1_expr2num_min0_max5000_md0_0_hardneg0.0_trainingsize100000-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno-embsize128 --dataset_path propercache/data/datastores/mathtask1_expr2num_min0_max5000_md0_0 --eval_set_path propercache/data/evalsets/mathtask1_expr2num_min0_max5000_md0_0_evalsize500 --k 1 --save_preds "qwen06bsinglemathtask1d1" --forceredo "yes" --colbert_dvecs 1 --colbert_qvecs 1

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr5e-06-mathtask1_expr2num_min0_max5000_md0_0_hardneg0.0_trainingsize100000-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno-embsize8 --dataset_path propercache/data/datastores/mathtask1_expr2num_min0_max5000_md0_0 --eval_set_path propercache/data/evalsets/mathtask1_expr2num_min0_max5000_md0_0_evalsize500 --k 100 --save_preds "qwen06bsinglemathtask1d1" --forceredo "yes" --colbert_dvecs 1 --colbert_qvecs 1

# check new impossible setting 
# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-4B --dataset_path propercache/data/datastores/wikipedia_docs_15k_with_impossible_poschunks --eval_set_path propercache/data/evalsets/wikipedia_eval_impossible_500_processed --k 100 --save_preds "qwen4bimpossible" --forceredo "yes"


# python scripts/wikipedia_eval.py --index_type single --model_name Qwen/Qwen3-Embedding-0.6B --dataset_path propercache/data/datastores/wikipedia_docs_15k_with_impossible_poschunks --eval_set_path propercache/data/evalsets/wikipedia_eval_impossible_500_processed --k 100 --save_preds "qwen06bimpossible" --forceredo "yes"

# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-google-bert_bert-large-uncased-bs8-e1-lr5e-06-sbert-wikipedia_train_impossible_20k_processed --dataset_path propercache/data/datastores/wikipedia_docs_15k_with_impossible_poschunks --eval_set_path propercache/data/evalsets/wikipedia_eval_impossible_500_processed --k 100 --save_preds "sbertlarge20kimpossible" --forceredo "yes" 

# python scripts/wikipedia_eval.py --index_type single --model_name propercache/cache/sbert_training/contrastive-facebook_opt-1.3b-bs8-e1-lr5e-06-sbert-wikipedia_train_impossible_20k_processed --dataset_path propercache/data/datastores/wikipedia_docs_15k_with_impossible_poschunks --eval_set_path propercache/data/evalsets/wikipedia_eval_impossible_500_processed --k 100 --save_preds "singleopt1b20kimpossible" --forceredo "yes" 


# msmarco eval stuff
# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs64-e1-lr3e-06-msmarco_500k-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno-embsize128 --dataset_path propercache/data/datastores/msmarco_20k_docs --eval_set_path propercache/data/evalsets/msmarco_20k_docs --k 100 --save_preds "colbertlargemsmarco_q1d1"  --colbert_dvecs 1 --colbert_qvecs 1

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs64-e1-lr3e-06-msmarco_500k-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno-embsize1024 --dataset_path propercache/data/datastores/msmarco_20k_docs --eval_set_path propercache/data/evalsets/msmarco_20k_docs --k 100 --save_preds "colbertlargemsmarco_q1d1_embsize1024"  --colbert_dvecs 1 --colbert_qvecs 1

# python scripts/wikipedia_eval.py --index_type colbert --model_name propercache/cache/colbert_training/contrastive-google-bert_bert-large-uncased-bs64-e1-lr3e-06-msmarco_500k-maxsim-divd0.0-divq0.0-qv32-dv512-cosine-temp0.02-omodneither-dodefaulttrainyes-embsize128 --dataset_path propercache/data/datastores/msmarco_20k_docs --eval_set_path propercache/data/evalsets/msmarco_20k_docs --k 100 --save_preds "colbertlargemsmarco_q32d512_embsize128" 

python scripts/wikipedia_eval.py --index_type bm25 --model_name bm25 --dataset_path propercache/data/datastores/msmarco_20k_docs --eval_set_path propercache/data/evalsets/msmarco_20k_docs --k 100 --save_preds "bm25" 

# vector stuff
## run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv100-cosine-temp0.02-omodneither-dodefaulttrainno 1 100 32 "yes" "yes" "qwen30kqv1dv100dim128"
## run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv100-cosine-temp0.02-omodneither-dodefaulttrainno-embsize512 1 100 32 "yes" "no" "qwen30kqv1dv100dim512"
## run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv100-cosine-temp0.02-omodneither-dodefaulttrainno 100 100 32 "no" "no"
## run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv1-dv1-cosine-temp0.02-omodneither-dodefaulttrainno 1 1 32 "yes" "yes" "qwen30kqv1dv1dim128"
## run_wikipedia_eval_vectors propercache/cache/colbert_training/contrastive-Qwen_Qwen3-Embedding-0.6B-bs8-e1-lr8e-06-gutenberg_gmini_30k_nosame-maxsim-divd0.0-divq0.0-qv100-dv1-cosine-temp0.02-omodneither-dodefaulttrainno 100 1 32 "yes" "yes" "qwen30kqv100dv1dim128"
