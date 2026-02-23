MODEL=output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final


# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" --mode pos --llm_model gemini-2.5-flash-lite --k 10 --concat_original

# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" --mode pos --llm_model gemini-2.5-flash-lite --k 10

python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Can you continue the following text for 2-3 sentences:\\n\\n{}" --mode pos --llm_model gemini-2.5-flash-lite --k 10 --concat_original


# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" --mode pos --llm_model gemini-2.5-flash-lite --k 10

# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" --mode none --llm_model gemini-2.5-flash-lite --k 10
