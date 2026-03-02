#!/bin/bash


MODEL=output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final
MODEL=output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv400-pqv0-pdv0-embsize128/final



prompt_try() {
    local PROMPT="$1"
    local API_MODEL="$2"
    local EXTRA_FLAGS="${3:-}"
    # python scripts/paraphrase_robustness_eval.py --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "$PROMPT" --mode pos --concat_original --k 10 --max_concurrent 50 --llm_model $API_MODEL $EXTRA_FLAGS

    python scripts/paraphrase_robustness_eval.py --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv400-pqv0-pdv0-embsize128/final --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "$PROMPT" --mode pos --concat_original --k 10 --max_concurrent 50 --llm_model $API_MODEL $EXTRA_FLAGS
}

# dv400-only version for faster iteration (skip dv100)
prompt_try_dv400() {
    local PROMPT="$1"
    local API_MODEL="$2"
    local MODE="$3"
    local EXTRA_FLAGS="${4:-}"
    python scripts/paraphrase_robustness_eval.py --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "$PROMPT" --mode $MODE --k 10 --max_concurrent 50 --llm_model $API_MODEL $EXTRA_FLAGS
    # python scripts/paraphrase_robustness_eval.py --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv400-pqv0-pdv0-embsize128/final --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "$PROMPT" --mode $MODE --k 10 --max_concurrent 50 --llm_model $API_MODEL $EXTRA_FLAGS
}


# NOTE baseline performance: dv100 is 0.36, dv400 is 0.36

# NOTE for the below prompt dv100 gets 0.37 dv400 gets 0.39. This is better than the other prompts further below
# A: prompt_try "Can you elaborate on ideas from the following text (3-4 sentences)? Avoid repeating info and try to add new information:\\n\\n{}" gemini-2.5-flash-lite --concat_original

# B: prompt_try "Can you elaborate on ideas from the following text (3-4 sentences)? Only add new content and information not stated in the original text. Avoid re-using too many words:\\n\\n{}" gemini-2.5-flash-lite --concat_original

# C: prompt_try "Can you elaborate on ideas from the following text (3-4 sentences)? Only add new content and information not stated in the original text. Avoid re-using too many words:\\n\\n{}" gemini-2.5-flash --concat_original

# D: prompt_try "Can you add extra analysis/information not covered in the original text (don't repeat anything stated or implied)? Add 4-5 sentences:\\n\\n{}" gemini-2.5-flash-lite --concat_original

# ============================================================
# NEW EXPERIMENTS — targeting dv400 > 0.45
# Key insight: ColBERT MaxSim needs query-token ↔ doc-token matches.
# The core gap: queries are questions, docs are answers → vocab mismatch.
# Strategy: add query-style vocabulary INTO positive docs.
# ============================================================

# E (pseudo-questions / reverse-HyDE): generate questions the doc answers.
#   Rationale: directly creates query-vocabulary tokens in the document.
#   Tokens like "How do I deposit...", "Can I...", "What should I do when..."
#   will have very high similarity to actual query tokens.
#   HYPOTHESIS: biggest win — bridges Q&A vocab gap directly.
# 0.38
# prompt_try_dv400 "Generate 3 natural questions (starting with How, What, Can, Why, Should, etc.) that someone would type when searching for the information in this financial text. One question per line:\n\n{}" gemini-2.5-flash-lite "--concat_original"

# F (Q&A reformulation, no concat): rewrite as "Q: ... A: ..." entry.
#   Rationale: full reformulation combining question-style + answer-style text.
#   No concat_original since the reformulation includes the content.
#   Potentially very clean: every query token can match the Q part.
# 0.40
# prompt_try_dv400 "Rewrite the following financial text as a Q&A entry: first write 1-2 natural questions (as \"Q:\") that this text answers, then write the answer (as \"A:\") in clear financial language. Preserve all key facts:\n\n{}" gemini-2.5-flash-lite "pos"

# 0.38, which is a win I think
# prompt_try_dv400 "Rewrite the following financial text as a Q&A entry: first write 1-2 natural questions (as \"Q:\") that this text answers, then write the answer (as \"A:\") in clear financial language. Preserve all key facts:\n\n{}" gemini-2.5-flash-lite "all"

# prompt_try_dv400 "Rewrite the following financial text as a Q&A entry: first write 1-2 natural questions (as \"Q:\") that this text answers, then write the answer (as \"A:\") in clear financial language. Preserve all key facts:\n\n{}" gemini-2.5-flash-lite "all"

# multichunk with 400 model
python scripts/paraphrase_robustness_eval.py --model_name output/ModernBERT-base/ModernBERT-base-pylate-pairwise-0.0003-qv1-dv100-embsize128/final --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt  "Rewrite the following financial text as a Q&A entry: first write 1-2 natural questions (as \"Q:\") that this text answers, then write the answer (as \"A:\") in clear financial language. Preserve all key facts:\n\n{}" --mode "all" --k 10 --max_concurrent 50 --llm_model gemini-2.5-flash-lite --use_multichunk


# G (vocabulary bridge): rephrase using diverse synonyms and search-style terms.
#   Rationale: adds alternative phrasings that may match query vocabulary better
#   than the original answer-style text.
# 0.37
# prompt_try_dv400 "Rephrase the following financial text using different vocabulary: use synonyms, alternative financial terms, and question-style phrasing that someone searching for this information would use. Produce 3-4 sentences:\n\n{}" gemini-2.5-flash-lite "--concat_original"

# H (context bridge + pseudo-questions, flash quality):
#   Like E but with a higher-quality model, in case flash-lite questions are poor.
# 0.39
# prompt_try_dv400 "Generate 3 natural questions (starting with How, What, Can, Why, Should, etc.) that someone would type when searching for the information in this financial text. One question per line:\n\n{}" gemini-2.5-flash "--concat_original"

# I (explicit query-vocabulary injection):
# 0.38
#   Ask the model to literally rephrase as someone searching would say it.
# prompt_try_dv400 "Imagine someone is typing a search query to find the following financial information. Write 3-4 versions of how they might phrase their question (different angles, different vocabulary), then add a brief plain-language summary:\n\n{}" gemini-2.5-flash-lite "--concat_original"



# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" --mode pos --llm_model gemini-2.5-flash-lite --k 10 --concat_original

# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" --mode pos --llm_model gemini-2.5-flash-lite --k 10

# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Can you continue the following text for 2-3 sentences:\\n\\n{}" --mode none --llm_model gemini-2.5-flash-lite --k 10
# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Can you continue the following text for 2-3 sentences:\\n\\n{}" --mode pos --llm_model gemini-2.5-flash-lite --k 10 --concat_original



# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" --mode pos --llm_model gemini-2.5-flash-lite --k 10

# python scripts/paraphrase_robustness_eval.py --model_name $MODEL --eval_set_path propercache/data/evalsets/fiqa_testset --corpus_path propercache/data/datastores/fiqacorpus --paraphrase_prompt "Rewrite the following in simpler language:\\n\\n{}" --mode none --llm_model gemini-2.5-flash-lite --k 10
