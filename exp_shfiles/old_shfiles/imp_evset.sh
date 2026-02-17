# python scripts/generate_impossible_eval_set.py \
#   --datastore_path propercache/data/datastores/wikipedia_docs_200k_cleaned \
#   --output_eval_path propercache/data/evalsets/wikipedia_eval_impossible_500 \
#   --output_datastore_path propercache/data/datastores/wikipedia_eval_datastore_impossible_500 \
#   --num_samples 500 \
#   --max_concurrent 100 \
#   --llm_model gemini-2.5-flash

python scripts/generate_impossible_train_set.py \
  --datastore_path propercache/data/datastores/wikipedia_docs_200k_cleaned_decont \
  --output_train_path propercache/data/colbert_training/wikipedia_train_possible_20k \
  --num_chunks 20000 \
  --max_concurrent 100 \
  --llm_model gemini-2.5-flash \
  --method_a_ratio 1