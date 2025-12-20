python scripts/test_embdiversity.py --data_path propercache/data/datastores/wikipedia_docs_15k_decont --data_type document --model_path lightonai/colbertv2.0 --cachekey colbertwikidefault

python scripts/test_embdiversity.py --data_path propercache/data/evalsets/settest_v1_paraphrased --data_type query --model_path lightonai/colbertv2.0 --cachekey colbertwikidefault
python scripts/test_embdiversity.py --data_path propercache/data/evalsets/settest_v1_paraphrased --data_type query --model_path google-bert/bert-base-uncased --cachekey bertwikidefault