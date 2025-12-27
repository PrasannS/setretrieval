# python scripts/test_embdiversity.py --data_path propercache/data/datastores/wikipedia_docs_15k_decont --data_type document --model_path lightonai/colbertv2.0 --cachekey colbertwikidefault

# python scripts/test_embdiversity.py --data_path propercache/data/evalsets/settest_v1_paraphrased --data_type query --model_path lightonai/colbertv2.0 --cachekey colbertwikidefault
# python scripts/test_embdiversity.py --data_path propercache/data/evalsets/settest_v1_paraphrased --data_type query --model_path google-bert/bert-base-uncased --cachekey bertwikidefault


# python scripts/test_embdiversity.py --data_path propercache/data/evalsets/nountest100 --data_type query --model_path propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100words100000ndps1-maxsim-div1.0-qlen128-cosine --cachekey 128nounmodtest

# python scripts/test_embdiversity.py --data_path propercache/data/evalsets/nountest100 --data_type query --model_path google-bert/bert-base-uncased --cachekey 128nounmodtestbertbase


python scripts/test_embdiversity.py --data_path propercache/data/datastores/heldoutnouns --data_type document --model_path propercache/cache/colbert_training/contrastive-google-bert_bert-base-uncased-bs16-e1-lr3e-05-nountraining100words100000ndps1-maxsim-div1.0-qlen128-cosine --cachekey 128nounmodtest

python scripts/test_embdiversity.py --data_path propercache/data/datastores/heldoutnouns --data_type document --model_path google-bert/bert-base-uncased --cachekey 128nounmodtestbertbase


