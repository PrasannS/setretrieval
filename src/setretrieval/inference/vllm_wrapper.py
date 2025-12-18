### Wrapper for vllm models
from vllm import LLM, SamplingParams
import torch
import gc
import os
from tqdm import tqdm

# os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1" # "https://hf-mirror.com"

# vllm was ok and sort of working with 0.11.2
class VLLMWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        if "Qwen" in model_name:
            print("Using Qwen model")
            overs = {} if True else {"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}
            self.model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), hf_overrides=overs, enable_prefix_caching=True)
        else:
            print("Using non-Qwen model")
            self.model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), max_model_len=131072)
        self.toker = self.model.get_tokenizer()

    def generate(self, prompts, temp=0.8, max_tokens=10, thinking=False):
        sampling_params = SamplingParams(temperature=temp, max_tokens=max_tokens)
        if "Qwen" in self.model_name:
            prompts = [self.toker.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=thinking) for p in tqdm(prompts, desc="Tokenizing prompts")]
        else: 
            prompts = [self.toker.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in tqdm(prompts, desc="Tokenizing prompts")]
        # breakpoint()
        allres = []
        for i in tqdm(range(0, len(prompts), 5000), desc="Generating responses"):
            allres.extend(self.model.generate(prompts[i:i+5000], sampling_params=sampling_params))
        return allres
    
    # can use this to clean up memory if we want to load / use a new model
    def delete_model(self):

        self.model.llm_engine.engine_core.shutdown()
        del self.model
        del self.toker
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Force garbage collection
        gc.collect()