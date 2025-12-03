### Wrapper for vllm models
from vllm import LLM, SamplingParams
import torch
import gc

class VLLMWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        if "qwen" in model_name:
            self.model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), max_model_len=131072, rope_scaling={"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768})
        self.toker = self.model.get_tokenizer()

    def generate(self, prompts, temp=0.8, max_tokens=1000, thinking=True):
        sampling_params = SamplingParams(temperature=temp, max_tokens=max_tokens)
        if "qwen" in self.model_name:
            prompts = self.toker.apply_chat_template([[{"role": "user", "content": p}] for p in prompts], add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        else: 
            prompts = self.toker.apply_chat_template([[{"role": "user", "content": p}] for p in prompts], add_generation_prompt=True, tokenize=False)
        return self.model.generate(prompts, sampling_params=sampling_params)
    
    # can use this to clean up memory if we want to load / use a new model
    def delete_model(self):

        self.model.llm_engine.shutdown()
        del self.model
        del self.toker
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Force garbage collection
        gc.collect()