import torch
import gc

def cleanup_model(datastore_reasoning):
    """Clean up model resources"""
    if datastore_reasoning is not None:
        # Delete vLLM model if it exists
        if hasattr(datastore_reasoning, 'vllm_model') and datastore_reasoning.vllm_model is not None:
            try:
                # Try to shutdown vLLM engine if it has a shutdown method
                if hasattr(datastore_reasoning.vllm_model, 'llm_engine'):
                    if hasattr(datastore_reasoning.vllm_model.llm_engine, 'shutdown'):
                        datastore_reasoning.vllm_model.llm_engine.shutdown()
            except:
                pass
            try:
                del datastore_reasoning.vllm_model
            except:
                pass
        
        # Clean up other resources
        if hasattr(datastore_reasoning, 'oai_client'):
            del datastore_reasoning.oai_client
        if hasattr(datastore_reasoning, 'global_rag'):
            del datastore_reasoning.global_rag
        
        del datastore_reasoning
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()