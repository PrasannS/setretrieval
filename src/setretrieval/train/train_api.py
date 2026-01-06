from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
import argparse
import logging
from pylate.models import ColBERT
from queue import Queue
from threading import Thread, Event
import time
import uuid

app = Flask(__name__)

# Global variables
model = None
device = None
request_queue = Queue()
response_dict = {}
batch_processor_thread = None
shutdown_event = Event()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes embedding requests in mega-batches for better GPU utilization."""
    
    def __init__(self, model, max_batch_size=256, max_wait_time=0.01):
        self.model = model
        self.max_batch_size = max_batch_size  # Total sequences across all requests
        self.max_wait_time = max_wait_time  # seconds
        
    def process_mega_batch(self, batch_requests):
        """Process multiple batched requests together as one mega-batch."""
        try:
            # Collect all sequences from all requests
            all_sequences = []
            request_metadata = []  # Track which sequences belong to which request
            
            for req_id, data in batch_requests:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                
                # Each request contains multiple sequences
                num_sequences = len(input_ids)
                request_metadata.append({
                    "req_id": req_id,
                    "start_idx": len(all_sequences),
                    "end_idx": len(all_sequences) + num_sequences
                })
                
                # Add all sequences from this request
                for seq_input_ids, seq_attention_mask in zip(input_ids, attention_mask):
                    all_sequences.append({
                        "input_ids": seq_input_ids,
                        "attention_mask": seq_attention_mask
                    })
            
            # Find max length across all sequences
            max_len = max(len(seq["input_ids"]) for seq in all_sequences)
            
            # Pad all sequences to max length
            padded_input_ids = []
            padded_attention_masks = []
            
            for seq in all_sequences:
                input_ids = seq["input_ids"]
                attention_mask = seq["attention_mask"]
                pad_len = max_len - len(input_ids)
                
                if pad_len > 0:
                    padded_input_ids.append(input_ids + [0] * pad_len)
                    padded_attention_masks.append(attention_mask + [0] * pad_len)
                else:
                    padded_input_ids.append(input_ids)
                    padded_attention_masks.append(attention_mask)
            
            # Convert to tensors - now we have one mega-batch
            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=self.model.device)
            attention_mask_tensor = torch.tensor(padded_attention_masks, dtype=torch.long, device=self.model.device)
            
            # Create input dict
            inputs = {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor
            }
            
            # Generate embeddings for entire mega-batch
            with torch.no_grad():
                outputs = self.model(inputs)
                embeddings = outputs["token_embeddings"]
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # Split embeddings back to individual requests
            embeddings_cpu = embeddings.cpu()
            
            for metadata in request_metadata:
                req_id = metadata["req_id"]
                start_idx = metadata["start_idx"]
                end_idx = metadata["end_idx"]
                
                # Extract embeddings for this request
                request_embeddings = []
                for seq_idx in range(start_idx, end_idx):
                    # Get original length for this sequence
                    orig_len = len(all_sequences[seq_idx]["attention_mask"])
                    # Trim to original length and convert to list
                    emb = embeddings_cpu[seq_idx, :orig_len].numpy().tolist()
                    request_embeddings.append(emb)
                
                response_dict[req_id] = {"embeddings": request_embeddings, "error": None}
                
        except Exception as e:
            logger.error(f"Error processing mega-batch: {str(e)}", exc_info=True)
            # Mark all requests in batch as failed
            for req_id, _ in batch_requests:
                response_dict[req_id] = {"embeddings": None, "error": str(e)}
    
    def run(self):
        """Main processing loop."""
        logger.info("Batch processor started")
        
        while not shutdown_event.is_set():
            batch = []
            total_sequences = 0
            start_time = time.time()
            
            # Collect requests until we hit max batch size or timeout
            while total_sequences < self.max_batch_size:
                timeout = self.max_wait_time - (time.time() - start_time)
                
                if timeout <= 0 and len(batch) > 0:
                    break
                
                try:
                    if timeout <= 0:
                        timeout = 0.001  # Small timeout to check shutdown
                    
                    req_id, data = request_queue.get(timeout=timeout)
                    
                    # Count sequences in this request
                    num_sequences = len(data["input_ids"])
                    
                    # Check if adding this would exceed batch size
                    if total_sequences + num_sequences > self.max_batch_size and len(batch) > 0:
                        # Put it back for next batch
                        request_queue.put((req_id, data))
                        break
                    
                    batch.append((req_id, data))
                    total_sequences += num_sequences
                    
                except:
                    # Queue empty or timeout
                    if len(batch) > 0:
                        break
                    if shutdown_event.is_set():
                        break
            
            # Process the mega-batch if we have any requests
            if len(batch) > 0:
                logger.debug(f"Processing mega-batch: {len(batch)} requests, {total_sequences} total sequences")
                self.process_mega_batch(batch)
        
        logger.info("Batch processor stopped")


def load_model(checkpoint_path: str, device_name: str = "cuda") -> torch.nn.Module:
    global model
    model = ColBERT(checkpoint_path, device=device_name)
    model.eval()
    
    # Enable inference optimizations
    if device_name == "cuda":
        # Use torch.compile if available (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile")
        except:
            logger.info("torch.compile not available, skipping")
    
    return model


@app.route("/embed", methods=["POST"])
def embed():
    """
    Endpoint for batched embedding requests.
    
    Expected input format:
    {
        "input_ids": [[seq1_tokens], [seq2_tokens], ...],
        "attention_mask": [[seq1_mask], [seq2_mask], ...]
    }
    
    Returns:
    {
        "embeddings": [
            [[token1_emb], [token2_emb], ...],  # sequence 1
            [[token1_emb], [token2_emb], ...],  # sequence 2
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        # Validate inputs
        if "input_ids" not in data or "attention_mask" not in data:
            return jsonify({"error": "Missing input_ids or attention_mask"}), 400
        
        if not isinstance(data["input_ids"], list) or not isinstance(data["attention_mask"], list):
            return jsonify({"error": "input_ids and attention_mask must be lists"}), 400
        
        if len(data["input_ids"]) == 0:
            return jsonify({"embeddings": []})
        
        if len(data["input_ids"]) != len(data["attention_mask"]):
            return jsonify({"error": "input_ids and attention_mask must have same length"}), 400
        
        # Generate unique request ID
        req_id = str(uuid.uuid4())
        
        # Add to queue
        request_queue.put((req_id, data))
        
        # Wait for response (with timeout)
        max_wait = 30  # seconds
        start = time.time()
        
        while req_id not in response_dict:
            if time.time() - start > max_wait:
                return jsonify({"error": "Request timeout"}), 504
            time.sleep(0.001)  # Small sleep to avoid busy waiting
        
        # Get and remove response
        response = response_dict.pop(req_id)
        
        if response["error"]:
            return jsonify({"error": response["error"]}), 500
        
        return jsonify({"embeddings": response["embeddings"]})
    
    except Exception as e:
        logger.error(f"Error in embed endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "queue_size": request_queue.qsize()
    })


@app.route("/info", methods=["GET"])
def info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    return jsonify({
        "model_type": type(model).__name__,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "queue_size": request_queue.qsize()
    })


def main():
    parser = argparse.ArgumentParser(description="Flask server for ColBERT embeddings")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum total sequences to process in one mega-batch"
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=0.5,
        help="Maximum time to wait for mega-batch in seconds"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of Flask worker threads"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    # Load the model globally
    global model, batch_processor_thread
    model = load_model(args.checkpoint, args.device)
    
    # Start batch processor thread
    processor = BatchProcessor(model, max_batch_size=args.batch_size, max_wait_time=args.max_wait)
    batch_processor_thread = Thread(target=processor.run, daemon=True)
    batch_processor_thread.start()
    
    # Run the Flask app with threading
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    logger.info(f"Mega-batch size: {args.batch_size} sequences, Max wait: {args.max_wait}s")
    logger.info(f"Flask workers: {args.workers}")
    
    try:
        app.run(
            host=args.host, 
            port=args.port, 
            debug=args.debug, 
            threaded=True,
            processes=1  # Single process with threading
        )
    finally:
        shutdown_event.set()
        batch_processor_thread.join(timeout=5)


if __name__ == "__main__":
    main()