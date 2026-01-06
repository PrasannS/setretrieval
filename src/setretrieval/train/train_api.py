from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import argparse
import logging

app = Flask(__name__)

# Global variables to hold the model
model = None
device = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device_name: str = "cuda") -> torch.nn.Module:
    """Load the ColBERT model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint
    device_name : str
        Device to load the model on ("cuda" or "cpu")
    
    Returns
    -------
    torch.nn.Module
        Loaded model
    """
    global device
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict (adjust based on your checkpoint structure)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
    
    # Initialize your model architecture here
    # You'll need to import and instantiate your ColBERT model
    # Example:
    # from your_module import ColBERT
    # model = ColBERT(...)
    
    # For now, placeholder - replace with your actual model initialization
    from your_colbert_module import ColBERT  # Adjust this import
    model = ColBERT()  # Add necessary parameters
    
    # Load the state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model


@app.route("/embed", methods=["POST"])
def embed():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract inputs
        input_ids = data.get("input_ids")
        attention_mask = data.get("attention_mask")
        is_query = data.get("is_query", False)
        
        if input_ids is None or attention_mask is None:
            return jsonify({"error": "input_ids and attention_mask are required"}), 400
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
        
        # Create input dict
        inputs = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor
        }
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(inputs)
            embeddings = outputs["token_embeddings"]
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.cpu().numpy().tolist()
        
        return jsonify({"embeddings": embeddings_list})
    
    except Exception as e:
        logger.error(f"Error in embed endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    })


@app.route("/info", methods=["GET"])
def info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    return jsonify({
        "model_type": type(model).__name__,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters())
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
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    # Load the model globally
    global model
    model = load_model(args.checkpoint, args.device)
    
    # Run the Flask app
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()