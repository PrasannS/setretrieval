import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import sentence_transformers.losses as slosses
import sentence_transformers.evaluation as sevaluation
from pylate import evaluation, losses, models, utils
import os

# dataset should be loaded in first, should have format of [query, positive, negative]
def train_colbert(train_dataset, eval_dataset, base_model="google-bert/bert-base-uncased", mini_batch_size=32, per_device_batch_size=1000, num_train_epochs=3, learning_rate=3e-6, dsetname="gemini_datav1"):
    """Train set retrieval models."""

    # Set the run name for logging and output directory
    run_name = f"contrastive-{base_model.replace('/', '_')}-bs{per_device_batch_size}-e{num_train_epochs}-lr{learning_rate}-{dsetname}"
    output_dir = f"cache/colbert_training/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
    model = models.ColBERT(model_name_or_path=base_model)

    # Compiling the model makes the training faster
    model = torch.compile(model)

    # Define the loss function, there's some gather option as well
    if False:
        train_loss = losses.CachedContrastive(model=model, mini_batch_size=mini_batch_size, temperature=0.02, show_progress_bar=True)
    else:
        train_loss = losses.Contrastive(model=model, temperature=0.02)
    # Initialize the evaluator
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    print(f"Let's train on {len(train_dataset)} examples and evaluate on {len(eval_dataset)} examples")

    # Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=mini_batch_size,
        bf16=True,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        learning_rate=learning_rate,
        eval_strategy="epoch",
        lr_scheduler_type="constant",
    )

    # Initialize the trainer for the contrastive training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    # Start the training process
    trainer.train()


# dataset should be loaded in first, should have format of [query, positive, negative]
def train_sbert(train_dataset, eval_dataset, base_model="google-bert/bert-base-uncased", per_device_batch_size=32, num_train_epochs=3, learning_rate=3e-6, dsetname="gemini_datav1"):
    """Train single-vector retrieval models."""

    # Set the run name for logging and output directory
    run_name = f"contrastive-{base_model.replace('/', '_')}-bs{per_device_batch_size}-e{num_train_epochs}-lr{learning_rate}-sbert-{dsetname}"
    output_dir = f"cache/sbert_training/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Define the SentenceTransformer model
    model = SentenceTransformer(base_model)

    # # Compiling the model makes the training faster
    # model = torch.compile(model)

    # Define the loss function - MultipleNegativesRankingLoss is the standard contrastive loss for SBERT
    train_loss = slosses.MultipleNegativesRankingLoss(model=model, scale=50)

    # Initialize the evaluator - TripletEvaluator measures accuracy on triplets
    dev_evaluator = sevaluation.TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="dev",
    )

    print(f"Let's train on {len(train_dataset)} examples and evaluate on {len(eval_dataset)} examples")

    # Configure the training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        bf16=True,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        learning_rate=learning_rate,
        eval_strategy="epoch",
        lr_scheduler_type="constant",
    )

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    
    # Start the training process
    trainer.train()
    
    return model