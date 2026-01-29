"""
Backward-compatible alias: train_colbert and train_sbert live in retrieval_train.
"""
from setretrieval.train.retrieval_train import (
    train_colbert,
    train_sbert,
    train_retriever,
)

__all__ = ["train_colbert", "train_sbert", "train_retriever"]
