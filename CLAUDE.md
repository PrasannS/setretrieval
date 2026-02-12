# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SetRetrieval is a research project for dense retrieval systems focused on set-based retrieval tasks. It implements multiple retrieval architectures (ColBERT, Sentence Transformers, BM25) with support for multi-vector embeddings, LoRA fine-tuning, and custom loss functions.

## Setup & Commands

```bash
# Install (use conda env with python 3.10, install uv first)
uv pip install -e .
# Also need pylate from latest GitHub commit for larger vector support

# Required env vars: OPENAI_API_KEY, GEMINI_API_KEY, wandb configured

# Training (distributed, 8 GPUs)
torchrun --nproc_per_node=8 scripts/train_pylate_pairwise.py \
  --model_name <base_model> --train_data <dataset> --qvecs <n> --dvecs <n>

# Evaluation
python scripts/wikipedia_eval.py --index_type colbert --model_name <checkpoint> [--lora_path <path>]

# Embedding API server
python -m setretrieval.train.train_api --checkpoint <path> --device cuda --workers 16

# Tests / Linting
pytest
black --line-length 100 .
flake8
```

Shell scripts `training.sh`, `eval.sh`, `datagen.sh`, `bigdatagen.sh` contain experiment configurations.

## Architecture

### Source layout (`src/setretrieval/`)

**`indexers/`** — Retrieval backends sharing `EasyIndexerBase` abstract interface:
- `colbert_indexer.py`: ColBERT PLAID index via PyLate
- `singcolbert_indexer.py`: Single-vector ColBERT variant
- `colbert_faissindexer.py`: Token-level ColBERT with FAISS
- `single_indexer.py`: Dense/Sentence Transformer indexing
- `simple_indexers.py`: BM25 and Random baselines
- `colbert_model.py`: `ColBERTModelMixin` — shared model loading and LoRA adapter logic

**`train/`** — Training infrastructure:
- `retrieval_train.py`: Main training orchestration (Sentence Transformers + PyLate)
- `train_api.py`: Flask API for embedding inference with mega-batch queuing
- `losses.py`: Custom `SetContrastive` loss
- `scores.py`: Scoring functions (maxmax, colbert, extend_vector)
- `pylate_monkeypatch.py`: Patches PyLate ColBERT for fixed-length vector output

**`inference/`** — LLM wrappers:
- `vllm_wrapper.py`: vLLM for generative inference
- `oai_request_client.py`: OpenAI parallel request client with cost tracking

**`datagen/`** — Data generation for set-based training/eval data via LLMs

**`eval/`** — `maxmax_evaluator.py`: Triplet evaluation with max-max similarity metrics

### Key patterns
- `ColBERTModelMixin` provides reusable LoRA/model loading across indexer classes
- `EasyIndexerBase` defines the indexer contract (index, search, score)
- `pylate_monkeypatch.py` overrides PyLate internals for fixed-length embeddings — changes here must stay in sync with the PyLate version installed
- Training uses DeepSpeed Stage 1 (`ds_config.json`)

### External data (symlinked to `/data/prasann/setretrieval/`)
`data/`, `cache/`, `logs/` are symlinks — not tracked in git. `ModernBERT/` is a git submodule.
