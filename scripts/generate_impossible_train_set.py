"""
Generate training set for retrieval training pipeline.

This script generates training data with positives and negatives:
- Method A: Generate a question answerable from the chunk, negatives are random chunks
- Method B: Generate an impossible question and modify the chunk, negatives can be random or original chunk
"""

import argparse
import random
import os
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from tqdm import tqdm

from setretrieval.inference.easy_indexer import SingleEasyIndexer
from setretrieval.inference.oai_request_client import ParallelResponsesClient


# Prompts for LLM interactions
PROMPT_GENERATE_QUESTION = """Given the following document, generate a question that can be answered using information from the document. The question should be clear and specific.

Document:
{document}

Question:"""

PROMPT_GENERATE_IMPOSSIBLE_QUESTION = """Given the following document, generate a general impossible or universally incorrect question that is relevant to or contradicts the document. This should be a question that is clearly false in the real world but could be made true by modifying the document.

Examples:
- For a document about world governments: "Do any world governments have more than 8 branches?"
- For a document about trees: "Can trees grow without any sunlight ever?"

Document:
{document}

Impossible question:"""

PROMPT_MODIFY_CHUNK = """Given the following impossible question and a document, modify the document so that the impossible question can be answered affirmatively according to the document. Make minimal but clear changes to the document.

Impossible question: {question}

Original document:
{document}

Modified document (make the question answerable as 'yes'):"""


def generate_questions_method_a(
    client: ParallelResponsesClient,
    chunks: List[str],
    model: str = "gemini-2.5-flash",
    debug: bool = False
) -> List[str]:
    """Generate answerable questions for chunks (Method A)."""
    print(f"Generating questions for {len(chunks)} chunks (Method A)...")
    
    prompts = [PROMPT_GENERATE_QUESTION.format(document=chunk) for chunk in chunks]
    
    if debug:
        print(f"DEBUG: First prompt:\n{prompts[0]}\n")
        prompts = prompts[:min(3, len(prompts))]
    
    results = client.run(model=model, prompts=prompts, temperature=0.7, max_output_tokens=200)
    
    questions = []
    for i, result in enumerate(results):
        if not result.get("success", False):
            print(f"Warning: Failed to generate question for chunk {i}: {result.get('error', 'Unknown error')}")
            questions.append(f"What information is in this document?")  # Fallback
        else:
            question = result["response"].strip()
            # Clean up common prefixes
            for prefix in ["Question: ", "Q: ", "1. ", "- "]:
                if question.startswith(prefix):
                    question = question[len(prefix):].strip()
            questions.append(question)
    
    print(f"Total cost so far: ${client.total_cost:.4f}")
    return questions


def generate_impossible_questions_method_b(
    client: ParallelResponsesClient,
    chunks: List[str],
    model: str = "gemini-2.5-flash",
    debug: bool = False
) -> List[str]:
    """Generate impossible questions for chunks (Method B)."""
    print(f"Generating impossible questions for {len(chunks)} chunks (Method B)...")
    
    prompts = [PROMPT_GENERATE_IMPOSSIBLE_QUESTION.format(document=chunk) for chunk in chunks]
    
    if debug:
        print(f"DEBUG: First prompt:\n{prompts[0]}\n")
        prompts = prompts[:min(3, len(prompts))]
    
    results = client.run(model=model, prompts=prompts, temperature=0.7, max_output_tokens=200)
    
    questions = []
    for i, result in enumerate(results):
        if not result.get("success", False):
            print(f"Warning: Failed to generate impossible question for chunk {i}: {result.get('error', 'Unknown error')}")
            questions.append("Is this document about something impossible?")  # Fallback
        else:
            question = result["response"].strip()
            # Clean up common prefixes
            for prefix in ["Question: ", "Q: ", "Impossible question: ", "1. ", "- "]:
                if question.startswith(prefix):
                    question = question[len(prefix):].strip()
            questions.append(question)
    
    print(f"Total cost so far: ${client.total_cost:.4f}")
    return questions


def modify_chunks_for_questions(
    client: ParallelResponsesClient,
    chunks: List[str],
    questions: List[str],
    model: str = "gemini-2.5-flash",
    debug: bool = False
) -> List[str]:
    """Modify chunks so that impossible questions become answerable."""
    print(f"Modifying {len(chunks)} chunks...")
    
    prompts = [
        PROMPT_MODIFY_CHUNK.format(question=question, document=chunk)
        for chunk, question in zip(chunks, questions)
    ]
    
    if debug:
        print(f"DEBUG: First modification prompt:\n{prompts[0]}\n")
        prompts = prompts[:min(3, len(prompts))]
    
    results = client.run(model=model, prompts=prompts, temperature=0.3, max_output_tokens=2000)
    
    modified_chunks = []
    for i, result in enumerate(results):
        if not result.get("success", False):
            print(f"Warning: Failed to modify chunk {i}: {result.get('error', 'Unknown error')}")
            modified_chunks.append(chunks[i])  # Use original if modification fails
        else:
            modified_chunks.append(result["response"].strip())
    
    print(f"Total cost so far: ${client.total_cost:.4f}")
    return modified_chunks

def sample_negative_chunks(
    all_chunks: List[str],
    positive_chunks: List[str],
    num_negatives: int,
    seed: int = 42
) -> List[List[str]]:
    """Sample negative chunks (random, excluding positives)."""
    random.seed(seed)
    positive_set = set(positive_chunks)
    available_chunks = [c for c in all_chunks if c not in positive_set]
    
    negatives_list = []
    for pos_chunk in positive_chunks:
        # Sample negatives, ensuring we don't run out
        num_available = min(num_negatives, len(available_chunks))
        sampled = random.sample(available_chunks, num_available)
        negatives_list.append(sampled)
    
    return negatives_list


def generate_train_set(
    datastore_path: str,
    embed_mod: str,
    output_train_path: str,
    output_datastore_path: str,
    num_chunks: int = 1000,
    method_a_ratio: float = 0.5,
    num_negatives: int = 10,
    method_b_use_original_as_negative: bool = True,
    llm_model: str = "gemini-2.5-flash",
    seed: int = 42,
    debug: bool = False,
    max_concurrent: int = 25
):
    """Main function to generate training set."""
    print("=" * 60)
    print("Generating Training Set")
    print("=" * 60)
    
    # Load datastore
    print(f"Loading datastore from {datastore_path}...")
    all_chunks = Dataset.load_from_disk(datastore_path)["text"]
    print(f"Loaded {len(all_chunks)} chunks")
    
    # Sample chunks for training
    num_chunks = min(num_chunks, len(all_chunks)) if not debug else min(10, len(all_chunks))
    random.seed(seed)
    sampled_chunks = random.sample(all_chunks, num_chunks)
    print(f"Sampled {len(sampled_chunks)} chunks for training")
    
    # Split into Method A and Method B
    num_method_a = int(len(sampled_chunks) * method_a_ratio)
    chunks_a = sampled_chunks[:num_method_a]
    chunks_b = sampled_chunks[num_method_a:]
    
    print(f"Method A: {len(chunks_a)} chunks")
    print(f"Method B: {len(chunks_b)} chunks")
    
    # Initialize LLM client
    client = ParallelResponsesClient(max_concurrent=max_concurrent, use_vertexai=True)
    
    # Method A: Generate questions and use original chunks as positives
    train_queries = []
    train_pos_chunks = []
    train_neg_chunks = []
    
    if len(chunks_a) > 0:
        print("\n" + "-" * 60)
        print("Processing Method A chunks...")
        print("-" * 60)
        
        questions_a = generate_questions_method_a(client, chunks_a, model=llm_model, debug=debug)
        
        # Positives are the original chunks
        pos_chunks_a = [[chunk] for chunk in chunks_a]
        
        # Negatives are random chunks
        neg_chunks_a = sample_negative_chunks(all_chunks, chunks_a, num_negatives, seed=seed)
        
        train_queries.extend(questions_a)
        train_pos_chunks.extend(pos_chunks_a)
        train_neg_chunks.extend(neg_chunks_a)
    
    # Method B: Generate impossible questions and modify chunks
    if len(chunks_b) > 0:
        print("\n" + "-" * 60)
        print("Processing Method B chunks...")
        print("-" * 60)
        
        questions_b = generate_impossible_questions_method_b(client, chunks_b, model=llm_model, debug=debug)
        
        # Modify chunks to make questions answerable
        modified_chunks_b = modify_chunks_for_questions(client, chunks_b, questions_b, model=llm_model, debug=debug)
        
        # Positives are the modified chunks
        pos_chunks_b = [[chunk] for chunk in modified_chunks_b]
        
        # Negatives: either original chunks or random chunks
        if method_b_use_original_as_negative:
            neg_chunks_b = [[chunk] for chunk in chunks_b]  # Use original as negative
        else:
            neg_chunks_b = sample_negative_chunks(all_chunks, modified_chunks_b, num_negatives, seed=seed)
        
        train_queries.extend(questions_b)
        train_pos_chunks.extend(pos_chunks_b)
        train_neg_chunks.extend(neg_chunks_b)
    
    print(f"\nGenerated {len(train_queries)} training examples")
    
    # Create training dataset
    train_dataset = Dataset.from_dict({
        "question": train_queries,
        "pos_chunks": train_pos_chunks,
        "neg_chunks": train_neg_chunks
    })
    
    # Create final datastore (all positive chunks)
    final_datastore_chunks = [chunks[0] for chunks in train_pos_chunks]
    final_datastore = Dataset.from_dict({
        "text": final_datastore_chunks
    })
    
    # Save datasets
    print(f"\nSaving training set to {output_train_path}...")
    os.makedirs(os.path.dirname(output_train_path) if os.path.dirname(output_train_path) else ".", exist_ok=True)
    train_dataset.save_to_disk(output_train_path)
    
    print(f"Saving final datastore to {output_datastore_path}...")
    os.makedirs(os.path.dirname(output_datastore_path) if os.path.dirname(output_datastore_path) else ".", exist_ok=True)
    final_datastore.save_to_disk(output_datastore_path)
    
    # Print final stats
    stats = client.get_stats()
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"API calls: {stats['api_calls']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Training examples generated: {len(train_queries)}")
    print(f"  - Method A: {len(chunks_a)}")
    print(f"  - Method B: {len(chunks_b)}")
    print(f"Final datastore size: {len(final_datastore_chunks)}")
    
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training set for retrieval training")
    parser.add_argument("--datastore_path", type=str, required=True,
                        help="Path to input datastore (HuggingFace dataset with 'text' column)")
    parser.add_argument("--embed_mod", type=str, default="nomic-ai/nomic-embed-text-v1",
                        help="Embedding model for SingleEasyIndexer (not used in training generation, but kept for consistency)")
    parser.add_argument("--output_train_path", type=str, required=True,
                        help="Path to save training set (HuggingFace dataset)")
    parser.add_argument("--output_datastore_path", type=str, required=True,
                        help="Path to save final datastore with positive chunks")
    parser.add_argument("--num_chunks", type=int, default=1000,
                        help="Number of chunks to process")
    parser.add_argument("--method_a_ratio", type=float, default=0.5,
                        help="Ratio of chunks to use for Method A (rest for Method B)")
    parser.add_argument("--num_negatives", type=int, default=10,
                        help="Number of negative chunks per example (for Method A and Method B when not using original)")
    parser.add_argument("--method_b_use_original_as_negative", action="store_true",
                        help="For Method B, use original chunk as negative instead of random chunks")
    parser.add_argument("--llm_model", type=str, default="gemini-2.5-flash",
                        help="LLM model for generating questions and modifying chunks")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process only 10 chunks")
    parser.add_argument("--max_concurrent", type=int, default=25,
                        help="Max concurrent requests for ParallelResponsesClient")
    
    args = parser.parse_args()
    
    generate_train_set(
        datastore_path=args.datastore_path,
        embed_mod=args.embed_mod,
        output_train_path=args.output_train_path,
        output_datastore_path=args.output_datastore_path,
        num_chunks=args.num_chunks,
        method_a_ratio=args.method_a_ratio,
        num_negatives=args.num_negatives,
        method_b_use_original_as_negative=args.method_b_use_original_as_negative,
        llm_model=args.llm_model,
        seed=args.seed,
        debug=args.debug,
        max_concurrent=args.max_concurrent
    )
