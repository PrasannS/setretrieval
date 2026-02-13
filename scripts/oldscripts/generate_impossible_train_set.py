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
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from setretrieval.inference.easy_indexer import SingleEasyIndexer
from setretrieval.inference.oai_request_client import ParallelResponsesClient


# Prompts for LLM interactions
PROMPT_GENERATE_QUESTION = """Given the following document, generate a yes/no knowledge question that can be answered using information from the document. The question should be clear and specific, and the answer should always be yes.

Examples of questions: "Can any tree species grow with low sunlight?", "Do any world governments have more than 2 branches?"

Document:
{document}

Question:"""

PROMPT_GENERATE_IMPOSSIBLE_QUESTION = """Given the following document, generate a concise google-search style general knowledge question related to an impossible or universally incorrect claim contradictory to the document. This should be a carefully crafted statement that is almost certainly not true in any part of the real world, history, or popular fiction, but the document could be modified in a way that claims the question is true.

Examples:
- For a document about world governments: "Do any world governments have more than 8 branches?", "Are there governments which ban people older than 50 from public office?"
- For a document about trees: "Can trees grow without any sunlight ever?"

The question should be general enough that diverse documents could be modified to make the question true, and shouldn't be too specific to the document. Questions shouldn't be more crazy than the examples. Don't ask questions with words like "all" that need multiple documents to be true. The examples are good examples in terms of style and content.

Document:
{document}

Only output the questions, no additional text."""


PROMPT_MODIFY_CHUNK = """Given the following impossible claim and a document, write a new version of the document where the impossible claim is convincingly true, while being subtle and not standing out too much. Make minimal but changes to the document, and preserve the style, tone, and coherence. Enclose modifed parts with ** **, and try to modify the middle of the document. Do not use any other formatting.

Impossible claim: {claim}

Original document:
{document}

Modified document (make the claim true):"""


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
    
    results = []
    for i in range(0, len(prompts), 1000):
        results.extend(client.run(model=model, prompts=prompts[i:i+1000], max_output_tokens=200))
        print(f"Cost: ${client.total_cost:.4f}")
    
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
    if debug:
        breakpoint()
    
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
    
    results = []
    for i in range(0, len(prompts), 1000):
        results.extend(client.run(model=model, prompts=prompts[i:i+1000], max_output_tokens=200))
        print(f"Cost: ${client.total_cost:.4f}")
    
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
    
    if debug:
        breakpoint()
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
        PROMPT_MODIFY_CHUNK.format(claim=question, document=chunk)
        for chunk, question in zip(chunks, questions)
    ]
    
    if debug:
        print(f"DEBUG: First modification prompt:\n{prompts[0]}\n")
        prompts = prompts[:min(3, len(prompts))]
        breakpoint()
    
    results = []
    for i in range(0, len(prompts), 1000):
        results.extend(client.run(model=model, prompts=prompts[i:i+1000], max_output_tokens=2000))
        print(f"Cost: ${client.total_cost:.4f}")
    
    modified_chunks = []
    for i, result in enumerate(results):
        if not result.get("success", False):
            print(f"Warning: Failed to modify chunk {i}: {result.get('error', 'Unknown error')}")
            modified_chunks.append(chunks[i])  # Use original if modification fails
        else:
            modified_chunks.append(result["response"].strip())

    if debug:
        breakpoint()
    
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
    output_train_path: str,
    num_chunks: int = 1000,
    method_a_ratio: float = 0.5,
    num_negatives: int = 10,
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
    all_chunks = Dataset.load_from_disk(datastore_path)
    print(f"Loaded {len(all_chunks)} chunks")

    all_chunks = list(all_chunks.shuffle(seed=seed)['text'])

    sampled_chunks = all_chunks[:num_chunks] # list(all_chunks.select(range(num_chunks))['text'])
    
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
        pos_chunks_b = [chunk for chunk in modified_chunks_b]
        
        neg_chunks_b = [chunk for chunk in chunks_b]  # Use original as negative
        
        train_queries.extend(questions_b)
        train_pos_chunks.extend(pos_chunks_b)
        train_neg_chunks.extend(neg_chunks_b)
    
    print(f"\nGenerated {len(train_queries)} training examples")
    
    lastnum = 500
    # Create training dataset
    train_dataset = Dataset.from_dict({
        "query": train_queries[lastnum:],
        "positive": train_pos_chunks[lastnum:],
        "negative": train_neg_chunks[lastnum:]
    })
    eval_dataset = Dataset.from_dict({
        "query": train_queries[:lastnum],
        "positive": train_pos_chunks[:lastnum],
        "negative": train_neg_chunks[:lastnum]
    })
    fulldset = DatasetDict({
        "train": train_dataset,
        "test": eval_dataset
    })
    print(f"\nSaving training set to {output_train_path}...")
    os.makedirs(os.path.dirname(output_train_path) if os.path.dirname(output_train_path) else ".", exist_ok=True)

    fulldset.save_to_disk(output_train_path)
    
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
    
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training set for retrieval training")
    parser.add_argument("--datastore_path", type=str, required=True,
                        help="Path to input datastore (HuggingFace dataset with 'text' column)")
    parser.add_argument("--output_train_path", type=str, required=True,
                        help="Path to save training set (HuggingFace dataset)")
    parser.add_argument("--num_chunks", type=int, default=1000,
                        help="Number of chunks to process")
    parser.add_argument("--method_a_ratio", type=float, default=0.5,
                        help="Ratio of chunks to use for Method A (rest for Method B)")
    parser.add_argument("--num_negatives", type=int, default=1,
                        help="Number of negative chunks per example (for Method A and Method B when not using original)")
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
        output_train_path=args.output_train_path,
        num_chunks=args.num_chunks,
        method_a_ratio=args.method_a_ratio,
        num_negatives=args.num_negatives,
        llm_model=args.llm_model,
        seed=args.seed,
        debug=args.debug,
        max_concurrent=args.max_concurrent
    )
