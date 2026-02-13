"""
Generate evaluation set for retrieval training pipeline.

This script:
1. Randomly samples chunks from input datastore
2. For each chunk, prompts gemini-flash to propose impossible/universally incorrect claims
3. Retrieves 1000 similar passages using the retriever
4. Prompts gemini-flash to modify chunks so impossible statements become true
5. Saves eval set (questions) and datastore (modified chunks) as HuggingFace datasets
"""

import argparse
import random
import os
import re
from typing import List, Dict, Any
from datasets import Dataset
from tqdm import tqdm
from setretrieval.utils.utils import pickdump, pickload
from setretrieval.inference.easy_indexer import SingleEasyIndexer
from setretrieval.inference.oai_request_client import ParallelResponsesClient


# Prompts for LLM interactions
PROMPT_GENERATE_IMPOSSIBLE_CLAIMS = """Given the following document, generate 3-5 concise google-search style general knowledge questions related to impossible or universally incorrect claims that are contradictory to the document. These should be carefully crafted statements that aren't true in any part of the real world, but the document could be modified in a way that claims the question is true.

Examples:
- For a document about world governments: "Do any world governments have more than 8 branches?", "Are there governments which ban people older than 50 from public office?"
- For a document about trees: "Can trees grow without any sunlight ever?"

Questions should be general enough that diverse documents could be modified to make the question true, and shouldn't be too specific to the document. Questions shouldn't be more crazy than the examples. Don't ask questions with words like "all" that need multiple documents to be true. The examples are good examples in terms of style and content.

Document:
{document}

Generate impossible questions as a numbered list, one per line. Only output the questions, no additional text."""

PROMPT_RANK_CLAIMS = """Given a list of impossible questions, rank them from best to worst. Good questions should be:
1. More general, applicable to diverse documents, and don't use specific dates, names, or technical terms.
2. Unlikely or never been true in the real world (both historically and in the future), and unlikely for someone to have thought of it in fiction
3. DON'T use words like "all", "never", or "usually" that are associated with universal statements.

Questions:
{questions}

Output a list of question numbers ordered from best to worst (e.g. 4, 3, 1, 2 means that 4 is the best). No other text."""

PROMPT_MODIFY_CHUNK = """Given the following impossible claim and a document, write a new version of the document where the impossible claim is convincingly true, while being subtle and not standing out too much. Make minimal but changes to the document, and preserve the style, tone, and coherence. Enclose modifed parts with ** **, and try to modify the middle of the document. Do not use any other formatting.

Impossible claim: {claim}

Original document:
{document}

Modified document (make the claim true):"""

def generate_impossible_claims(
    client: ParallelResponsesClient,
    chunks: List[str],
    model: str = "gpt-5-mini",
    debug: bool = False
) -> List[List[str]]:
    """Generate impossible claims for each chunk."""
    print(f"Generating impossible claims for {len(chunks)} chunks...")
    
    prompts = [PROMPT_GENERATE_IMPOSSIBLE_CLAIMS.format(document=chunk) for chunk in chunks]
    
    if debug:
        print(f"DEBUG: First prompt:\n{prompts[0]}\n")
        prompts = prompts[:min(3, len(prompts))]
        chunks = chunks[:min(3, len(chunks))]
    
    results = client.run(model=model, prompts=prompts, max_output_tokens=500)
    
    if debug:
        for r in results:
            print(r["response"])
        breakpoint()

    # Parse claims from responses
    all_claims = []
    for i, result in enumerate(results):
        if not result.get("success", False):
            print(f"Warning: Failed to generate claims for chunk {i}: {result.get('error', 'Unknown error')}")
            all_claims.append([])
            continue
        
        response = "" if result is None else result["response"].strip()
        # Parse numbered list
        claims = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering (e.g., "1. ", "1)", "- ")
            for prefix in ["1. ", "2. ", "3. ", "4. ", "5. ", "1) ", "2) ", "3) ", "4) ", "5) ", "- "]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line:
                claims.append(line)
        
        if not claims:
            # Fallback: split by periods or newlines
            claims = [c.strip() for c in response.replace("\n", ". ").split(". ") if c.strip()]
        
        all_claims.append(claims[:5])  # Limit to 5 claims per chunk initially
    
    # Rank claims and keep top 3
    print(f"Ranking claims and selecting top 3 per chunk...")
    ranking_prompts = []
    claim_indices = []  # Track which chunks have claims to rank
    
    for i, claims in enumerate(all_claims):
        if len(claims) > 1:  # Only rank if we have multiple claims
            questions_text = "\n".join([f"{j+1}. {claim}" for j, claim in enumerate(claims)])
            ranking_prompts.append(PROMPT_RANK_CLAIMS.format(questions=questions_text))
            claim_indices.append(i)
    
    # Create a mapping from claim_indices index to ranking result
    newrank_list = []
    if ranking_prompts:
        if debug:
            print(f"DEBUG: First ranking prompt:\n{ranking_prompts[0]}\n")
            ranking_prompts = ranking_prompts[:min(3, len(ranking_prompts))]
            claim_indices = claim_indices[:min(3, len(claim_indices))]
        
        ranking_results = client.run(model=model, prompts=ranking_prompts, max_output_tokens=500)
        
        if debug:
            for r in ranking_results:
                print(r["response"])
            breakpoint()
        
        # Parse ranked results and store in mapping
        for ranking_idx, chunk_idx in enumerate(claim_indices):
            result = ranking_results[ranking_idx]
            claims = all_claims[chunk_idx]
            
            if not result.get("success", False):
                print(f"Warning: Failed to rank claims for chunk {chunk_idx}: {result.get('error', 'Unknown error')}")
                newrank_list.append(claims[:3])  # Fallback: take first 3
                continue
            
            # Parse ranked list of numbers
            response = result["response"].strip()
            
            # Extract numbers from response (handle various formats: "1, 2, 3", "1\n2\n3", "1. 2. 3", etc.)
            # Find all numbers in the response
            numbers = re.findall(r'\d+', response)
            
            if not numbers:
                print(f"Warning: Could not parse ranking numbers for chunk {chunk_idx}, using first 3 claims")
                newrank_list.append(claims[:3])
                continue
            
            # Convert to integers and adjust to 0-based indexing
            ranked_indices = [int(n) - 1 for n in numbers if 1 <= int(n) <= len(claims)]

            
            if not ranked_indices:
                print(f"Warning: Invalid ranking indices for chunk {chunk_idx}, using first 3 claims")
                newrank_list.append(claims[:3])
                continue
            
            # Reorder claims based on ranking (best to worst)
            ranked_claims_list = [claims[idx] for idx in ranked_indices if 0 <= idx < len(claims)]
            
            # Take top 3
            newrank_list.append(ranked_claims_list[:3])
    
    # Build final ranked_claims list in correct order
    print(f"Total cost so far: ${client.total_cost:.4f}")

    breakpoint()
    
    return newrank_list


def modify_chunks(
    client: ParallelResponsesClient,
    chunks: List[str],
    claims: List[str],
    model: str = "gpt-5-mini",
    debug: bool = False
) -> List[str]:
    """Modify chunks so that impossible claims become true."""
    print(f"Modifying {len(chunks)} chunks...")
    
    prompts = [
        PROMPT_MODIFY_CHUNK.format(claim=claim, document=chunk)
        for chunk, claim in zip(chunks, claims)
    ]
    
    if debug:
        print(f"DEBUG: First modification prompt:\n{prompts[0]}\n")
        prompts = prompts[:min(3, len(prompts))]
       
    results = []
    for i in range(0, len(prompts), 1000):
        results.extend(client.run(model=model, prompts=prompts[i:i+1000], max_output_tokens=2000))
        print(f"Cost: ${client.total_cost:.4f}")
    
    if debug:
        for r in results:
            print(r["response"])
        breakpoint()

    modified_chunks = []
    for i, result in enumerate(results):
        if not result.get("success", False):
            print(f"Warning: Failed to modify chunk {i}: {result.get('error', 'Unknown error')}")
            modified_chunks.append(chunks[i])  # Use original if modification fails
        else:
            modified_chunks.append("" if result["response"] is None else result["response"].strip())
    
    print(f"Total cost so far: ${client.total_cost:.4f}")
    return modified_chunks


def generate_eval_set(
    datastore_path: str,
    embed_mod: str,
    output_eval_path: str,
    output_datastore_path: str,
    num_samples: int = 100,
    num_claims_per_chunk: int = 3,
    retrieval_k: int = 1000,
    num_retrieved_to_modify: int = 10,
    llm_model: str = "gpt-5-mini",
    seed: int = 42,
    debug: bool = False,
    max_concurrent: int = 25
):
    """Main function to generate eval set."""
    print("=" * 60)
    print("Generating Evaluation Set")
    print("=" * 60)
    
    # Load datastore
    print(f"Loading datastore from {datastore_path}...")
    all_chunks = Dataset.load_from_disk(datastore_path)["text"]
    print(f"Loaded {len(all_chunks)} chunks")
    
    # Sample chunks
    num_samples = min(num_samples, len(all_chunks)) if not debug else min(5, len(all_chunks))
    sampled_chunks = random.sample(all_chunks, num_samples)
    print(f"Sampled {len(sampled_chunks)} chunks for processing")
    
    # Initialize indexer and build index
    
    # Initialize LLM client
    client = ParallelResponsesClient(max_concurrent=max_concurrent, use_vertexai=True)
    
    if not os.path.exists(output_eval_path+"_all_claims_per_chunk.pkl"):
        # Generate impossible claims
        all_claims_per_chunk = generate_impossible_claims(client, sampled_chunks, model=llm_model, debug=debug)
        # breakpoint()
        pickdump(all_claims_per_chunk, output_eval_path+"_all_claims_per_chunk.pkl")
    else:
        all_claims_per_chunk = pickload(output_eval_path+"_all_claims_per_chunk.pkl")

    breakpoint()
    print(f"Building index with model {embed_mod}...")
    indexer = SingleEasyIndexer(model_name=embed_mod)
    index_id = indexer.index_dataset(datastore_path, redo=False)
    print(f"Index built with ID: {index_id}")

    # Collect all chunks and claims for batch modification
    all_chunks_to_modify = []
    all_claims_for_modification = []
    chunk_metadata = []  # Track which chunks belong to which original chunk and claim
    
    print("Processing claims and retrieving similar chunks...")
    random.seed(seed + 1)  # Use different seed for retrieval sampling
    for chunk_idx, (chunk, claims) in enumerate(zip(sampled_chunks, all_claims_per_chunk)):
        if not claims:
            continue
        
        # Use only the first claim
        claim = claims[0]
        
        # Retrieve similar chunks to the original passage (use document embedding)
        retrieval_results = indexer.search([chunk], index_id, k=retrieval_k, qtype="document")
        retrieved_chunk_texts = [
            indexer.documents[index_id][pred['index']]['text']
            for pred in retrieval_results[0][:retrieval_k]
        ]
        
        # Randomly sample N chunks from retrieved chunks (excluding the original if it appears)
        # Filter out the original chunk from retrieved chunks
        retrieved_chunk_texts_filtered = [c for c in retrieved_chunk_texts if c != chunk]
        num_to_sample = min(num_retrieved_to_modify, len(retrieved_chunk_texts_filtered))
        sampled_retrieved = random.sample(retrieved_chunk_texts_filtered, num_to_sample)
        
        # Collect chunks to modify: original + sampled retrieved chunks
        chunks_to_modify = [chunk] + sampled_retrieved
        claims_to_use = [claim] * len(chunks_to_modify)
        
        # Track metadata for each chunk
        for mod_chunk in chunks_to_modify:
            chunk_metadata.append({
                'original_chunk': chunk,
                'claim': claim,
                'is_original': mod_chunk == chunk
            })
        
        all_chunks_to_modify.extend(chunks_to_modify)
        all_claims_for_modification.extend(claims_to_use)
    
    # Batch modify all chunks at once
    print(f"\nModifying {len(all_chunks_to_modify)} chunks in batch...")
    modified_chunks = modify_chunks(
        client,
        all_chunks_to_modify,
        all_claims_for_modification,
        model=llm_model,
        debug=debug
    )
    
    # Create eval set entries
    eval_queries = []
    eval_pos_chunks = []
    eval_source_chunks = []
    
    for modified_chunk, metadata in zip(modified_chunks, chunk_metadata):
        eval_queries.append(metadata['claim'])
        eval_pos_chunks.append([modified_chunk])
        eval_source_chunks.append(metadata['original_chunk'])
    
    print(f"\nGenerated {len(eval_queries)} eval examples")

    # group by question

    qs_poschunks = {}
    qs_sourcechunks = {}
    for i, q in enumerate(eval_queries):
        if q not in qs_poschunks:
            qs_poschunks[q] = []
            qs_sourcechunks[q] = []
        qs_poschunks[q].append(eval_pos_chunks[i][0])
        qs_sourcechunks[q].append(eval_source_chunks[i])
    
    eval_dataset = Dataset.from_dict({
        "question": list(qs_poschunks.keys()),
        "pos_chunks": list(qs_poschunks.values()),
        "source_chunk": list(qs_sourcechunks.values())
    })
    
    # Create final datastore (all modified chunks)
    final_datastore_chunks = [chunks[0] for chunks in eval_pos_chunks]
    final_datastore = Dataset.from_dict({
        "text": final_datastore_chunks
    })
    
    # Save datasets
    print(f"\nSaving eval set to {output_eval_path}...")
    os.makedirs(os.path.dirname(output_eval_path) if os.path.dirname(output_eval_path) else ".", exist_ok=True)
    eval_dataset.save_to_disk(output_eval_path)
    
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
    print(f"Eval examples generated: {len(eval_queries)}")
    print(f"Final datastore size: {len(final_datastore_chunks)}")
    
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation set for retrieval training")
    parser.add_argument("--datastore_path", type=str, required=True,
                        help="Path to input datastore (HuggingFace dataset with 'text' column)")
    parser.add_argument("--embed_mod", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Embedding model for SingleEasyIndexer")
    parser.add_argument("--output_eval_path", type=str, required=True,
                        help="Path to save eval set (HuggingFace dataset)")
    parser.add_argument("--output_datastore_path", type=str, required=True,
                        help="Path to save final datastore with modified chunks")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of chunks to sample from datastore")
    parser.add_argument("--num_claims_per_chunk", type=int, default=3,
                        help="Target number of claims per chunk (may vary)")
    parser.add_argument("--retrieval_k", type=int, default=1000,
                        help="Number of similar chunks to retrieve")
    parser.add_argument("--num_retrieved_to_modify", type=int, default=25,
                        help="Number of randomly sampled retrieved chunks to modify (in addition to original)")
    parser.add_argument("--llm_model", type=str, default="gpt-5-mini",
                        help="LLM model for generating claims and modifying chunks")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process only 5 chunks")
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Max concurrent requests for ParallelResponsesClient")
    
    args = parser.parse_args()
    
    generate_eval_set(
        datastore_path=args.datastore_path,
        embed_mod=args.embed_mod,
        output_eval_path=args.output_eval_path,
        output_datastore_path=args.output_datastore_path,
        num_samples=args.num_samples,
        num_claims_per_chunk=args.num_claims_per_chunk,
        retrieval_k=args.retrieval_k,
        num_retrieved_to_modify=args.num_retrieved_to_modify,
        llm_model=args.llm_model,
        seed=args.seed,
        debug=args.debug,
        max_concurrent=args.max_concurrent
    )
