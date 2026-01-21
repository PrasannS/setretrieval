from __future__ import annotations
import numpy as np
import torch
from pylate.utils.tensor import convert_to_tensor
from pylate.scores import colbert_scores

def maxmax_scores_pairwise(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
) -> torch.Tensor:
    scores = []
    # breakpoint()
    for query_embedding, document_embedding in zip(
        queries_embeddings, documents_embeddings
    ):
        query_embedding = convert_to_tensor(query_embedding)
        document_embedding = convert_to_tensor(document_embedding)

        query_document_score = torch.einsum(
            "sh,th->st",
            query_embedding,
            document_embedding,
        )

        scores.append(query_document_score.max(axis=-1).values.max(axis=-1).values)

    return torch.stack(scores, dim=0)

def maxmax_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
) -> torch.Tensor:
    # print("Using maxmax_scores!")

    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)

    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)
    # breakpoint()
    scores = scores.max(axis=-1).values.max(axis=-1).values
    return scores

def colbert_scores_multipos(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
    ordered: bool = True,
) -> torch.Tensor:
    
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)
    
    batch_size, num_query_tokens, embed_dim = queries_embeddings.shape
    _, num_positives, num_doc_tokens, _ = documents_embeddings.shape
    
    # Compute all pairwise similarities between all queries and all documents
    # Shape: (batch_size_queries, batch_size_docs, num_positives, num_query_tokens, num_doc_tokens)
    scores = torch.einsum("aqe,bpde->abpqd", queries_embeddings, documents_embeddings)
    
    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        # Shape: (batch_size_queries, 1, 1, num_query_tokens, 1)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
    
    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        # Shape: (1, batch_size_docs, num_positives, 1, num_doc_tokens)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(3)
    
    # For each query token and positive doc, find best matching doc token
    # Shape: (batch_size_queries, batch_size_docs, num_positives, num_query_tokens)
    max_per_query_per_doc = scores.max(axis=-1).values
    
    # Iteratively assign each query token to the best unused document
    total_scores = torch.zeros(batch_size, batch_size, device=queries_embeddings.device)
    

    for a in range(batch_size):  # query index
        query_norms = queries_embeddings[a].norm(dim=-1)  # Shape: (num_query_tokens,)
        non_zero_queries = query_norms > 0
        
        for b in range(batch_size):  # document index
            query_scores = max_per_query_per_doc[a, b]  # Shape: (num_positives, num_query_tokens)
            used_docs = torch.zeros(num_positives, dtype=torch.bool, device=queries_embeddings.device)
            qind = 0
            for q_idx in range(num_query_tokens):
                # skip zero query tokens
                if not non_zero_queries[q_idx]:
                    continue
                if used_docs.all():
                    break
                
                
                # Mask out already used documents
                mask = ~used_docs
                available_scores = torch.where(
                    mask,
                    query_scores[:, q_idx],
                    torch.tensor(float('-inf'), device=queries_embeddings.device, dtype=query_scores.dtype)
                )
                if ordered:
                    best_doc_idx = qind
                else:
                    best_doc_idx = available_scores.argmax()

                # Find best available document for this query token
                best_score = available_scores[best_doc_idx]
                
                # Only add score if it's not -inf (i.e., valid match exists)
                if not torch.isinf(best_score):
                    total_scores[a, b] += best_score
                    used_docs[best_doc_idx] = True
                qind += 1
    
    return total_scores

def route_colbscores_multipos(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
    ordered: bool = True,
):
    # breakpoint()
    # use the fancy thing for positives
    if len(documents_embeddings.shape) == 4:
        scores = colbert_scores_multipos(queries_embeddings, documents_embeddings, queries_mask, documents_mask, ordered)
    else: # otherwise use normal colbert scores (this might need some refinement)
        scores = colbert_scores(queries_embeddings, documents_embeddings, queries_mask, documents_mask)
    # breakpoint()
    return scores

def route_colbscores_multiquery(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
    ordered: bool = True,
):
    # breakpoint()
    # use the fancy thing for positives
    if len(queries_embeddings.shape) == 4:
        scores = colbert_scores_multiquery(queries_embeddings, documents_embeddings, queries_mask, documents_mask, ordered)
    else:
        scores = colbert_scores(queries_embeddings, documents_embeddings, queries_mask, documents_mask)
    return scores


def colbert_scores_multiquery(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
    ordered: bool = True,
) -> torch.Tensor:
    """Case D: Each document token must match to a different positive query.
    Returns (batch_size, batch_size) score matrix for all query-document pairs.
    
    Parameters
    ----------
    queries_embeddings : Shape (batch_size, num_queries, num_tokens_queries, embedding_size)
    documents_embeddings : Shape (batch_size, num_tokens_documents, embedding_size)
    queries_mask : Shape (batch_size, num_queries, num_tokens_queries)
    documents_mask : Shape (batch_size, num_tokens_documents)
    
    Returns
    -------
    scores : Shape (batch_size, batch_size)
    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)
    
    batch_size, num_queries, num_query_tokens, embed_dim = queries_embeddings.shape
    _, num_doc_tokens, _ = documents_embeddings.shape
    
    # Compute all pairwise similarities between all queries and all documents
    # Shape: (batch_size_queries, batch_size_docs, num_queries, num_doc_tokens, num_query_tokens)
    scores = torch.einsum("aqte,bde->abqdt", queries_embeddings, documents_embeddings)
    
    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        # Shape: (batch_size_queries, 1, num_queries, 1, num_query_tokens)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)
    
    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        # Shape: (1, batch_size_docs, 1, num_doc_tokens, 1)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2).unsqueeze(4)
    
    # For each document token and query, find best matching query token
    # Shape: (batch_size_queries, batch_size_docs, num_queries, num_doc_tokens)
    max_per_doc_per_query = scores.max(axis=-1).values
    
    # Iteratively assign each document token to the best unused query
    total_scores = torch.zeros(batch_size, batch_size, device=queries_embeddings.device)
    
    for a in range(batch_size):  # query index
        for b in range(batch_size):  # document index
            doc_scores = max_per_doc_per_query[a, b]  # Shape: (num_queries, num_doc_tokens)
            used_queries = torch.zeros(num_queries, dtype=torch.bool, device=queries_embeddings.device)
            
            # Check which document embeddings are non-zero
            doc_norms = documents_embeddings[b].norm(dim=-1)  # Shape: (num_doc_tokens,)
            non_zero_docs = doc_norms > 0
            dind = 0
            
            for d_idx in range(num_doc_tokens):
                # Skip document embeddings that are zero
                if not non_zero_docs[d_idx]:
                    continue
                if used_queries.all():
                    break
                
                # Create mask without inplace operation
                mask = ~used_queries
                available_scores = torch.where(
                    mask,
                    doc_scores[:, d_idx],
                    torch.tensor(float('-inf'), device=queries_embeddings.device, dtype=doc_scores.dtype)
                )
                if ordered:
                    best_query_idx = dind
                else:
                    best_query_idx = available_scores.argmax()
                # Find best available query for this document token
                best_score = available_scores[best_query_idx]
                
                # Only add score if it's not -inf (i.e., valid match exists)
                if not torch.isinf(best_score):
                    total_scores[a, b] += best_score
                    used_queries[best_query_idx] = True
                dind += 1
    return total_scores