from pylate import evaluation, indexes, models, retrieve

# Step 1: Initialize the ColBERT model
if __name__ == "__main__":
    dataset = "scifact" # Choose the dataset you want to evaluate
    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cuda" # "cpu" or "cuda" or "mps"
    )

    # Step 2: Create a Voyager index
    index = indexes.Voyager(
        index_folder="pylate-index",
        index_name=dataset,
        override=True,  # Overwrite any existing index
    )

    # Step 3: Load the documents, queries, and relevance judgments (qrels)
    documents, queries, qrels = evaluation.load_beir(
        dataset,  # Specify the dataset (e.g., "scifact")
        split="test",  # Specify the split (e.g., "test")
    )
    breakpoint()

    # Step 4: Encode the documents
    documents_embeddings = model.encode(
        [document["text"] for document in documents],
        batch_size=32,
        is_query=False,  # Indicate that these are documents
        show_progress_bar=True,
    )

    # Step 5: Add document embeddings to the index
    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )

    # Step 6: Encode the queries
    queries_embeddings = model.encode(
        queries,
        batch_size=32,
        is_query=True,  # Indicate that these are queries
        show_progress_bar=True,
    )

    # Step 7: Retrieve top-k documents
    retriever = retrieve.ColBERT(index=index)
    scores = retriever.retrieve(
        queries_embeddings=queries_embeddings,
        k=100,  # Retrieve the top 100 matches for each query
    )

    # Step 8: Evaluate the retrieval results
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=queries,
        metrics=[f"ndcg@{k}" for k in [1, 3, 5, 10, 100]] # NDCG for different k values
        + [f"hits@{k}" for k in [1, 3, 5, 10, 100]]       # Hits at different k values
        + ["map"]                                         # Mean Average Precision (MAP)
        + ["recall@10", "recall@100"]                     # Recall at k
        + ["precision@10", "precision@100"],              # Precision at k
    )

    print(results)