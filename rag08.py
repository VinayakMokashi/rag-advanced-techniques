"""
Cross-encoder reranking for retrieval.

A BI-ENCODER (like a sentence-transformer) embeds the query and each document
separately and compares them with cosine similarity -- fast, but the query and
document never "see" each other. A CROSS-ENCODER instead feeds the
(query, document) pair through the model together and outputs a single relevance
score. That joint attention makes it much more accurate at ranking, at the cost
of speed (the model must run once per candidate document).

Typical RAG use: retrieve a handful of candidates cheaply with a bi-encoder, then
rerank just those candidates with a cross-encoder for a more accurate final order.

This script scores one fixed query against a list of documents with the
``ms-marco-MiniLM`` cross-encoder and prints them from most to least relevant.

How to run
----------
    python rag08.py

The first run downloads the cross-encoder model (~80 MB) and caches it.
"""

from sentence_transformers import CrossEncoder

query = "What is the capital of France?"
documents = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital and largest city of Germany.",
    "France is a country in Western Europe with several overseas territories.",
    "The Eiffel Tower is located in Paris, France.",
    "Lyon is the third-largest city in France after Paris and Marseille.",
    "France has a population of approximately 67 million people.",
    "The capital of Italy is Rome, which is known as the Eternal City.",
    "Paris hosts many famous landmarks including the Louvre Museum."
]

print("Loading cross-encoder model (cross-encoder/ms-marco-MiniLM-L6-v2)...")
cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# Score every (query, document) pair jointly. Higher score = more relevant.
query_doc_pairs = [[query, doc] for doc in documents]
cross_encoder_scores = cross_encoder_model.predict(query_doc_pairs)

# Sort documents by score, highest first. This is the reranked order.
ranked = sorted(zip(cross_encoder_scores, documents), reverse=True)
ranked_documents = [doc for _, doc in ranked]

print(f"\nQuery: {query}\n")
print("Documents ranked by relevance (higher score = more relevant):")
for rank, (score, doc) in enumerate(ranked, 1):
    print(f"  {rank}. [{score:6.2f}] {doc}")
