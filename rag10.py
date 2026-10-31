"""
Measuring a reranker: bi-encoder vs. cross-encoder with precision@k and recall@k.

A BI-ENCODER embeds the query and each document separately, so a document that
merely shares the query's words can outrank one that actually answers it. A
CROSS-ENCODER reads the (query, document) pair together. ``rag08.py`` shows the
reranked order; this script measures whether that order is better.

Five documents are ranked for one query, using hand-written relevance labels: two
documents state the price and the rest just share the same words. At each cut-off
``k`` we ask two questions:

  - precision@k -- of the top k documents, how many are relevant?
  - recall@k    -- of all the relevant documents, how many are in the top k?

How to run
----------
    python rag10.py

The first run downloads both models (~170 MB) and caches them.
"""

from sentence_transformers import CrossEncoder, SentenceTransformer

QUERY = "What is the price of Coke 500ml?"
DOCS = [
    "Coke 500ml costs Rs 40 at our store.",
    "Customers keep asking the price of Coke 500ml.",
    "Coke 500ml price tags were reprinted in a new font this week.",
    "A 500ml bottle of Coca-Cola sells for forty rupees here.",
    "Employees can apply for leave on the HR portal.",
]
# 1 = the document answers the query. Documents 1 and 2 only share its words.
RELEVANT = [1, 0, 0, 1, 0]


def print_metrics(name, ranking):
    """Print the ranking, then precision and recall at every cut-off."""
    print(f"\n--- {name} ranking: {ranking} ---")
    for rank, index in enumerate(ranking, 1):
        label = "relevant" if RELEVANT[index] else "        "
        print(f"  {rank}. [{label}] {DOCS[index]}")
    print()
    for k in range(1, len(DOCS) + 1):
        hits = sum(RELEVANT[index] for index in ranking[:k])
        print(f"  k={k}  precision={hits / k:.2f}  recall={hits / sum(RELEVANT):.2f}")


print("Loading models (first run downloads them)...")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

print(f"\nQuery: {QUERY}")

# Bi-encoder: embed separately, then sort by cosine similarity.
query_vector = bi_encoder.encode(QUERY)
doc_vectors = bi_encoder.encode(DOCS)
similarities = bi_encoder.similarity(query_vector, doc_vectors)[0]
print_metrics("bi-encoder", similarities.argsort(descending=True).tolist())

# Cross-encoder: score each (query, document) pair jointly.
print_metrics("cross-encoder", [hit["corpus_id"] for hit in cross_encoder.rank(QUERY, DOCS)])
