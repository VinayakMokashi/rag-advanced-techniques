"""
Query expansion with HyDE (Hypothetical Document Embeddings).

Keyword search only finds the words you typed: BM25 can match "I want running
shoes" only against titles that contain those words. HyDE asks an LLM to write a
hypothetical answer first, then searches with the query plus that answer, which
brings in related product names and vocabulary.

``HYDE_ANSWER`` below stands in for the LLM response, so the script runs offline
and gives the same result every time.

How to run
----------
    python rag13.py
"""

import numpy as np
from rank_bm25 import BM25Okapi

QUERY = "I want running shoes"
PRODUCT_TITLES = [
    "Adidas Ultraboost sneakers for men , Marathon Shoes",
    "Nike Air Zoom Pegasus trainers",
    "Puma casual footwear for everyday wear",
    "Woodland leather trekking boots",
    "Asics Gel-Kayano stability trainers",
    "Reebok CrossFit gym footwear",
    "Clarks formal leather boots",
    "Skechers lightweight walking sneakers",
    "New Balance Fresh Foam trail trainers",
    "Bata budget-friendly casual footwear",
    "Fila sports footwear for gym workouts",
    "Converse high-top casual sneakers",
]
HYDE_ANSWER = """Nike Air Jordan - Lightweight daily trainer with responsive cushioning.

Adidas Ultraboost 23 - High comfort running shoe with energy-returning midsole.

Asics Gel-Kayano 30 - Stability shoe designed for overpronators and long runs.

New Balance Fresh Foam 1080v13 - Plush cushioned shoe for marathon training."""

TOP_N = 3
# Titles with no real word overlap score around zero, so they are left out.
MIN_SCORE = 0.1


def search(bm25, text):
    """Return the top titles for ``text`` as (title, score) pairs, best first."""
    scores = bm25.get_scores(text.lower().split())
    best = np.argsort(scores)[::-1][:TOP_N]
    return [(PRODUCT_TITLES[i], scores[i]) for i in best if scores[i] > MIN_SCORE]


bm25 = BM25Okapi([title.lower().split() for title in PRODUCT_TITLES])

print(f'Query: "{QUERY}"\n')

print("Top product titles WITHOUT HyDE (BM25):")
for title, score in search(bm25, QUERY):
    print(f"- {title} (score: {score:.2f})")

print("\nTop product titles WITH HyDE (BM25):")
for title, score in search(bm25, QUERY + " " + HYDE_ANSWER):
    print(f"- {title} (score: {score:.2f})")
