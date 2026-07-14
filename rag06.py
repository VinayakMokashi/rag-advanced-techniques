"""
Late chunking vs. traditional chunking.

References:
  Paper: https://arxiv.org/pdf/2409.04701
  Code:  https://github.com/jina-ai/late-chunking

TRADITIONAL chunking splits a document into pieces FIRST and then embeds each
piece on its own -- so a chunk knows nothing about the text around it. LATE
chunking flips the order: it runs the WHOLE document through the transformer
first (so every token attends to the full context) and only then pools the token
embeddings into per-chunk vectors. The result is chunk embeddings that still
"remember" the surrounding document, which usually improves retrieval --
especially for chunks full of references like "it" or "the city".

This script builds both kinds of embeddings for a small Berlin document and
prints a table showing, for each query, whether traditional vs. late chunking
retrieved a chunk that actually contains the answer.

How to run
----------
    python rag06.py

First run downloads the MiniLM (~90 MB) and Jina (~550 MB) models and caches them.
"""

import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LateChunking:
    """Builds traditional and late-chunking embeddings for one document and
    provides retrieval + evaluation helpers to compare the two approaches."""

    def __init__(self, document):
        self.document = document
        # Bi-encoder used for the *traditional* chunk embeddings + query encoding.
        self.sentence_model = self.setup_sentence_model()
        # Jina model supports late chunking: it embeds a long document and lets
        # us pool its per-token embeddings into per-chunk vectors.
        self.jina_tokenizer = self.setup_jina_tokenizer()
        self.jina_model = self.setup_jina_model()

        # Build both representations up front.
        self.chunks, self.embeddings = self.perform_chunking()
        self.late_chunks, self.late_embeddings = self.perform_late_chunking()

    def setup_sentence_model(self):
        print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def setup_jina_tokenizer(self):
        print("Loading Jina tokenizer...")
        return AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

    def setup_jina_model(self):
        print("Loading Jina model (first run downloads ~550 MB)...")
        return AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

    def perform_chunking(self, chunk_size=50, overlap=10):
        """TRADITIONAL chunking: split into word windows, embed each separately.

        The document is split into overlapping windows of ``chunk_size`` words
        (stepping ``chunk_size - overlap`` words forward each time), and every
        window is embedded independently by the sentence model.
        """
        print("Performing traditional chunking + embedding...")
        words = self.document.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(' '.join(words[i:i + chunk_size]))
        chunk_embeddings = self.sentence_model.encode(chunks)
        return chunks, chunk_embeddings

    def perform_late_chunking(self):
        """LATE chunking: embed the whole document, then pool tokens per chunk.

        Steps:
          1. Tokenize the document, keeping each token's character offset.
          2. Use sentence-ending periods (a "." followed by space/newline) as
             chunk boundaries, recording both the character span (to slice the
             readable text) and the token span (to pool embeddings).
          3. Run the FULL document through the model once, so every token
             embedding is contextualized by the entire document.
          4. For each chunk, mean-pool the token embeddings inside its token span
             to get that chunk's embedding.
        """
        print("Performing late chunking + embedding...")
        inputs_with_offsets = self.jina_tokenizer(self.document, return_tensors='pt', return_offsets_mapping=True)
        token_offsets = inputs_with_offsets['offset_mapping'][0]
        token_ids = inputs_with_offsets['input_ids'][0]

        # Walk the tokens and split on sentence-ending periods. For each chunk we
        # store two parallel spans: character positions (to slice readable text)
        # and token indices (to pool the matching token embeddings later).
        punctuation_mark_id = self.jina_tokenizer.convert_tokens_to_ids('.')
        chunk_positions, token_span_annotations = [], []
        span_start_char, span_start_token = 0, 0

        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets)):
            if i < len(token_ids)-1:
                # A period followed by a space/newline marks a sentence boundary.
                if token_id == punctuation_mark_id and self.document[end:end+1] in [' ', '\n']:
                    chunk_positions.append((span_start_char, int(end)))
                    token_span_annotations.append((span_start_token, i+1))
                    span_start_char, span_start_token = int(end)+1, i+1

        chunks = [self.document[start:end].strip() for start, end in chunk_positions]

        # Embed the ENTIRE document once -- this single full-context pass is what
        # makes the chunking "late".
        inputs = self.jina_tokenizer(self.document, return_tensors='pt')
        model_output = self.jina_model(**inputs)
        token_embeddings = model_output[0]

        # Mean-pool the token embeddings within each chunk's token span.
        embeddings = []
        for start_token, end_token in token_span_annotations:
            if end_token > start_token:
                chunk_embedding = token_embeddings[0, start_token:end_token].mean(dim=0)
                embeddings.append(chunk_embedding.detach().cpu().numpy())
        embeddings = np.array(embeddings)
        return chunks, embeddings

    def retrieval(self, query):
        """Return the best TRADITIONAL chunk for ``query`` and its similarity."""
        query_embedding = self.sentence_model.encode(query).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        best_match_idx = np.argmax(similarities)
        return self.chunks[best_match_idx], similarities[best_match_idx]

    def late_retrieval(self, query):
        """Return the best LATE-CHUNKING chunk for ``query`` and its similarity."""
        query_embedding = self.jina_model.encode(query).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.late_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        return self.late_chunks[best_match_idx], similarities[best_match_idx]

    def evaluate_retrieval(self, queries_with_answers):
        """For each query, check whether traditional retrieval returned a chunk
        that contains the expected answer text."""
        results = []
        for query, answer_text in queries_with_answers:
            # Ground truth: which chunks actually contain the answer string?
            answer_chunks = [i for i, chunk in enumerate(self.chunks) if answer_text in chunk]
            best_chunk, similarity = self.retrieval(query)
            best_chunk_idx = self.chunks.index(best_chunk)
            contains_answer = best_chunk_idx in answer_chunks
            results.append({
                'query': query,
                'answer': answer_text,
                'traditional_correct': contains_answer,
            })
        return results

    def evaluate_late_retrieval(self, queries_with_answers):
        """Same as ``evaluate_retrieval`` but for the late-chunking embeddings."""
        results = []
        for query, answer_text in queries_with_answers:
            answer_chunks = [i for i, chunk in enumerate(self.late_chunks) if answer_text in chunk]
            best_chunk, similarity = self.late_retrieval(query)
            best_chunk_idx = self.late_chunks.index(best_chunk)
            contains_answer = best_chunk_idx in answer_chunks
            results.append({
                'query': query,
                'answer': answer_text,
                'late_chunking_correct': contains_answer,
            })
        return results

    def create_comparison_table(self, queries_with_answers):
        """Run both evaluations and combine them into one pandas DataFrame."""
        retrieval_results = self.evaluate_retrieval(queries_with_answers)
        late_retrieval_results = self.evaluate_late_retrieval(queries_with_answers)

        combined_results = []
        for trad, late in zip(retrieval_results, late_retrieval_results):
            combined_results.append({
                'query': trad['query'],
                'answer': trad['answer'],
                'traditional_correct': trad['traditional_correct'],
                'late_chunking_correct': late['late_chunking_correct'],
            })

        return pd.DataFrame(combined_results)


if __name__ == "__main__":
    document = """
    Berlin is the capital and largest city of Germany. The city has a rich history dating back centuries. It was founded in the 13th century and has been a significant cultural and political center throughout European history.

    The metropolis experienced dramatic changes during the 20th century, including two world wars and a period of division. After reunification, it underwent extensive reconstruction and modernization efforts.

    Its population reached 3.85 million inhabitants in 2023, making it the most populous urban area in the country. This represents a significant increase from previous decades, driven largely by immigration and economic opportunities.

    The city is known for its vibrant cultural scene and historical significance. Many tourists visit its famous landmarks each year, contributing significantly to the local economy. The Brandenburg Gate stands as its most iconic symbol.
    """

    # (query, expected answer substring) pairs used to score retrieval.
    queries_with_answers = [
        ("What is Berlin's population?",                "3.85 million inhabitants"),
        ("How many people live in the German capital?", "3.85 million inhabitants"),
        ("What famous landmark is in Berlin?",          "Brandenburg Gate"),
        ("What is the city known for?",                 "vibrant cultural scene"),
        ("When was it founded?",                        "13th century"),
        ("When was Berlin founded?",                         "13th century"),
        ("What happened to Berlin during the 20th century?", "two world wars"),
        ("What's the most iconic symbol",                    "Brandenburg Gate"),
        ("What happened in the 20th century?",               "two world wars")
    ]

    # Show all DataFrame columns instead of letting pandas truncate them to "...".
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 45)

    retrieval = LateChunking(document)

    query = "What is Berlin's population?"
    best_chunk, similarity = retrieval.retrieval(query)
    print(f"\nExample query: {query!r}")
    print(f"  Best traditional chunk (similarity {similarity:.3f}):")
    print(f"    {best_chunk[:90]}...")

    print("\n===== Traditional vs Late Chunking (retrieval correctness) =====")
    comparison_df = retrieval.create_comparison_table(queries_with_answers)
    print(comparison_df)
