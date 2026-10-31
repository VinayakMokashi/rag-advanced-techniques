"""
Three ways to chunk the same document: fixed-length, semantic and structural.

All three run on one short markdown document, so the chunks can be compared
directly:

  - FIXED-LENGTH -- ``CharacterTextSplitter`` only cuts at its ``separator``. With
                    the default (a blank line), a paragraph longer than
                    ``chunk_size`` stays whole. Cutting at single line breaks gets
                    much closer to the target size.
  - SEMANTIC     -- ``SemanticChunker`` embeds the sentences and starts a new chunk
                    where the meaning shifts.
  - STRUCTURAL   -- ``MarkdownHeaderTextSplitter`` cuts at headers and moves them
                    out of the text and into metadata.

How to run
----------
    python rag12.py

The first run downloads the embedding model (~90 MB) and caches it.
"""

import logging
import textwrap

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter

# strip() matters: trailing whitespace otherwise becomes an empty final semantic chunk.
TEXT = textwrap.dedent("""
    # Machine Learning Fundamentals

    Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

    ## Types of Machine Learning

    ### Supervised Learning
    Supervised learning uses labeled training data to learn a mapping function from input to output. Common examples include:
    - Classification: Predicting categories (spam/not spam)
    - Regression: Predicting continuous values (house prices)

    ### Unsupervised Learning
    Unsupervised learning finds hidden patterns in data without labeled examples:
    - Clustering: Grouping similar data points
    - Dimensionality reduction: Reducing feature space

    ### Reinforcement Learning
    An agent learns through interaction with an environment, receiving rewards or penalties for actions taken.
    """).strip()


def print_chunks(title, chunks):
    """Print each chunk's length and text, plus its metadata if it has any."""
    print(f"\n{'=' * 70}\n{title}: {len(chunks)} chunks\n{'=' * 70}")
    for i, chunk in enumerate(chunks, 1):
        content = getattr(chunk, "page_content", chunk)
        print(f"Chunk {i} (length: {len(content)}):")
        print(content)
        if getattr(chunk, "metadata", None):
            print(f"Metadata: {chunk.metadata}")
        print("-" * 20)


# --- Fixed-length ---------------------------------------------------------------
# The first splitter overshoots chunk_size, which is the point of the comparison,
# so silence LangChain's "Created a chunk of size ..." warning and show the lengths.
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)
blank_line_splitter = CharacterTextSplitter(chunk_size=160, chunk_overlap=20)
print_chunks("Fixed-length, cutting at blank lines (default)", blank_line_splitter.split_text(TEXT))

line_break_splitter = CharacterTextSplitter(chunk_size=160, chunk_overlap=20, separator="\n")
print_chunks("Fixed-length, cutting at line breaks", line_break_splitter.split_text(TEXT))

# --- Semantic -------------------------------------------------------------------
print("\nLoading embedding model (first run downloads it)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_splitter = SemanticChunker(embeddings)
print_chunks("Semantic", semantic_splitter.create_documents([TEXT]))

# --- Structural -----------------------------------------------------------------
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
print_chunks("Markdown headers", markdown_splitter.split_text(TEXT))
