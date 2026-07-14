"""
Chunking visualization for Retrieval-Augmented Generation (RAG).

This script splits a text document into overlapping chunks with LangChain's
``RecursiveCharacterTextSplitter`` and prints each chunk to the terminal,
highlighting the piece of text that OVERLAPS with the following chunk. Seeing the
overlap makes it obvious how ``chunk_size`` and ``chunk_overlap`` interact -- the
overlap is what preserves context across chunk boundaries during retrieval.

It also builds a Chroma vector store from the chunks (using HuggingFace
sentence-embeddings) and exposes a ``search`` method for quick similarity queries.

How to run
----------
    python rag04.py

The script reads ``doc.txt`` from the current folder, so run it from this
directory. The first run downloads the embedding model (~400 MB) and caches it.
"""

import textwrap

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VisualizeChunking:
    """Loads a document, splits it into overlapping chunks, and can visualize or
    search them.

    The whole pipeline runs in ``__init__``, so once the object is constructed it
    already holds the chunks (``self.docs``) and a ready-to-query vector store
    (``self.vectorstore``).
    """

    def __init__(self, file_path, chunk_size=200, chunk_overlap=20):
        # chunk_size / chunk_overlap are deliberately small so the overlap is
        # easy to see when the chunks are printed.
        self.text = self.load_document(file_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = self.split_document()
        self.embedding_model = self.get_embedding_model()
        self.vectorstore = self.create_vectorstore()

    def load_document(self, file_path):
        """Read the whole document into a single string."""
        print(f"Loading document: {file_path}")
        # encoding='utf-8' keeps this portable: Windows defaults to cp1252, which
        # raises on characters like em-dashes or smart quotes.
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"  Loaded {len(text)} characters.")
        return text

    def split_document(self):
        """Split the text into overlapping chunks.

        ``RecursiveCharacterTextSplitter`` tries to break on natural boundaries
        (paragraphs, then sentences, then words) while keeping each chunk close to
        ``chunk_size`` characters, with ``chunk_overlap`` characters shared between
        neighbouring chunks.
        """
        print(f"Splitting text (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs = text_splitter.split_text(self.text)
        print(f"  Created {len(docs)} chunks.")
        return docs

    def get_embedding_model(self):
        """Return the HuggingFace sentence-embedding model used for the vector store."""
        print("Loading embedding model (first run downloads ~400 MB)...")
        return HuggingFaceEmbeddings()

    def create_vectorstore(self):
        """Embed every chunk and store the vectors in an in-memory Chroma index."""
        print("Building Chroma vector store...")
        store = Chroma.from_texts(texts=self.docs, embedding=self.embedding_model)
        print("  Vector store ready.\n")
        return store

    def search(self, query, k=1):
        """Return the ``k`` chunks most similar to ``query`` (semantic search)."""
        return self.vectorstore.similarity_search(query, k=k)

    def view_splits(self):
        """Print every chunk, highlighting the overlap it shares with the next one.

        For each chunk we find the longest suffix that is also the prefix of the
        following chunk -- that shared span is the overlap -- and print it on a
        blue background (ANSI escape code) so it is easy to spot.
        """
        for i in range(len(self.docs) - 1):
            current_chunk = self.docs[i]
            next_chunk = self.docs[i + 1]

            print(f"\n\n{'='*80}")
            print(f"Chunk {i+1}/{len(self.docs)} - Length: {len(current_chunk)} chars")
            print(f"{'='*80}")

            # Find the overlap: the longest suffix of the current chunk that is
            # also a prefix of the next chunk.
            overlap_text = ""
            for j in range(len(current_chunk)):
                if current_chunk[j:] == next_chunk[:len(current_chunk)-j]:
                    overlap_text = current_chunk[j:]
                    break

            if overlap_text:
                pre_overlap = current_chunk[:-len(overlap_text)]
                overlap_wrapped = textwrap.fill(" "+overlap_text, width=80)
                print_val = textwrap.fill(pre_overlap + overlap_wrapped, width=80)
                print(print_val, end="")
                # \033[44m ... \033[0m = blue background, then reset colour.
                print(textwrap.fill(f"\033[44m{overlap_wrapped}\033[0m", width=80), end='')
            else:
                print(textwrap.fill(current_chunk, width=80))

        # The loop always looks one chunk ahead, so it stops one short -- print
        # the final chunk (which has no "next" chunk) separately.
        print(f"\n{'='*80}")
        print(f"Chunk {len(self.docs)}/{len(self.docs)} - Length: {len(self.docs[-1])} chars")
        print(f"{'='*80}")
        print(textwrap.fill(self.docs[-1], width=80))


if __name__ == "__main__":
    print("===== Chunking Visualizer =====\n")
    processor = VisualizeChunking('doc.txt')
    print("Highlighted text = overlap shared with the next chunk.")
    processor.view_splits()

    # The vector store is ready too -- uncomment to try semantic search:
    # print(processor.search("How is coffee processed after harvesting?"))
    # print(processor.search("Tell me about modern space exploration companies"))
