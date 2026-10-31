# RAG Advanced Techniques

A hands-on continuation of the RAG series, exploring techniques that improve
**retrieval quality** in Retrieval-Augmented Generation: smarter **chunking**,
**reranking**, **query expansion**, and a full **PDF chat** application.

## What's inside

| File | Description |
|------|-------------|
| [`rag04.py`](rag04.py) | **Chunking visualization** — splits a document with `RecursiveCharacterTextSplitter` and prints each chunk, highlighting the text that **overlaps** with the next chunk so you can *see* how chunk size and overlap behave. |
| [`rag05.py`](rag05.py) | **PDF chat app** — a **Gradio** web UI that ingests PDFs, embeds them into a **Chroma** vector store, and answers questions with a **Groq**-hosted LLaMA model. Conversational RAG with memory (`ConversationalRetrievalChain`). |
| [`rag06.py`](rag06.py) | **Late chunking** — compares *traditional* chunking against the **late chunking** method (embed the whole document first, then pool per-chunk) using **Jina** embeddings, and prints which approach retrieves the right chunk. |
| [`rag08.py`](rag08.py) | **Cross-encoder reranking** — reorders candidate documents for a query with a **cross-encoder** (`ms-marco-MiniLM`) that scores query–document pairs jointly, a more accurate (but heavier) alternative to bi-encoder similarity. |
| [`rag10.py`](rag10.py) | **Measuring a reranker** — ranks labelled documents with a bi-encoder and a cross-encoder and prints **precision@k and recall@k** for both. On this example the cross-encoder reaches 1.00 / 1.00 at k=2, where the bi-encoder scores 0.50 / 0.50. |
| [`rag11.py`](rag11.py) | **Chunk size vs. retrieval** — splits one document at 50, 150 and 400 characters, indexes each in its own **Chroma** collection and runs the same query against all three. |
| [`rag12.py`](rag12.py) | **Chunking strategies side by side** — **fixed-length**, **semantic** (`SemanticChunker`) and **structural** (`MarkdownHeaderTextSplitter`) chunking on the same document. |
| [`rag13.py`](rag13.py) | **HyDE query expansion** — **BM25** keyword search with and without a hypothetical answer. Runs offline, using a fixed stand-in for the LLM's answer. |
| [`late_chunking_explained.ipynb`](late_chunking_explained.ipynb) | **Late chunking, step by step** — early vs. late embeddings for two chunks of one sentence, using `bge-small` and plain `transformers`. Saved with outputs, and it measures where late chunking helps and where it doesn't. A companion to `rag06.py`, which compares retrieval on a longer document. |

> **Related repos in this RAG series:** two sibling lessons live in their own repos —
> contextual embeddings (`rag07`) in
> [`rag-embeddings-basics`](https://github.com/VinayakMokashi/rag-embeddings-basics),
> and retrieval metrics / recall@k & precision@k (`rag09`) in
> [`rag-ragas-evaluation`](https://github.com/VinayakMokashi/rag-ragas-evaluation).

## Requirements

- Python 3.10+ (developed on 3.12)
- The packages in [`requirements.txt`](requirements.txt)
- A free [Groq API key](https://console.groq.com/keys) — **only** needed for `rag05.py`

```bash
pip install -r requirements.txt
```

> **Why the version pins?** `requirements.txt` pins `transformers<5`, `huggingface-hub<1`
> and `gradio<6`. The Jina model in `rag06.py` imports `transformers.onnx`, which was
> removed in transformers 5.x; and gradio 6 requires a newer `huggingface-hub` that
> conflicts with transformers 4.x. These pins keep every script working in one env.

> **First run downloads models** (~1 GB total: sentence-transformers, Jina, cross-encoder,
> and `bge-small` for the notebook) into your local HuggingFace cache. This happens once;
> later runs are fast.

## Setup (Groq key for `rag05.py`)

Copy the example env file and add your key:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_groq_api_key_here
```

## How to run

```bash
# 1) Chunking visualization (reads doc.txt in this folder)
python rag04.py

# 2) Late chunking — prints a traditional-vs-late comparison table
python rag06.py

# 3) Cross-encoder reranking — prints documents ranked by relevance
python rag08.py

# 4) PDF chat app — opens a browser tab (needs GROQ_API_KEY)
python rag05.py

# 5) Reranker metrics — precision@k and recall@k for both models
python rag10.py

# 6) Chunk size vs. retrieval
python rag11.py

# 7) Fixed-length, semantic and structural chunking side by side
python rag12.py

# 8) HyDE query expansion with BM25
python rag13.py
```

For `rag05.py`: upload one or more PDFs, click **Process PDFs**, then ask questions
in the chat box. Stop the server with `Ctrl+C`.

The late chunking notebook is saved with its outputs, so you can read it on GitHub
without running it. To run it yourself:

```bash
pip install jupyter
jupyter notebook late_chunking_explained.ipynb
```
