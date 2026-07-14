# RAG Advanced Techniques

A hands-on continuation of the RAG series, exploring techniques that improve
**retrieval quality** in Retrieval-Augmented Generation: smarter **chunking**,
**reranking**, and a full **PDF chat** application.

## What's inside

| File | Description |
|------|-------------|
| [`rag04.py`](rag04.py) | **Chunking visualization** — splits a document with `RecursiveCharacterTextSplitter` and prints each chunk, highlighting the text that **overlaps** with the next chunk so you can *see* how chunk size and overlap behave. |
| [`rag05.py`](rag05.py) | **PDF chat app** — a **Gradio** web UI that ingests PDFs, embeds them into a **Chroma** vector store, and answers questions with a **Groq**-hosted LLaMA model. Conversational RAG with memory (`ConversationalRetrievalChain`). |
| [`rag06.py`](rag06.py) | **Late chunking** — compares *traditional* chunking against the **late chunking** method (embed the whole document first, then pool per-chunk) using **Jina** embeddings, and prints which approach retrieves the right chunk. |
| [`rag08.py`](rag08.py) | **Cross-encoder reranking** — reorders candidate documents for a query with a **cross-encoder** (`ms-marco-MiniLM`) that scores query–document pairs jointly, a more accurate (but heavier) alternative to bi-encoder similarity. |

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
> conflicts with transformers 4.x. These pins keep all four scripts working in one env.

> **First run downloads models** (~1 GB total: sentence-transformers, Jina, cross-encoder)
> into your local HuggingFace cache. This happens once; later runs are fast.

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
```

For `rag05.py`: upload one or more PDFs, click **Process PDFs**, then ask questions
in the chat box. Stop the server with `Ctrl+C`.
