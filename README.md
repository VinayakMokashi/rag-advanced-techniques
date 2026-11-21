# RAG Advanced Techniques

A hands-on continuation of the RAG series, exploring techniques that improve
**retrieval quality** in Retrieval-Augmented Generation: smarter **chunking**,
**reranking**, **query expansion**, a full **PDF chat** application, and **agentic RAG**
workflows that check their own retrieval and answers.

## What's inside

| File | Description |
|------|-------------|
| [`rag04.py`](rag04.py) | **Chunking visualization** — splits a document with `RecursiveCharacterTextSplitter` and prints each chunk, highlighting the text that **overlaps** with the next chunk so you can *see* how chunk size and overlap behave. |
| [`rag05.py`](rag05.py) | **PDF chat app** — a **Gradio** web UI that ingests PDFs, embeds them into a **Chroma** vector store, and answers questions with `gpt-oss-120b` hosted on **Groq**. Conversational RAG with memory (`ConversationalRetrievalChain`). |
| [`rag06.py`](rag06.py) | **Late chunking** — compares *traditional* chunking against the **late chunking** method (embed the whole document first, then pool per-chunk) using **Jina** embeddings, and prints which approach retrieves the right chunk. |
| [`rag08.py`](rag08.py) | **Cross-encoder reranking** — reorders candidate documents for a query with a **cross-encoder** (`ms-marco-MiniLM`) that scores query–document pairs jointly, a more accurate (but heavier) alternative to bi-encoder similarity. |
| [`rag10.py`](rag10.py) | **Measuring a reranker** — ranks labelled documents with a bi-encoder and a cross-encoder and prints **precision@k and recall@k** for both. On this example the cross-encoder reaches 1.00 / 1.00 at k=2, where the bi-encoder scores 0.50 / 0.50. |
| [`rag11.py`](rag11.py) | **Chunk size vs. retrieval** — splits one document at 50, 150 and 400 characters, indexes each in its own **Chroma** collection and runs the same query against all three. |
| [`rag12.py`](rag12.py) | **Chunking strategies side by side** — **fixed-length**, **semantic** (`SemanticChunker`) and **structural** (`MarkdownHeaderTextSplitter`) chunking on the same document. |
| [`rag13.py`](rag13.py) | **HyDE query expansion** — **BM25** keyword search with and without a hypothetical answer. Runs offline, using a fixed stand-in for the LLM's answer. |
| [`rag14.py`](rag14.py) | **Agentic RAG** — retrieval is a tool the model *may* call. A **LangGraph** loop grades what the search returns, and either writes the answer or rewrites the question and searches again (capped). Checks by code that the model searched when it should. |
| [`rag15.py`](rag15.py) | **Adaptive RAG with corrective and self-checking steps** — a router picks the handbook, a web search or a direct reply; retrieved passages are graded, with a rewrite and then a **web fallback** when none is relevant (CRAG); every answer is checked for **unsupported claims** and for whether it **answers the question** (Self-RAG). Checks each test question's route and ending by code. |
| [`kestrel_handbook.md`](kestrel_handbook.md) | Knowledge base for `rag14`/`rag15`: the customer handbook of **Kestrel Outfitters**, a made-up outdoor-gear shop, so the model cannot answer from memory. |
| [`late_chunking_explained.ipynb`](late_chunking_explained.ipynb) | **Late chunking, step by step** — early vs. late embeddings for two chunks of one sentence, using `bge-small` and plain `transformers`. Saved with outputs, and it measures where late chunking helps and where it doesn't. A companion to `rag06.py`, which compares retrieval on a longer document. |

> **Related repos in this RAG series:** two sibling projects live in their own repos —
> contextual embeddings (`rag07`) in
> [`rag-embeddings-basics`](https://github.com/VinayakMokashi/rag-embeddings-basics),
> and retrieval metrics / recall@k & precision@k (`rag09`) in
> [`rag-ragas-evaluation`](https://github.com/VinayakMokashi/rag-ragas-evaluation).

## Requirements

- Python 3.10+ (developed on 3.12)
- The packages in [`requirements.txt`](requirements.txt)
- An LLM API key, only for the scripts that call a model:
  - `rag05.py` needs a free [Groq API key](https://console.groq.com/keys);
  - `rag14.py` and `rag15.py` need a [Cerebras](https://cloud.cerebras.ai) **or** a Groq key
    (both serve `gpt-oss-120b`; Cerebras is used when both are set).
- An internet connection for `rag15.py`'s web search (no search API key needed).

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

## Setup (API keys for `rag05`, `rag14`, `rag15`)

Copy the example env file and add your key(s):

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_groq_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here   # optional; rag14/rag15 prefer it when set
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

# 9) Agentic RAG: the model decides when to search (needs an API key)
python rag14.py

# 10) Adaptive RAG with grading, web fallback and answer checks (needs an API key)
python rag15.py
```

For `rag05.py`: upload one or more PDFs, click **Process PDFs**, then ask questions
in the chat box. Stop the server with `Ctrl+C`.

The late chunking notebook is saved with its outputs, so you can read it on GitHub
without running it. To run it yourself:

```bash
pip install jupyter
jupyter notebook late_chunking_explained.ipynb
```

## Agentic RAG (`rag14`, `rag15`)

Plain RAG retrieves once and trusts whatever comes back. These two scripts put the model in
charge of checking: it decides whether to search, grades what it found, and checks its own
answer before returning it. Both are [LangGraph](https://langchain-ai.github.io/langgraph/)
workflows over [`kestrel_handbook.md`](kestrel_handbook.md), a made-up shop handbook, so
every handbook answer has to come from retrieval rather than the model's memory. Embeddings
(`all-MiniLM-L6-v2`) run locally; the model is `gpt-oss-120b` on Cerebras or Groq.

**`rag14.py` — the model decides when to search.** Searching the handbook is a tool. For a
greeting the model just replies; for a policy question it calls the tool, a grader judges
whether the passages answer the question, and if not, the question is rewritten and the
model searches again. The rewrite loop runs at most once (`MAX_REWRITES`), so a question the
handbook can't answer ends with "the handbook does not cover it":

```
Q: Do you rent out tents or kayaks?
  [agent]    searches: 'tent rental kayak rental'
  [grade]    passages are not relevant -> rewrite
  [rewrite]  'tent kayak rental policy'
  [agent]    searches: 'rental policy tent kayak'
  [grade]    still not relevant after 1 rewrite(s) -> generate anyway
  path:   agent -> retrieve -> rewrite -> agent -> retrieve -> generate
  answer: The handbook does not cover it.
```

**`rag15.py` — route, grade, correct, check.** One graph combines three published methods:

| Step | Method | What happens |
|---|---|---|
| Route | Adaptive RAG | The router sends each message to the **handbook**, a **web search** (general gear advice, news) or a **direct reply** (small talk). |
| Grade | Corrective RAG | Each retrieved passage is graded on its own. None relevant → rewrite the question and search again; still none → **fall back to the web**. |
| Check | Self-RAG | Is every claim supported by the sources? If not, the answer is rewritten with the unsupported claims named (up to `MAX_ATTEMPTS`). Does it answer the question? If not, and the web hasn't been tried, search the web. |

```
Q: Does Kestrel Outfitters rent out tents?
  [route]    handbook
  [grade]    0 of 3 passages relevant
  [rewrite]  'Tent Rental Services Kestrel Outfitters'
  [grade]    0 of 3 passages relevant
  [grade]    the handbook has nothing relevant -> web search
  [web]      'kestrel outfitters tent rental': 5 results
  [check]    grounded, does NOT answer the question
  path:    route -> retrieve -> grade -> rewrite -> retrieve -> grade -> web_search -> generate -> check
  answer:  The provided sources do not mention Kestrel Outfitters or indicate whether they rent out tents.
  verdict: could not find an answer
```

Every loop is capped, so no question can cycle forever. Each script ends by checking its test
questions **by code**: whether `rag14` searched when it should, which route each `rag15`
question took and how it ended, and that handbook answers contain the right facts (`$149`,
`21` days, `600` points and so on).

### Measured results

| Script | Provider | Runs | Result | Tokens per run | Time per run |
|---|---|---|---|---|---|
| `rag14.py` | Cerebras | 2 (one from a fresh clone + fresh venv) | 4/4 both runs | ~5.8k | ~40 s |
| `rag14.py` | Groq | 1 | 4/4 | ~5.4k | ~80 s |
| `rag15.py` | Cerebras | 3 (fresh clone + fresh venv) | 6/6 every run | 18–20k | ~1 min |
| `rag15.py` | Groq | 1 | 6/6 | ~17k | ~2.5 min |

Groq is slower only because its free tier allows 8,000 tokens a minute, so the client waits
and retries.

### Things worth knowing

- **Web snippets are short, and the model fills the gaps from memory.** Each search result is
  a sentence or two, so on the web question the model sometimes adds facts the snippets
  don't contain ("…without risk of hypothermia"), and the grounding check rejects
  them. An earlier version of the question ("what rating for camping at 0 °C?") showed the
  check catching a real error: the model converted 17–22 °F to "−5 to −10 °C" (it is about
  −8 to −6 °C). Because live results change from run to run, the web test accepts either
  correct ending: a supported answer, or a refusal of an unsupported one.
- **Free web search is flaky.** The `ddgs` endpoints sometimes answer "No results found" for a
  query that works seconds later, so each search is retried up to four times.
- **The graders are LLM judgements too.** In one run the usefulness check failed a correct
  answer that gave the rating in °F for a question asked in °C, after passing a near-identical
  answer in another run; its prompt now says that units and approximations still count.
- **gpt-oss sometimes repeats its reply** (`"abcabc"`). `write()` keeps one copy.
- **The web can't know a private handbook.** For the tents question, the correct outcome is
  "could not find an answer", and that is what the script returns, even though the web
  results describe other outfitters' tent rentals. In a real deployment you might skip the web
  fallback for questions about your own policies.

## Credits

`rag14.py` and `rag15.py` are rewrites of the ideas in the
[LangGraph RAG examples](https://github.com/langchain-ai/langgraph/tree/main/examples/rag)
(MIT License): agentic RAG, [Corrective RAG](https://arxiv.org/abs/2401.15884),
[Self-RAG](https://arxiv.org/abs/2310.11511) and [Adaptive RAG](https://arxiv.org/abs/2403.14403).
The versions here run on `gpt-oss-120b` with local embeddings and a keyless web search, and
add capped loops, retry feedback and code-checked test questions.
