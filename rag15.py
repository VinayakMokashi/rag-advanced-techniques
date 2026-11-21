"""
Adaptive RAG: route each question, then check the evidence and the answer.

One LangGraph workflow combines three published ideas:

* Adaptive RAG (routing) - a router sends each question to the right source: the
  shop's handbook, a web search, or a direct reply for small talk.
* Corrective RAG (CRAG) - every retrieved passage is graded. If none is relevant, the
  question is rewritten and the handbook searched again; if that also fails, the
  workflow falls back to a web search.
* Self-RAG - every answer is checked twice: is each claim supported by the sources
  (no hallucination), and does it actually answer the question? An unsupported answer
  is rewritten with the unsupported claims pointed out; an answer that misses the
  question triggers the web fallback.

    route --> handbook --> retrieve --> grade --(none relevant)--> rewrite --> retrieve
      |                                   |                     (after MAX_REWRITES) --> web search
      |--> web search -----------------> generate <------------------------------------------'
      |                                   |
      '--> direct reply --> END        check --(unsupported)--> generate again (MAX_ATTEMPTS)
                                          |--(misses the question)--> web search (once)
                                          '--(good)--> END

Every loop has a cap, so no question can cycle forever.

The knowledge base is kestrel_handbook.md, the made-up customer handbook of a fictional
outdoor-gear shop, so the model cannot answer handbook questions from memory. Web search
uses the free ddgs metasearch package (DuckDuckGo, Bing and others) and needs no API key.
At the end the script checks by code that each test question took the expected route,
ended the expected way, and that answers contain the handbook's facts.

How to run
----------
    python rag15.py

Needs CEREBRAS_API_KEY or GROQ_API_KEY in .env (Cerebras is used when both are set).
The first run downloads the embedding model (~90 MB) and caches it.
"""

import os
import sys
import time
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model replies can contain non-ASCII

HANDBOOK = Path(__file__).with_name("kestrel_handbook.md")
TOP_K = 3
WEB_RESULTS = 5
MAX_REWRITES = 1  # handbook searches with a rewritten question before falling back to the web
MAX_ATTEMPTS = 2  # answers written from the same sources before giving up on grounding

# ---------------------------------------------------------------------------
# Model: gpt-oss-120b on Cerebras or Groq (both speak the OpenAI API)
# ---------------------------------------------------------------------------
load_dotenv()
if os.getenv("CEREBRAS_API_KEY"):
    PROVIDER, MODEL, BASE_URL, API_KEY = (
        "Cerebras", "gpt-oss-120b", "https://api.cerebras.ai/v1", os.environ["CEREBRAS_API_KEY"])
elif os.getenv("GROQ_API_KEY"):
    PROVIDER, MODEL, BASE_URL, API_KEY = (
        "Groq", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"])
else:
    sys.exit("No API key found: put CEREBRAS_API_KEY or GROQ_API_KEY in a .env file (see .env.example).")


def chat_model(max_tokens=1024):
    """gpt-oss counts its hidden reasoning against max_tokens, so it is always set."""
    return ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0,
                      max_tokens=max_tokens, reasoning_effort="low", max_retries=6)


# ---------------------------------------------------------------------------
# Sources: the handbook (one chunk per section, embedded locally) and the web
# ---------------------------------------------------------------------------
splitter = MarkdownHeaderTextSplitter([("##", "section")])
chunks = [
    Document(page_content=f"{d.metadata['section']}\n{d.page_content}", metadata=d.metadata)
    for d in splitter.split_text(HANDBOOK.read_text(encoding="utf-8"))
    if "section" in d.metadata  # skip the intro above the first section
]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = Chroma.from_documents(chunks, embeddings, collection_name="kestrel_rag15").as_retriever(
    search_kwargs={"k": TOP_K})


def search_web(query, tries=4):
    """Top web results (via the ddgs package) as text passages; an empty list if every try fails.

    Free search endpoints sometimes answer "No results found" for a query that works a few
    seconds later, so each search is retried after a short pause.
    """
    from ddgs import DDGS
    for attempt in range(1, tries + 1):
        try:
            hits = DDGS().text(query, max_results=WEB_RESULTS)
            return [f"{h['title']}\n{h['body']}\nSource: {h['href']}" for h in hits]
        except Exception as error:  # rate limits, network problems, no results
            print(f"  [web]      try {attempt} failed: {type(error).__name__}: {error}")
            time.sleep(2 * attempt)
    return []


# ---------------------------------------------------------------------------
# The model's jobs: route, grade, rewrite, write, check
# ---------------------------------------------------------------------------
class Route(BaseModel):
    source: Literal["handbook", "web_search", "direct"]


class Relevance(BaseModel):
    relevant: bool = Field(description="True if the passage contains information that answers the question")


class Grounding(BaseModel):
    unsupported_claims: list[str] = Field(description="Claims in the answer that the sources do not support")
    grounded: bool = Field(description="True if every factual claim in the answer is supported by the sources")


class Usefulness(BaseModel):
    answers_question: bool = Field(description="True if the answer gives the customer what they asked for")


router = chat_model().with_structured_output(Route)
relevance_grader = chat_model().with_structured_output(Relevance)
grounding_grader = chat_model().with_structured_output(Grounding)
usefulness_grader = chat_model().with_structured_output(Usefulness)
writer = chat_model(max_tokens=2048)


def write(prompt):
    """Free-text reply from the model. gpt-oss now and then sends its reply twice in a row
    ("abcabc"), so an exact repeat is cut back to one copy."""
    text = writer.invoke(prompt).content.strip()
    for cut in range(len(text) // 2 - 2, len(text) // 2 + 3):
        if 0 < cut < len(text) and text[:cut].strip() == text[cut:].strip():
            return text[:cut].strip()
    return text


ROUTER_PROMPT = """Route a message sent to the assistant of Kestrel Outfitters, an outdoor-gear shop.
- handbook: questions about Kestrel Outfitters itself: its stores, policies, prices, fees, shipping,
  returns, orders, Summit Club loyalty programme, warranty, repairs, price matching, gift cards or services.
- web_search: questions that need outside or up-to-date information, such as general advice about
  outdoor gear and activities, or news.
- direct: greetings, thanks or small talk that needs no information.

Message: {question}"""


class State(TypedDict, total=False):
    question: str       # what the customer asked (answers and checks always use this)
    query: str          # what the handbook is searched with (may be rewritten)
    source: str         # the router's choice
    documents: list[str]
    answer: str
    feedback: str       # why the previous answer was rejected
    rewrites: int
    attempts: int
    used_web: bool
    verdict: str        # how the run ended
    next: str           # the check node's decision


def route(state):
    source = router.invoke(ROUTER_PROMPT.format(question=state["question"])).source
    print(f"  [route]    {source}")
    return {"source": source, "query": state["question"], "rewrites": 0, "attempts": 0, "used_web": False}


def retrieve(state):
    return {"documents": [doc.page_content for doc in retriever.invoke(state["query"])]}


def grade(state):
    kept = []
    for passage in state["documents"]:
        verdict = relevance_grader.invoke(
            "You check whether a passage from a shop's handbook helps answer a customer's question. "
            "Answer relevant=true only if it contains information that answers at least part of the "
            "question; sharing a keyword is not enough.\n\n"
            f"Question: {state['question']}\n\nPassage:\n{passage}"
        )
        if verdict.relevant:
            kept.append(passage)
    print(f"  [grade]    {len(kept)} of {len(state['documents'])} passages relevant")
    return {"documents": kept}


def after_grading(state) -> Literal["generate", "rewrite", "web_search"]:
    if state["documents"]:
        return "generate"
    if state["rewrites"] < MAX_REWRITES:
        return "rewrite"
    print("  [grade]    the handbook has nothing relevant -> web search")
    return "web_search"


def rewrite(state):
    query = write(
        "A search of a shop's customer handbook found nothing relevant for this question. Rewrite it as "
        "a short search query that uses the words such a handbook would use. Reply with the query only.\n\n"
        f"Question: {state['question']}"
    )
    print(f"  [rewrite]  {query!r}")
    return {"query": query, "rewrites": state["rewrites"] + 1}


def web_search(state):
    query = write(
        "Turn this customer question into a short web search query of plain words (no symbols or "
        f"punctuation). Reply with the query only.\n\nQuestion: {state['question']}"
    )
    documents = search_web(query)
    print(f"  [web]      {query!r}: {len(documents)} results")
    return {"documents": documents, "used_web": True, "attempts": 0, "feedback": ""}


def generate(state):
    sources = "\n\n---\n\n".join(state["documents"]) or "(no sources were found)"
    retry_note = ""
    if state.get("feedback"):
        retry_note = (f"\n\nYour previous answer was rejected: {state['feedback']} "
                      "Write a new answer that states only what the sources say.")
    answer = write(
        "Answer the customer's question using only facts stated in the sources below, in at most four "
        "sentences. Leave out anything the sources do not say, even if you know it to be true. Give "
        "numbers exactly as the sources state them, in the same units, without converting them. If the "
        "sources answer only part of the question, answer that part; if they do not answer it at all, "
        "say so plainly instead of guessing."
        f"{retry_note}\n\nQuestion: {state['question']}\n\nSources:\n{sources}"
    )
    attempt = state.get("attempts", 0) + 1
    print(f"  [generate] attempt {attempt}" + (" (web sources)" if state.get("used_web") else ""))
    return {"answer": answer, "attempts": attempt}


def check(state):
    if not state["documents"]:
        print("  [check]    no sources, so nothing to check")
        return {"verdict": "no sources found", "next": "end"}
    sources = "\n\n---\n\n".join(state["documents"])
    grounding = grounding_grader.invoke(
        "Check an answer against its sources. List every factual claim in the answer that the sources "
        "do not support. Saying that the sources do not cover something is not a claim.\n\n"
        f"Sources:\n{sources}\n\nAnswer:\n{state['answer']}"
    )
    if not grounding.grounded:
        print(f"  [check]    NOT grounded: {grounding.unsupported_claims}")
        if state["attempts"] < MAX_ATTEMPTS:
            return {"feedback": "it made claims the sources do not support: "
                                + "; ".join(grounding.unsupported_claims), "next": "generate"}
        return {"verdict": "answer not supported by the sources", "next": "end"}
    useful = usefulness_grader.invoke(
        "Does this answer give the customer the information they asked for? It still does if it uses "
        "different units or gives an approximate figure. It does not if it says the information is not "
        "available or misses the main point of the question.\n\n"
        f"Question: {state['question']}\n\nAnswer:\n{state['answer']}"
    ).answers_question
    print(f"  [check]    grounded, {'answers the question' if useful else 'does NOT answer the question'}")
    if useful:
        return {"verdict": "answered", "next": "end"}
    if not state["used_web"]:
        return {"next": "web_search"}
    return {"verdict": "could not find an answer", "next": "end"}


def direct_reply(state):
    answer = write(
        "You are the friendly assistant of Kestrel Outfitters, an outdoor-gear shop. Reply briefly to "
        f"this message:\n\n{state['question']}"
    )
    return {"answer": answer, "verdict": "direct reply"}


graph = StateGraph(State)
for name, node in [("route", route), ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                   ("web_search", web_search), ("generate", generate), ("check", check),
                   ("direct_reply", direct_reply)]:
    graph.add_node(name, node)
graph.add_edge(START, "route")
graph.add_conditional_edges("route", lambda s: s["source"],
                            {"handbook": "retrieve", "web_search": "web_search", "direct": "direct_reply"})
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", after_grading)
graph.add_edge("rewrite", "retrieve")
graph.add_edge("web_search", "generate")
graph.add_edge("generate", "check")
graph.add_conditional_edges("check", lambda s: s["next"],
                            {"generate": "generate", "web_search": "web_search", "end": END})
graph.add_edge("direct_reply", END)
app = graph.compile()


# ---------------------------------------------------------------------------
# Test questions, with the behaviour each one should show (checked by code)
# ---------------------------------------------------------------------------
TESTS = [
    dict(question="How long do I have to return a GPS watch, and what do I need to send back with it?",
         source="handbook", used_web=False, facts=["21", "charger"]),
    dict(question="I'm a Summit Club member. How many points would a $200 order earn me, "
                  "and what can I exchange them for?",
         source="handbook", used_web=False, facts=["600", "$20"]),
    dict(question="How much does it cost to ship a kayak, and can I get it by express?",
         source="handbook", used_web=False, facts=["149"]),
    # General gear advice, so the answer must come from the web. Live search snippets change from
    # run to run, so two endings are correct: a supported answer, or the grounding check refusing
    # an answer that adds claims the snippets do not make.
    dict(question="What is the difference between a sleeping bag's comfort rating and its limit rating?",
         used_web=True, verdict=("answered", "answer not supported by the sources"), facts=[]),
    # Not in the handbook: rewrite, search again, then fall back to the web.
    dict(question="Does Kestrel Outfitters rent out tents?", source="handbook", used_web=True, facts=[]),
    dict(question="Hi! Thanks for the help earlier.", source="direct", facts=[]),
]


def run(question):
    """Run one question; return the nodes visited and the final state."""
    path, state = [], {}
    for update in app.stream({"question": question}, stream_mode="updates"):
        for node, values in update.items():
            path.append(node)
            state.update(values or {})
    return path, state


def failed_checks(test, state):
    problems = []
    if "source" in test and state.get("source") != test["source"]:
        problems.append(f"routed to {state.get('source')}, expected {test['source']}")
    if "used_web" in test and state.get("used_web", False) != test["used_web"]:
        problems.append(f"used_web={state.get('used_web', False)}, expected {test['used_web']}")
    if "verdict" in test and state.get("verdict") not in test["verdict"]:
        problems.append(f"ended with '{state.get('verdict')}', expected one of {test['verdict']}")
    missing = [f for f in test["facts"] if f.lower() not in state.get("answer", "").lower()]
    if missing:
        problems.append(f"answer is missing {missing}")
    return problems


def main():
    print(f"Model: {MODEL} on {PROVIDER} | handbook: {len(chunks)} sections, top {TOP_K} retrieved\n")
    summary = []
    with get_usage_metadata_callback() as usage:
        for test in TESTS:
            print(f"Q: {test['question']}")
            path, state = run(test["question"])
            problems = failed_checks(test, state)
            summary.append((test["question"], state.get("verdict", "?"), problems))
            print(f"  path:    {' -> '.join(path)}")
            print(f"  answer:  {' '.join(state.get('answer', '').split())}")
            print(f"  verdict: {state.get('verdict')}")
            print(f"  check:   {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}\n")

    tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
    print("Summary")
    for question, verdict, problems in summary:
        print(f"  {'PASS' if not problems else 'FAIL'}  {verdict:<38} {question[:60]}")
    print(f"\n{sum(not p for _, _, p in summary)}/{len(summary)} questions behaved as expected "
          f"| {tokens:,} tokens used")


if __name__ == "__main__":
    main()
