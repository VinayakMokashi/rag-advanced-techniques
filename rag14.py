"""
Agentic RAG: the model decides when to search.

Plain RAG retrieves for every question. Here retrieval is a tool the model may call,
and a small LangGraph loop checks what came back before an answer is written:

    agent --(calls the search tool?)--> retrieve --> grade the passages
      |                                                  |            |
      no: answers directly                          relevant     not relevant
      |                                                  |            |
     END                                             generate     rewrite the question
                                                         |        and let the agent
                                                        END       search again

The rewrite loop is capped (MAX_REWRITES), so a question the handbook cannot answer
ends with "the handbook doesn't cover this" instead of searching forever.

The knowledge base is kestrel_handbook.md, the made-up customer handbook of a fictional
outdoor-gear shop, so the model cannot answer from memory. At the end the script checks
by code that the agent searched when it should, and that answers contain the handbook's
facts.

How to run
----------
    python rag14.py

Needs CEREBRAS_API_KEY or GROQ_API_KEY in .env (Cerebras is used when both are set).
The first run downloads the embedding model (~90 MB) and caches it.
"""

import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model replies can contain non-ASCII

HANDBOOK = Path(__file__).with_name("kestrel_handbook.md")
TOP_K = 3
MAX_REWRITES = 1

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
# Knowledge base: one chunk per handbook section, embedded locally into Chroma
# ---------------------------------------------------------------------------
splitter = MarkdownHeaderTextSplitter([("##", "section")])
chunks = [
    Document(page_content=f"{d.metadata['section']}\n{d.page_content}", metadata=d.metadata)
    for d in splitter.split_text(HANDBOOK.read_text(encoding="utf-8"))
    if "section" in d.metadata  # skip the intro above the first section
]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = Chroma.from_documents(chunks, embeddings, collection_name="kestrel_rag14").as_retriever(
    search_kwargs={"k": TOP_K})


@tool
def search_handbook(query: str) -> str:
    """Search the Kestrel Outfitters customer handbook: stores and opening hours, returns
    and refunds, shipping, order changes, the Summit Club loyalty programme, warranty and
    repairs, price matching and gift cards."""
    return "\n\n---\n\n".join(doc.page_content for doc in retriever.invoke(query))


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class AgentState(MessagesState):
    rewrites: int


class Relevance(BaseModel):
    relevant: bool = Field(description="True if the passages contain information that answers the question")


AGENT_PROMPT = (
    "You are the customer assistant of Kestrel Outfitters, an outdoor-gear shop. For any question "
    "about Kestrel's stores, policies, prices, fees or services, call search_handbook before you answer. "
    "For greetings or questions about what you can help with, answer directly without searching."
)
agent_model = chat_model().bind_tools([search_handbook])
grader = chat_model().with_structured_output(Relevance)
writer = chat_model(max_tokens=2048)


def write(prompt):
    """Free-text reply from the model. gpt-oss now and then sends its reply twice in a row
    ("abcabc"), so an exact repeat is cut back to one copy."""
    text = writer.invoke(prompt).content.strip()
    for cut in range(len(text) // 2 - 2, len(text) // 2 + 3):
        if 0 < cut < len(text) and text[:cut].strip() == text[cut:].strip():
            return text[:cut].strip()
    return text


def user_question(state):
    """The customer's original question (the first message)."""
    return state["messages"][0].content


def agent(state):
    reply = agent_model.invoke([SystemMessage(AGENT_PROMPT)] + state["messages"])
    if reply.tool_calls:
        print(f"  [agent]    searches: {reply.tool_calls[0]['args'].get('query')!r}")
    else:
        print("  [agent]    answers without searching")
    return {"messages": [reply]}


def grade_passages(state) -> Literal["generate", "rewrite"]:
    passages = state["messages"][-1].content
    verdict = grader.invoke(
        "You check whether passages from a shop's handbook help answer a customer's question. "
        "Answer relevant=true only if they contain information that answers at least part of it; "
        "sharing a keyword is not enough.\n\n"
        f"Question: {user_question(state)}\n\nPassages:\n{passages}"
    )
    if verdict.relevant:
        print("  [grade]    passages are relevant -> generate")
        return "generate"
    if state["rewrites"] >= MAX_REWRITES:
        print(f"  [grade]    still not relevant after {MAX_REWRITES} rewrite(s) -> generate anyway")
        return "generate"
    print("  [grade]    passages are not relevant -> rewrite")
    return "rewrite"


def rewrite(state):
    better = write(
        "A search of a shop's customer handbook found nothing useful for this question. Rewrite it as "
        "a short search query that uses the words such a handbook would use. Reply with the query only.\n\n"
        f"Question: {user_question(state)}"
    )
    print(f"  [rewrite]  {better!r}")
    return {"messages": [HumanMessage(f"That search found nothing relevant. Search again for: {better}")],
            "rewrites": state["rewrites"] + 1}


def generate(state):
    passages = state["messages"][-1].content
    answer = write(
        "Answer the customer's question using only the handbook passages below, in at most four "
        "sentences. If the passages do not contain the answer, say that the handbook does not cover it "
        "instead of guessing.\n\n"
        f"Question: {user_question(state)}\n\nPassages:\n{passages}"
    )
    print("  [generate] answer written")
    return {"messages": [AIMessage(answer)]}


graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("retrieve", ToolNode([search_handbook]))
graph.add_node("rewrite", rewrite)
graph.add_node("generate", generate)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
graph.add_conditional_edges("retrieve", grade_passages)
graph.add_edge("rewrite", "agent")
graph.add_edge("generate", END)
app = graph.compile()


# ---------------------------------------------------------------------------
# Run a few questions and check the behaviour by code
# ---------------------------------------------------------------------------
QUESTIONS = [
    # (question, should it search?, facts the answer must contain)
    ("How much does freight delivery cost for a kayak, and how long does it take?", True, ["149", "10"]),
    ("What values do Kestrel gift cards come in, and do they ever expire?", True, ["25", "500"]),
    ("Hi! What kinds of questions can you help me with?", False, []),
    ("Do you rent out tents or kayaks?", True, []),  # not in the handbook
]


def ask(question):
    """Run one question; return the nodes visited and the final answer."""
    path, answer = [], ""
    for update in app.stream({"messages": [("user", question)], "rewrites": 0}, stream_mode="updates"):
        for node, values in update.items():
            path.append(node)
            if values and values.get("messages"):
                answer = values["messages"][-1].content
    return path, answer


def main():
    print(f"Model: {MODEL} on {PROVIDER} | handbook: {len(chunks)} sections, top {TOP_K} retrieved\n")
    results = []
    with get_usage_metadata_callback() as usage:
        for question, should_search, facts in QUESTIONS:
            print(f"Q: {question}")
            path, answer = ask(question)
            searched = "retrieve" in path
            missing = [f for f in facts if f.lower() not in answer.lower()]
            passed = searched == should_search and not missing
            results.append(passed)
            print(f"  path:   {' -> '.join(path)}")
            print(f"  answer: {' '.join(answer.split())}")
            print(f"  check:  {'PASS' if passed else 'FAIL'} (searched={searched}, expected {should_search}"
                  + (f"; missing facts {missing}" if missing else "") + ")\n")

    tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
    print(f"{sum(results)}/{len(results)} questions behaved as expected | {tokens:,} tokens used")


if __name__ == "__main__":
    main()
