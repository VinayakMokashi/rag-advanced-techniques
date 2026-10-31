"""
How chunk size changes what retrieval returns.

The same short document is split at three chunk sizes (50, 150 and 400 characters,
each with 10% overlap) and indexed in its own Chroma collection. One query then
runs against all three so you can compare what comes back: tiny chunks match
tightly but carry little information, while large chunks bring more context along
with the text you asked for.

How to run
----------
    python rag11.py

The first run downloads the embedding model (~90 MB) and caches it.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENT_TEXT = """
The human heart is a muscular organ that pumps blood through the circulatory system.
This blood delivers oxygen and nutrients to the body and removes metabolic waste. The heart
has four chambers: the right atrium, the right ventricle, the left atrium, and the left ventricle.
The average adult heart beats about 60 to 100 times per minute. The circulatory system consists of the heart,
blood vessels, and blood. It is a vital part of the human body, responsible for transporting essential substances.
A healthy lifestyle, including regular exercise and a balanced diet, is crucial for cardiovascular health.
"""

QUERY = "What is the primary function of the heart and its connection to the circulatory system?"
CHUNK_SIZES = [50, 150, 400]
K = 3

# Shared embedding model (384 dimensions, runs locally).
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [Document(page_content=DOCUMENT_TEXT, metadata={"source": "heart_document"})]

# One Chroma collection per chunk size, so the results never mix.
vector_dbs = {}
for size in CHUNK_SIZES:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=int(size * 0.1))
    chunks = text_splitter.split_documents(documents)
    vector_dbs[size] = Chroma.from_documents(chunks, embeddings, collection_name=f"heart_chunks_{size}")
    print(f"Chunk size {size}: {len(chunks)} chunks")

print(f"\nQuery: {QUERY}")

for size, db in vector_dbs.items():
    print(f"\n{'-' * 30}\nRetrieving with chunk size {size} (top {K}):")
    for rank, doc in enumerate(db.similarity_search(QUERY, k=K), 1):
        print(f"  {rank}. {' '.join(doc.page_content.split())}")
