"""
PDF chat application -- conversational RAG over your own PDFs.

Upload one or more PDF files through a Gradio web UI; the app splits them into
chunks, embeds the chunks into a persistent Chroma vector store, and then lets you
chat with their contents. Each question is answered by gpt-oss-120b, hosted on Groq,
using only the most relevant chunks retrieved from the PDFs, and a conversation
memory keeps track of the dialogue so follow-up questions work.

How to run
----------
    python rag05.py

A browser tab opens automatically. Upload PDFs, click "Process PDFs", then ask
questions in the chat box. Stop the server with Ctrl+C.

Requirements
------------
Set a Groq API key in a ``.env`` file next to this script (see ``.env.example``):

    GROQ_API_KEY=your_key_here

The first run also downloads the embedding model (~400 MB) and caches it.
"""

import os

from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# Load GROQ_API_KEY (and any other variables) from a local .env file.
load_dotenv()


class PDFChatApp:
    """A small conversational-RAG app: ingest PDFs, then chat with them."""

    def __init__(self, db_name="pdf_db"):
        # Groq-hosted model used to answer questions.
        self.model = "openai/gpt-oss-120b"
        # Folder where the Chroma vector store is persisted on disk.
        self.db_name = db_name

        self.documents = []             # raw pages loaded from the PDFs
        self.chunks = []                # the pages split into smaller chunks
        self.vectorstore = None         # Chroma index of the chunk embeddings
        self.conversation_chain = None  # retriever + LLM + memory chain

    def upload_pdf(self, pdf_files):
        """Load, chunk, embed and index the uploaded PDFs.

        Called when the user clicks "Process PDFs". ``pdf_files`` is whatever
        Gradio's File component passes -- either file paths (strings) or file
        objects that carry the path on a ``.name`` attribute.
        """
        print(f"Processing {len(pdf_files)} PDF file(s)...")
        self.documents = []
        for pdf_file in pdf_files:
            # Gradio may hand us either a path string or a temp-file object.
            if type(pdf_file) is str:
                file_path = pdf_file
            else:
                file_path = pdf_file.name
            loader = PyPDFLoader(file_path)
            # loader.load() returns one Document per page; tag each with metadata.
            self.documents.extend(self._add_metadata(doc, "pdf") for doc in loader.load())

        self.chunks = self.split_documents()
        self.vectorstore = self.create_vectorstore()
        self.conversation_chain = self.create_conversation_chain()

        summary = f"Uploaded {len(pdf_files)} PDF files and processed {len(self.documents)} documents into {len(self.chunks)} chunks."
        print(f"  {summary}")
        return summary

    def _add_metadata(self, doc, doc_type):
        """Attach a ``doc_type`` tag to a Document's metadata and return it."""
        doc.metadata["doc_type"] = doc_type
        return doc

    def split_documents(self, chunk_size=1000, chunk_overlap=200):
        """Split the loaded pages into overlapping chunks for retrieval."""
        if not self.documents:
            return []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(self.documents)
        return chunks

    def create_vectorstore(self):
        """Embed the chunks and persist them in a Chroma vector store.

        If a store already exists at ``self.db_name`` its collection is deleted
        first, so re-uploading starts from a clean index instead of mixing in
        chunks from a previous session.
        """
        if not self.chunks:
            return None

        embeddings = HuggingFaceEmbeddings()
        # Clear any previous collection so old PDFs don't linger in the index.
        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, embedding_function=embeddings).delete_collection()

        vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=self.db_name
        )
        return vectorstore

    def create_conversation_chain(self, k=4):
        """Build the conversational-retrieval chain (retriever + LLM + memory).

        ``k`` is the number of chunks retrieved per question. The
        ConversationBufferMemory lets the chain remember previous turns, so
        follow-up questions ("and what about X?") are understood in context.
        """
        if not self.vectorstore:
            return None

        # gpt-oss counts its hidden reasoning against max_tokens; without an explicit
        # limit, long answers can be cut off and come back empty.
        llm = ChatGroq(model=self.model, max_tokens=2048, reasoning_effort="low", max_retries=6)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        chain_args = {
            "llm": llm,
            "retriever": retriever,
            "memory": memory,
        }

        return ConversationalRetrievalChain.from_llm(**chain_args)

    def query(self, question):
        """Answer a single question using the retrieval chain."""
        if not self.conversation_chain:
            return "Please upload PDF files first."
        print(f"Q: {question}")
        result = self.conversation_chain.invoke({"question": question})
        print(f"A: {result['answer']}")
        return result["answer"]

    def chat_handler(self, message, history):
        """Adapter that lets Gradio's ChatInterface call ``query``.

        ``history`` is supplied by Gradio (the past messages), but the
        conversation chain tracks its own memory, so we only need ``message``.
        """
        return self.query(message)

    def launch_app(self):
        """Build the Gradio UI and launch the web server."""
        with gr.Blocks() as app:
            with gr.Row():
                gr.Markdown("# PDF Chat Application")

            # --- Upload / processing row ---
            with gr.Row():
                pdf_files = gr.File(file_count="multiple", label="Upload PDF Files")
                upload_button = gr.Button("Process PDFs")
                status_text = gr.Textbox(label="Status")

            upload_button.click(
                fn=self.upload_pdf,
                inputs=[pdf_files],
                outputs=[status_text]
            )

            # --- Chat interface ---
            chatbot = gr.ChatInterface(
                fn=self.chat_handler,
                examples=["What's in these documents?", "Summarize the key points"],
                title="Chat with your PDFs"
            )

        print("Launching PDF Chat App... a browser tab will open (press Ctrl+C to stop).")
        return app.launch(inbrowser=True)


if __name__ == "__main__":
    app = PDFChatApp()
    app.launch_app()
