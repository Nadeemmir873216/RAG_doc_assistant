# identify project root directory
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


# -------------------------------
# imports
# -------------------------------

import streamlit as st
import os

from rag.ingest import load_pdf
from rag.chunking import chunk_documents
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from rag.citation import format_sources


# -------------------------------
# session state initialization
# -------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "generator" not in st.session_state:
    st.session_state.generator = Generator()


# -------------------------------
# UI
# -------------------------------

st.title("📄 Intelligent Document Assistant")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

question = st.text_input("Ask a question about the documents")


# -------------------------------
# detect file changes
# -------------------------------

current_file_names = [f.name for f in uploaded_files] if uploaded_files else []

if current_file_names != st.session_state.uploaded_file_names:

    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.uploaded_file_names = current_file_names


# -------------------------------
# create upload folder
# -------------------------------

os.makedirs("data/uploads", exist_ok=True)


# -------------------------------
# build vector store (only once)
# -------------------------------

if uploaded_files and st.session_state.vector_store is None:

    st.info("Processing documents...")

    all_docs = []

    for uploaded_file in uploaded_files:

        file_path = f"data/uploads/{uploaded_file.name}"

        # save temporarily
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # extract text
        docs = load_pdf(file_path)

        all_docs.extend(docs)

        # delete file immediately
        if os.path.exists(file_path):
            os.remove(file_path)

    # chunk documents
    chunks = chunk_documents(all_docs)

    # create embeddings
    embedder = EmbeddingModel()

    texts = [c["text"] for c in chunks]

    embeddings = embedder.embed(texts)

    # build vector store
    vector_store = VectorStore(len(embeddings[0]))

    vector_store.add(embeddings, chunks)

    # create retriever
    retriever = Retriever(embedder, vector_store)

    # store in session
    st.session_state.vector_store = vector_store
    st.session_state.retriever = retriever

    st.success("Documents indexed successfully!")


# -------------------------------
# question answering
# -------------------------------

if question and st.session_state.retriever:

    retriever = st.session_state.retriever
    generator = st.session_state.generator

    retrieved_chunks = retriever.retrieve(question)

    answer = generator.generate(question, retrieved_chunks)

    sources = format_sources(retrieved_chunks)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for s in sources:
        st.write(s)