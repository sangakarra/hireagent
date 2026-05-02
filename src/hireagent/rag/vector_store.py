import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

# Where ChromaDB will write its files on disk.
# Using a path relative to the project root keeps everything self-contained.
CHROMA_PERSIST_DIR = "./data/chroma_db"


def create_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str = "resumes",
) -> Chroma:
    """
    Embed a list of Document chunks and store them in a ChromaDB collection.

    ChromaDB is a local vector database — think of it like SQLite but for
    embeddings. It stores each chunk alongside its vector so we can later
    search by semantic similarity rather than exact keywords.

    The collection is automatically persisted to CHROMA_PERSIST_DIR, so
    the data survives between Python sessions.

    Args:
        chunks:          Text chunks from document_loader.load_and_chunk_pdf().
        embeddings:      The embedding model from embeddings.get_embedding_model().
        collection_name: Logical name for this group of documents.

    Returns:
        The populated Chroma vector store object.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    # Delete the existing collection before re-creating it. Without this,
    # Chroma.from_documents() appends to the collection on every ingest call,
    # causing duplicate chunks to accumulate. Duplicate chunks mean the top-k
    # RAG query returns a different mix on each run → non-deterministic scores
    # even when the resume and job description haven't changed.
    _db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        _db.delete_collection(collection_name)
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    print(f"Stored {len(chunks)} chunks in collection '{collection_name}'.")
    return vector_store


def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
    collection_name: str = "resumes",
) -> Chroma:
    """
    Load an existing ChromaDB collection from disk.

    Call this after you've already ingested at least one resume.
    It reconnects to the on-disk data without re-embedding anything.

    Args:
        embeddings:      Must be the *same* model used during create_vector_store().
        collection_name: Must match the name used during create_vector_store().

    Returns:
        A ready-to-query Chroma vector store object.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


def query_vector_store(
    vector_store: Chroma,
    query: str,
    top_k: int = 3,
) -> List[Document]:
    """
    Find the top-k most semantically similar chunks for a given query.

    Under the hood this embeds the query text, then computes cosine similarity
    between the query vector and every stored chunk vector, returning the
    closest matches.

    Args:
        vector_store: A loaded or freshly created Chroma store.
        query:        The natural-language question to search for.
        top_k:        Number of chunks to return (more = more context, more tokens).

    Returns:
        A list of Document objects ordered by relevance (most relevant first).
    """
    # similarity_search embeds the query and returns the nearest neighbours
    results = vector_store.similarity_search(query, k=top_k)
    return results
