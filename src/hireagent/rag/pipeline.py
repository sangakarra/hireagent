"""
RAG pipeline entry point.

Ties together document loading, embedding, vector storage, and Claude API
into two simple functions:

  ingest_resume(pdf_path)  → load, chunk, embed, store
  query_resume(question)   → retrieve relevant chunks, ask Claude, return answer
"""

import hashlib
import anthropic
from dotenv import load_dotenv
from pathlib import Path
from typing import List

from .document_loader import load_and_chunk_pdf
from .embeddings import get_embedding_model
from .vector_store import create_vector_store, load_vector_store, query_vector_store

RESUME_HASH_FILE = Path("./data/resume.hash")

# Load ANTHROPIC_API_KEY (and any other vars) from the .env file.
# This must run before we create the Anthropic client.
load_dotenv()

# One shared Anthropic client for the whole module.
# It automatically reads ANTHROPIC_API_KEY from the environment.
client = anthropic.Anthropic()

# The system prompt tells Claude what role to play.
# We cache it (see query_resume) so repeated calls are cheaper —
# caching means the stable part is only billed at ~0.1× the normal rate.
SYSTEM_PROMPT = """You are an expert HR analyst who specialises in resume analysis \
and candidate evaluation. Answer questions about the candidate based solely on the \
resume excerpts provided. Be specific and cite relevant details from the text."""


def ingest_resume(pdf_path: str, collection_name: str = "resumes") -> str:
    """
    Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

    Run this once per resume (or whenever you add a new one).
    After this call, query_resume() can answer questions about the document.

    Args:
        pdf_path:        Path to the resume PDF file.
        collection_name: Name of the ChromaDB collection to store chunks in.

    Returns:
        SHA-256 hex digest of the PDF bytes — used as a cache key so analysis
        results can be invalidated automatically when the resume changes.
    """
    pdf_bytes = Path(pdf_path).read_bytes()
    resume_hash = hashlib.sha256(pdf_bytes).hexdigest()

    print(f"\n[1/3] Loading and chunking PDF: {pdf_path}")
    chunks = load_and_chunk_pdf(pdf_path)

    print("[2/3] Initialising embedding model...")
    embeddings = get_embedding_model()

    print("[3/3] Embedding chunks and storing in ChromaDB...")
    create_vector_store(chunks, embeddings, collection_name)

    # Persist the hash so run_analysis() can include it in its cache key.
    # Any change to the PDF (even a single byte) produces a different hash,
    # automatically invalidating cached results for the old resume.
    RESUME_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESUME_HASH_FILE.write_text(resume_hash)

    print(f"\nIngestion complete. {len(chunks)} chunks stored.\n")
    return resume_hash


def query_resume(
    question: str,
    collection_name: str = "resumes",
    top_k: int = 3,
) -> str:
    """
    RAG query pipeline: question → retrieve relevant chunks → Claude answer.

    Steps:
      1. Embed the question with the same model used during ingestion.
      2. Find the top-k most relevant resume chunks in ChromaDB.
      3. Build a prompt that gives Claude the retrieved context.
      4. Stream Claude's response and return the full answer text.

    Args:
        question:        Natural-language question about the candidate.
        collection_name: ChromaDB collection to search (must match ingest_resume).
        top_k:           Number of chunks to retrieve and pass to Claude.

    Returns:
        Claude's answer as a plain string.
    """
    # Step 1: Load the embedding model (same one used at ingest time).
    embeddings = get_embedding_model()

    # Step 2: Reconnect to the on-disk ChromaDB collection.
    store = load_vector_store(embeddings, collection_name)

    # Step 3: Semantic search — find the chunks most relevant to the question.
    retrieved_chunks = query_vector_store(store, question, top_k)

    # Step 4: Concatenate the retrieved chunks into a single context block.
    # The "---" separator helps Claude see where one chunk ends and another begins.
    context = "\n\n---\n\n".join(chunk.page_content for chunk in retrieved_chunks)

    # Step 5: Build the user turn.
    # The context comes from the vector store; the question comes from the caller.
    user_message = (
        f"Here are the most relevant excerpts from the candidate's resume:\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )

    # Step 6: Call Claude with prompt caching on the system prompt.
    #
    # cache_control on the system text tells Anthropic to cache that prefix.
    # On repeated calls the cached portion is served at ~0.1× the normal cost
    # instead of being re-processed from scratch each time.
    #
    # We stream the response and call get_final_message() at the end —
    # this gives us a complete Message object with usage stats.
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # Cache the stable system prompt across repeated queries
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        final = stream.get_final_message()

    # Extract the plain text from the first content block
    answer = next(
        (block.text for block in final.content if block.type == "text"),
        "",
    )
    return answer


def print_retrieved_chunks(question: str, collection_name: str = "resumes", top_k: int = 3) -> List:
    """Helper to inspect what the retrieval step actually finds before asking Claude."""
    embeddings = get_embedding_model()
    store = load_vector_store(embeddings, collection_name)
    chunks = query_vector_store(store, question, top_k)
    return chunks
