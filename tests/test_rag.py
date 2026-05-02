"""
Manual integration test for the RAG pipeline.

How to run:
    1. Copy a resume PDF into this project (e.g. copy Sangarshan_resume.pdf
       from the parent folder):
         cp ~/Desktop/sangarshan/Sangarshan_resume.pdf \
            ~/Desktop/sangarshan/hireagent/data/sample_resume.pdf

    2. Make sure .env contains your ANTHROPIC_API_KEY.

    3. From the project root:
         pip install -r requirements.txt
         python tests/test_rag.py
"""

import sys
import os

# Allow running the script directly from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hireagent.rag.pipeline import ingest_resume, query_resume, print_retrieved_chunks

# ── Config ────────────────────────────────────────────────────────────────────

# Path to the sample resume PDF.
# Adjust if your PDF lives somewhere else.
SAMPLE_PDF = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_resume.pdf"
)

QUESTION = "What programming languages does this person know?"

# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Guard: make sure the PDF exists before we start
    if not os.path.exists(SAMPLE_PDF):
        print(
            f"\n[ERROR] Sample PDF not found at:\n  {os.path.abspath(SAMPLE_PDF)}\n\n"
            "Please copy a resume PDF there, e.g.:\n"
            "  cp ~/Desktop/sangarshan/Sangarshan_resume.pdf "
            f"{os.path.abspath(SAMPLE_PDF)}\n"
        )
        sys.exit(1)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    separator("STEP 1 — Ingest resume into ChromaDB")
    ingest_resume(SAMPLE_PDF)

    # ── Step 2: Show retrieved chunks ─────────────────────────────────────────
    separator("STEP 2 — Retrieved chunks (before asking Claude)")
    print(f"Question: {QUESTION}\n")

    chunks = print_retrieved_chunks(QUESTION, top_k=3)
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", "?")
        print(f"--- Chunk {i} (source: {os.path.basename(source)}, page {page}) ---")
        print(chunk.page_content)
        print()

    # ── Step 3: Ask Claude ────────────────────────────────────────────────────
    separator("STEP 3 — Claude's answer")
    print(f"Question: {QUESTION}\n")

    answer = query_resume(QUESTION)
    print(answer)
    separator("Done")


if __name__ == "__main__":
    main()
