from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF resume and split it into overlapping text chunks.

    Why chunks? Language models and vector databases work best with focused,
    bite-sized pieces of text rather than one giant blob. Overlapping chunks
    (chunk_overlap) ensures we don't accidentally cut a sentence in half and
    lose meaning at the boundaries.

    Args:
        pdf_path: Absolute or relative path to the resume PDF.

    Returns:
        A list of LangChain Document objects, each holding one chunk of text
        plus metadata (source file, page number) automatically added by
        PyPDFLoader.
    """
    # Step 1: Load the PDF.
    # PyPDFLoader reads the file page-by-page and returns one Document per page.
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Step 2: Split pages into smaller, overlapping chunks.
    # chunk_size=500  → each chunk is at most 500 characters
    # chunk_overlap=50 → consecutive chunks share 50 characters so context
    #                    isn't lost at chunk boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        # Try to split on paragraph breaks first, then sentences, then words
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    print(f"Loaded {len(pages)} page(s) → split into {len(chunks)} chunks.")
    return chunks
