from langchain_huggingface import HuggingFaceEmbeddings


# The model name for the free, local HuggingFace embedding model.
# "all-MiniLM-L6-v2" is small (80 MB), fast, and produces 384-dimensional
# vectors. It's a great starting point — no API key required.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return a HuggingFace sentence-transformer embedding model.

    Embeddings convert text into a list of numbers (a vector) that captures
    the *meaning* of the text. Sentences with similar meaning end up close
    together in vector space, which is what makes semantic search possible.

    The model is downloaded from HuggingFace Hub on first use and cached
    locally in ~/.cache/huggingface/ for subsequent runs.

    Returns:
        A LangChain-compatible HuggingFaceEmbeddings object.
    """
    # model_kwargs: passed directly to the underlying SentenceTransformer.
    # device="cpu" works everywhere; swap to "cuda" if you have a GPU.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        # encode_kwargs control how batches of text are embedded
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings
