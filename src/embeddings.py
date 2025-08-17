import os, warnings

BACKEND = os.getenv("EMBEDDING_BACKEND", "ollama").lower()
MODEL_OVERRIDE = os.getenv("EMBEDDING_MODEL")

if BACKEND == "ollama":
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
        warnings.warn("Install langchain-ollama for updated OllamaEmbeddings.", DeprecationWarning)

    def get_embedding_function(model: str = None):
        model = model or MODEL_OVERRIDE or "nomic-embed-text"
        print(f"[embeddings] Using Ollama model: {model}")
        return OllamaEmbeddings(model=model)
else:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    def get_embedding_function(model: str = None):
        model = model or MODEL_OVERRIDE or "all-MiniLM-L6-v2"
        print(f"[embeddings] Using sentence-transformers model: {model}")
        return SentenceTransformerEmbeddings(model_name=model)