import os
from typing import List, Tuple, Optional
from langchain.schema import Document
from .embeddings import get_embedding_function

try:
    from langchain_chroma import Chroma            # new package (no .persist())
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback (has .persist())

class VectorStore:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str | None = None
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._emb = get_embedding_function(embedding_model)
        self._store: Optional[Chroma] = None

    def _ensure_store(self):
        if self._store is None:
            self._store = Chroma(
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
                embedding_function=self._emb
            )

    def _maybe_persist(self):
        # Older Chroma (community) exposes persist(); new langchain_chroma auto-persists.
        if hasattr(self._store, "persist"):
            try:
                self._store.persist()
            except Exception:
                pass  # ignore persistence edge cases

    def add_documents(self, docs: List[Document]):
        if not docs:
            print("[add] No docs to add.")
            return
        self._ensure_store()
        ids = [d.metadata.get("chunk_id") for d in docs]
        self._store.add_documents(docs, ids=ids)
        self._maybe_persist()

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        self._ensure_store()
        return self._store.similarity_search_with_score(query, k=k)

    def count(self) -> int:
        self._ensure_store()
        return self._store._collection.count()

    def stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "count": self.count()
        }