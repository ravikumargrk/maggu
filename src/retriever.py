from typing import List, Tuple
from langchain.schema import Document
from .vector_store import VectorStore

class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    def build_context(self, results: List[Tuple[Document, float]]) -> str:
        parts = []
        for i, (doc, score) in enumerate(results, 1):
            parts.append(
                f"=== Chunk {i} | score={score:.4f} | chunk_id={doc.metadata.get('chunk_id')} ===\n"
                f"{doc.page_content.strip()}"
            )
        return "\n\n".join(parts)