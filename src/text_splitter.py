import uuid
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._rebuild()

    def _rebuild(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def update(self, chunk_size: int = None, chunk_overlap: int = None):
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        self._rebuild()

    def split(self, docs: List[Document]) -> List[Document]:
        chunks = self._splitter.split_documents(docs)
        for c in chunks:
            # stable fields
            c.metadata.setdefault("chunk_id", str(uuid.uuid4()))
            c.metadata.setdefault("source", c.metadata.get("filename"))
            c.metadata["chunk_size"] = len(c.page_content)
        return chunks