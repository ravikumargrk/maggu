from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

def load_documents(source_path: str) -> List[Document]:
    """
    Load a .pdf or .txt file into Documents with basic metadata.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source path {source_path} does not exist.")
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif ext == ".txt":
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    for d in docs:
        d.metadata.setdefault("filename", path.name)
        d.metadata.setdefault("file_type", ext.lstrip("."))
    return docs