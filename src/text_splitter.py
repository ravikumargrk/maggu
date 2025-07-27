from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class TextSplitter:
    """Handles splitting documents into smaller chunks"""
    
    def __init__(self, chunk_size: int= 1000, chunk_overlap: int= 200):
        self.chunksize = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size= self.chunk_size,
            chunk_overlap= self.chunk_overlap,
            length_function= len,
            is_separator_regex= False,
        )
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        return self.splitter.split_documents(documents)

    def split_text(self, text: str) -> List[Document]:
        """Split a single text into smaller chunks"""
        return self.splitter.create_documents([text])
        