import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.schema import Document

class DocumentLoader:
    """ Handles loading different types of documents and web content"""

    def __init__(self, uploads_dir: str= "./data/uploads/"):
        self.uploads_dir= uploads_dir
        self.supported_formats = [".pdf", ".txt"]
        os.makedirs(uploads_dir, exist_ok= True)
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF documents"""
        loader= PyPDFLoader(file_path)
        return loader.load()
    
    def load_text(self, file_path: str) -> List[Document]:
        """Load text documents"""
        loader= TextLoader(file_path)
        return loader.load()
    
    def load_web(self, url:str) -> List[Document]:
        """Load web content"""
        loader= WebBaseLoader(url)
        return loader.load()

    def is_supported_format(self, file_path:str) -> bool:
        """Check if the file format is supported"""
        _, ext = os.path.splitext(file_path)
        return ext in self.supported_formats
