from langchain_community.embeddings.ollama import OllamaEmbeddings

class EmbeddingModel:
    """Handles vector embeddings for documents and queries"""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        """ 
        Initialize the embedding model
        Args:
        model_name : Ollama model to use for embeddings Default is nomic-embed-text 
        """
       self.model_name = model_name
       self.embeddings = OllamaEmbeddings(model= model_name)

       def get_embeddings(self):
        """Return the embedding model for use with vector stores"""
        return self.embeddings
