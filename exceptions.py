class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass

class EmbeddingError(RAGError):
    """Raised when there's an error generating embeddings."""
    pass

class PDFProcessingError(RAGError):
    """Raised when there's an error processing PDF files."""
    pass

class ChromaDBError(RAGError):
    """Raised when there's an error with ChromaDB operations."""
    pass

class ModelGenerationError(RAGError):
    """Raised when there's an error generating responses from the model."""
    pass 