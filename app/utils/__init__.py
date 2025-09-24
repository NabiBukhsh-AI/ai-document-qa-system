from .document_processor import DocumentProcessor
from .exceptions import (
    DocumentQAException,
    DocumentNotFoundError,
    EmbeddingServiceError,
    LLMServiceError,
    VectorStoreError,
)

PROCESSORS = {
    "document": DocumentProcessor,
}

EXCEPTIONS = {
    "base": DocumentQAException,
    "not_found": DocumentNotFoundError,
    "embedding": EmbeddingServiceError,
    "llm": LLMServiceError,
    "vector_store": VectorStoreError,
}

__all__ = [
    "DocumentProcessor",
    "DocumentQAException",
    "DocumentNotFoundError", 
    "EmbeddingServiceError",
    "LLMServiceError",
    "VectorStoreError",
    "PROCESSORS",
    "EXCEPTIONS",
]