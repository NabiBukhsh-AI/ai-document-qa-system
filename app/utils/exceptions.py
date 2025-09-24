from fastapi import HTTPException
from typing import Any, Dict, Optional

class DocumentQAException(Exception):
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class DocumentNotFoundError(DocumentQAException):
    def __init__(self, doc_id: str):
        super().__init__(f"Document with ID '{doc_id}' not found", 404)

class EmbeddingServiceError(DocumentQAException):
    def __init__(self, message: str = "Failed to generate embeddings"):
        super().__init__(message, 500)

class LLMServiceError(DocumentQAException):
    def __init__(self, message: str = "Failed to generate answer from LLM"):
        super().__init__(message, 500)

class VectorStoreError(DocumentQAException):
    def __init__(self, message: str = "Vector store operation failed"):
        super().__init__(message, 500)