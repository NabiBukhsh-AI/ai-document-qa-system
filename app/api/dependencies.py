from functools import lru_cache
from ..services.document_service import DocumentService
from ..services.llm_service import LLMService
from ..services.vector_store_service import VectorStoreService

@lru_cache()
def get_document_service() -> DocumentService:
    return DocumentService()

@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()

@lru_cache()
def get_vector_store_service() -> VectorStoreService:
    return VectorStoreService()