from .dependencies import (
    get_document_service,
    get_llm_service,
    get_vector_store_service,
)

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

TAGS_METADATA = [
    {
        "name": "documents",
        "description": "Document management operations. Upload, index, and manage documents.",
    },
    {
        "name": "query", 
        "description": "Question answering operations. Search documents and generate answers.",
    },
    {
        "name": "auth",
        "description": "Authentication operations. Get access tokens and manage sessions.",
    },
    {
        "name": "health",
        "description": "Health check and system status operations.",
    },
]

__all__ = [
    "get_document_service",
    "get_llm_service", 
    "get_vector_store_service",
    "API_VERSION",
    "API_PREFIX",
    "TAGS_METADATA",
]