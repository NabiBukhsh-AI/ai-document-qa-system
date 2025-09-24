from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DocumentResponse(BaseModel):
    id: str
    title: str
    content_preview: str = Field(...)
    chunk_count: int
    created_at: datetime
    metadata: Dict[str, Any]

class DocumentListResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    chunk_count: int

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    max_chunks: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class SourceDocument(BaseModel):
    doc_id: str
    title: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    response_time_ms: int

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class User(BaseModel):
    username: str