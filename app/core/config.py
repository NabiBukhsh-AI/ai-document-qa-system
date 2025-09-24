import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Aegasis Labs - Document QA"
    VERSION: str = "1.0.0"
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "nabibuksh.baloch01@gmail.com12345678")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 768
    
    GEMINI_MODEL: str = "gemini-2.5-flash"
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_EMBEDDING_DIMENSION: int = 1536
    
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "google")
    
    ENABLE_EMBEDDING_CACHE: bool = True
    EMBEDDING_CHUNK_SIZE: int = 8000
    EMBEDDING_BATCH_SIZE: int = 5
    
    VECTOR_STORE_PATH: str = "data/vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".txt", ".pdf", ".md"]
    UPLOAD_DIR: str = "uploads"
    
    LANGCHAIN_CACHE_TYPE: str = "memory"  # "memory", "redis", "sqlite"
    LANGCHAIN_VERBOSE: bool = False
    
    GOOGLE_EMBEDDINGS_RPM: int = 1500
    OPENAI_EMBEDDINGS_RPM: int = 3000
    
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    class Config:
        env_file = ".env"

settings = Settings()