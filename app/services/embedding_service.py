import google.generativeai as genai
from typing import List, Optional, Dict, Any
import asyncio
import time
import logging
import hashlib
import json
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.caches import InMemoryCache
from langchain.globals import set_llm_cache

try:
    from langchain_community.embeddings import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.config import settings
from ..utils.exceptions import EmbeddingServiceError

logger = logging.getLogger(__name__)

class LangChainEmbeddingService:
    """
    Enhanced embedding service using LangChain with Google Generative AI embeddings
    """
    
    def __init__(self):
        set_llm_cache(InMemoryCache())
        
        self.provider = getattr(settings, 'EMBEDDING_PROVIDER', 'google').lower()
        self._init_embeddings()
        
        self.requests_per_minute = 1500 if self.provider == "google" else 3000
        self.last_request_time = 0
        self.request_interval = 60.0 / self.requests_per_minute
        
        self.enable_cache = getattr(settings, 'ENABLE_EMBEDDING_CACHE', True)
        self.cache_dir = Path(settings.VECTOR_STORE_PATH) / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=getattr(settings, 'EMBEDDING_CHUNK_SIZE', 8000),
            chunk_overlap=200,
            length_function=len,
        )
        
        logger.info(f"Initialized LangChain embedding service with {self.provider} provider")
    
    def _init_embeddings(self):
        """Initialize embedding model based on provider"""
        try:
            if self.provider == "google":
                self._init_google_embeddings()
            elif self.provider == "openai":
                self._init_openai_embeddings()
            else:
                raise EmbeddingServiceError(f"Unsupported embedding provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} embeddings: {str(e)}")
            raise EmbeddingServiceError(f"Embedding initialization failed: {str(e)}")
    
    def _init_google_embeddings(self):
        """Initialize Google Generative AI embeddings"""
        if not settings.GOOGLE_API_KEY:
            raise EmbeddingServiceError("Google API key not configured")
        
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="retrieval_document"  # default task type
        )
        
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        
        logger.info(f"Initialized Google embeddings with model: {self.model}")
    
    def _init_openai_embeddings(self):
        """Initialize OpenAI embeddings as fallback"""
        if not OPENAI_AVAILABLE:
            raise EmbeddingServiceError("OpenAI package not installed")
        
        if not settings.OPENAI_API_KEY:
            raise EmbeddingServiceError("OpenAI API key not configured")
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
        )
        
        self.model = getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
        self.dimension = getattr(settings, 'OPENAI_EMBEDDING_DIMENSION', 1536)
        
        logger.info(f"Initialized OpenAI embeddings with model: {self.model}")
    
    async def _rate_limit(self):
        """Rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            sleep_time = self.request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, text: str, task_type: str = "document") -> str:
        """Generate cache key for embedding"""
        content = f"{text}:{task_type}:{self.model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache"""
        if not self.enable_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {str(e)}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """Save embedding to cache"""
        if not self.enable_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {str(e)}")
    
    def _prepare_text(self, text: str) -> str:
        """Clean and prepare text for embedding"""
        if not text:
            return ""
        
        cleaned = text.strip()
        
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        max_length = 8000 if self.provider == "google" else 8191  # conservative limits
        if len(cleaned) > max_length:
            logger.warning(f"Text length {len(cleaned)} exceeds limit, truncating to {max_length}")
            cleaned = cleaned[:max_length]
        
        return cleaned
    
    async def generate_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Generate embeddings for a list of texts using LangChain"""
        try:
            if not texts:
                return []
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            prepared_texts = [self._prepare_text(text) for text in texts]
            
            valid_texts = [(i, text) for i, text in enumerate(prepared_texts) if text]
            
            if not valid_texts:
                logger.warning("No valid texts found after preparation")
                return []
            
            embeddings_result = [[] for _ in range(len(texts))]  # Initialize with empty lists
            
            batch_size = 5 if self.provider == "google" else 100
            
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                batch_texts = [text for _, text in batch]
                batch_indices = [idx for idx, _ in batch]
                
                await self._rate_limit()
                
                try:
                    # check cache first
                    cached_embeddings = []
                    uncached_texts = []
                    uncached_indices = []
                    
                    for j, (original_idx, text) in enumerate(batch):
                        cache_key = self._get_cache_key(text, task_type)
                        cached = self._load_from_cache(cache_key)
                        
                        if cached:
                            cached_embeddings.append((original_idx, cached))
                        else:
                            uncached_texts.append(text)
                            uncached_indices.append((j, original_idx))
                    
                    if uncached_texts:
                        if self.provider == "google":
                            batch_embeddings = []
                            for text in uncached_texts:
                                embedding = await self._generate_google_embedding(text, task_type)
                                batch_embeddings.append(embedding)
                        else:
                            batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                                None, self.embeddings.embed_documents, uncached_texts
                            )
                        
                        for (batch_idx, original_idx), embedding in zip(uncached_indices, batch_embeddings):
                            cache_key = self._get_cache_key(uncached_texts[batch_idx], task_type)
                            self._save_to_cache(cache_key, embedding)
                            embeddings_result[original_idx] = embedding
                    
                    for original_idx, embedding in cached_embeddings:
                        embeddings_result[original_idx] = embedding
                
                except Exception as e:
                    logger.error(f"Failed to process batch {i//batch_size + 1}: {str(e)}")
                    for idx in batch_indices:
                        if not embeddings_result[idx]:
                            embeddings_result[idx] = []
            
            valid_embeddings = [emb for emb in embeddings_result if emb]
            
            logger.info(f"Successfully generated {len(valid_embeddings)} embeddings")
            return valid_embeddings
        
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise EmbeddingServiceError(f"Failed to generate embeddings: {str(e)}")
    
    async def _generate_google_embedding(self, text: str, task_type: str) -> List[float]:
        """Generate single embedding with Google AI"""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type=task_type
            )
            
            logger.debug(f"Google embedding raw response: {result}")
            
            if result and "embedding" in result:
                return result["embedding"]
            else:
                raise EmbeddingServiceError(f"Invalid response from Google embedding API: {result}")
        except Exception as e:
            logger.error(f"Google embedding API call failed: {e}")
            raise

    
    async def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embedding for a single text"""
        try:
            if not text or not text.strip():
                return []
            
            if task_type == "retrieval_document" and len(text) < 200:
                task_type = "retrieval_query"
            
            embeddings = await self.generate_embeddings([text], task_type)
            return embeddings[0] if embeddings else []
        
        except Exception as e:
            logger.error(f"Failed to generate single embedding: {str(e)}")
            raise EmbeddingServiceError(f"Failed to generate embedding: {str(e)}")
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding specifically optimized for queries"""
        return await self.generate_embedding(query, task_type="retrieval_query")
    
    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Embed LangChain Document objects"""
        try:
            texts = [doc.page_content for doc in documents]
            return await self.generate_embeddings(texts, task_type="retrieval_document")
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise EmbeddingServiceError(f"Failed to embed documents: {str(e)}")
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a query string"""
        return await self.generate_query_embedding(query)
    
    def create_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[Document]:
        """Create LangChain Document objects from texts"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        documents = []
        for text, metadata in zip(texts, metadatas):
            if len(text) > 8000:
                chunks = self.text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_length': len(text)
                    }
                    documents.append(Document(page_content=chunk, metadata=chunk_metadata))
            else:
                documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.dimension
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the embedding provider"""
        return {
            'provider': self.provider,
            'model': self.model,
            'dimension': self.dimension,
            'cache_enabled': self.enable_cache,
            'rate_limit_rpm': self.requests_per_minute,
            'max_text_length': 8000 if self.provider == "google" else 8191
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the embedding service connection"""
        try:
            test_text = "This is a test embedding."
            start_time = time.time()
            
            embedding = await self.generate_embedding(test_text)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'provider': self.provider,
                'model': self.model,
                'response_time_ms': response_time,
                'embedding_dimension': len(embedding) if embedding else 0,
                'test_embedding_length': len(embedding) if embedding else 0
            }
        except Exception as e:
            logger.error(f"Embedding service test failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'provider': self.provider,
                'error': str(e)
            }
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear the embedding cache"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            deleted_count = 0
            
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")
            
            logger.info(f"Cleared {deleted_count} cache files")
            return {
                'status': 'success',
                'deleted_files': deleted_count,
                'total_files': len(cache_files)
            }
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

# backward compatibility alias
EmbeddingService = LangChainEmbeddingService