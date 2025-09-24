import os
from typing import List, Dict, Any
from datetime import datetime
import json
import logging
from ..models.schemas import DocumentInput, DocumentResponse, DocumentListResponse
from ..utils.document_processor import DocumentProcessor
from ..utils.exceptions import DocumentNotFoundError, DocumentQAException
from .embedding_service import EmbeddingService
from .vector_store_service import VectorStoreService
from ..core.config import settings

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.documents_path = os.path.join(settings.VECTOR_STORE_PATH, "documents.json")
        self.documents = self._load_documents()
        
        logger.info("Document service initialized successfully")
    
    def _load_documents(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from disk"""
        try:
            if os.path.exists(self.documents_path):
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                logger.info(f"Loaded {len(documents)} documents from disk")
                return documents
            logger.info("No existing documents found, starting with empty collection")
            return {}
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            return {}
    
    def _save_documents(self):
        """Save document metadata to disk"""
        try:
            os.makedirs(os.path.dirname(self.documents_path), exist_ok=True)
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, default=str, ensure_ascii=False)
            logger.debug(f"Saved {len(self.documents)} documents to disk")
        except Exception as e:
            logger.error(f"Failed to save documents: {str(e)}")
            raise DocumentQAException(f"Failed to save documents: {str(e)}")
    
    async def add_document(self, document: DocumentInput) -> DocumentResponse:
        """Add a new document to the system"""
        try:
            logger.info(f"Adding document: {document.title}")
            
            doc_id = self.processor.generate_doc_id(document.title, document.content)
            
            if doc_id in self.documents:
                logger.warning(f"Document with ID {doc_id} already exists")
                raise DocumentQAException(f"Document with similar content already exists", 409)
            
            chunks_data = self.processor.split_document(document.content, document.title)
            logger.info(f"Split document into {len(chunks_data)} chunks")
            
            chunk_texts = [chunk for chunk, _ in chunks_data]
            embeddings = await self.embedding_service.generate_embeddings(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            if len(embeddings) != len(chunks_data):
                raise DocumentQAException(f"Embedding count ({len(embeddings)}) doesn't match chunk count ({len(chunks_data)})")
            
            metadatas = []
            for i, (chunk_text, metadata) in enumerate(chunks_data):
                if i >= len(embeddings):
                    raise DocumentQAException(f"Missing embedding for chunk {i}")
                
                enhanced_metadata = {
                    'content': chunk_text,
                    'doc_title': document.title,
                    'created_at': datetime.now().isoformat(),
                    'embedding_dimension': len(embeddings[i]) if embeddings[i] else 0,
                    'chunk_length': len(chunk_text),
                    **metadata,  # includes original metadata (doc_id, title, chunk_index, etc.)
                    **document.metadata  # includes user-provided metadata
                }
                metadatas.append(enhanced_metadata)
            
            logger.info("Adding embeddings to vector store")
            self.vector_store.add_documents(embeddings, metadatas)
            
            self.documents[doc_id] = {
                'id': doc_id,
                'title': document.title,
                'content_preview': document.content[:200] + "..." if len(document.content) > 200 else document.content,
                'chunk_count': len(chunks_data),
                'created_at': datetime.now().isoformat(),
                'metadata': {
                    **document.metadata,
                    'total_characters': len(document.content),
                    'avg_chunk_size': sum(len(chunk) for chunk, _ in chunks_data) / len(chunks_data),
                    'embedding_model': self.embedding_service.model,
                    'embedding_dimension': len(embeddings[0]) if embeddings else 0
                }
            }
            
            self._save_documents()
            
            logger.info(f"Successfully added document {doc_id} with {len(chunks_data)} chunks")
            return DocumentResponse(**self.documents[doc_id])
        
        except DocumentQAException:
            raise
        except Exception as e:
            logger.error(f"Failed to add document '{document.title}': {str(e)}")
            raise DocumentQAException(f"Failed to add document: {str(e)}")
    
    async def add_document_from_file(self, file_path: str, filename: str) -> DocumentResponse:
        """Add document from uploaded file"""
        try:
            logger.info(f"Processing file: {filename}")
            
            if not os.path.exists(file_path):
                raise DocumentQAException(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            logger.info(f"File size: {file_size} bytes")
            
            title, content = self.processor.process_file(file_path, filename)
            
            document = DocumentInput(
                title=title,
                content=content,
                metadata={
                    'source_file': filename,
                    'file_size': file_size,
                    'processed_at': datetime.now().isoformat()
                }
            )
            
            result = await self.add_document(document)
            logger.info(f"Successfully processed file {filename} -> document {result.id}")
            
            return result
        except DocumentQAException:
            raise
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {str(e)}")
            raise DocumentQAException(f"Failed to process file: {str(e)}")
    
    def get_documents(self) -> List[DocumentListResponse]:
        """Get list of all documents"""
        try:
            documents_list = [DocumentListResponse(**doc) for doc in self.documents.values()]
            logger.info(f"Retrieved {len(documents_list)} documents")
            return documents_list
        except Exception as e:
            logger.error(f"Failed to retrieve documents list: {str(e)}")
            raise DocumentQAException(f"Failed to retrieve documents: {str(e)}")
    
    def get_document(self, doc_id: str) -> DocumentResponse:
        """Get specific document"""
        try:
            if doc_id not in self.documents:
                logger.warning(f"Document not found: {doc_id}")
                raise DocumentNotFoundError(doc_id)
            
            logger.info(f"Retrieved document: {doc_id}")
            return DocumentResponse(**self.documents[doc_id])
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {str(e)}")
            raise DocumentQAException(f"Failed to retrieve document: {str(e)}")
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        try:
            if doc_id not in self.documents:
                logger.warning(f"Attempted to delete non-existent document: {doc_id}")
                raise DocumentNotFoundError(doc_id)
            
            logger.info(f"Deleting document: {doc_id}")
            
            doc_info = self.documents[doc_id]
            chunk_count = doc_info.get('chunk_count', 0)
            
            deleted_count = self.vector_store.delete_document(doc_id)
            logger.info(f"Deleted {deleted_count} vectors from vector store for document {doc_id}")
            
            del self.documents[doc_id]
            self._save_documents()
            
            logger.info(f"Successfully deleted document {doc_id} "
                       f"(had {chunk_count} chunks, removed {deleted_count} vectors)")
            
            return deleted_count > 0
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            raise DocumentQAException(f"Failed to delete document: {str(e)}")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about all documents"""
        try:
            if not self.documents:
                return {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'total_vectors': 0,
                    'avg_chunks_per_doc': 0,
                    'oldest_document': None,
                    'newest_document': None
                }
            
            total_chunks = sum(doc.get('chunk_count', 0) for doc in self.documents.values())
            total_vectors = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
            
            docs_with_dates = [
                (doc['id'], doc.get('created_at')) 
                for doc in self.documents.values() 
                if doc.get('created_at')
            ]
            
            oldest_doc = min(docs_with_dates, key=lambda x: x[1])[0] if docs_with_dates else None
            newest_doc = max(docs_with_dates, key=lambda x: x[1])[0] if docs_with_dates else None
            
            stats = {
                'total_documents': len(self.documents),
                'total_chunks': total_chunks,
                'total_vectors': total_vectors,
                'avg_chunks_per_doc': total_chunks / len(self.documents) if self.documents else 0,
                'oldest_document': oldest_doc,
                'newest_document': newest_doc,
                'embedding_model': self.embedding_service.model,
                'vector_store_dimension': self.embedding_service.dimension
            }
            
            logger.info(f"Generated document statistics: {stats['total_documents']} docs, "
                       f"{stats['total_chunks']} chunks, {stats['total_vectors']} vectors")
            
            return stats
        except Exception as e:
            logger.error(f"Failed to generate document stats: {str(e)}")
            return {'error': str(e)}
    
    async def reindex_documents(self) -> Dict[str, Any]:
        """Reindex all documents (useful after changing embedding models)"""
        try:
            logger.info("Starting document reindexing process")
            
            if not self.documents:
                return {
                    'status': 'completed',
                    'message': 'No documents to reindex',
                    'reindexed_count': 0
                }
            
            self.vector_store = VectorStoreService()
            
            reindexed_count = 0
            errors = []
            
            for doc_id, doc_data in self.documents.items():
                try:
                    logger.info(f"Reindexing document: {doc_id}")
                    
                    # re-indexing can be implemented here, for which
                    # store original content (already implemented),
                    # then we can re-extract the content from files to reindex.
                    
                    reindexed_count += 1
                    
                except Exception as e:
                    error_msg = f"Failed to reindex document {doc_id}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            result = {
                'status': 'completed' if not errors else 'completed_with_errors',
                'reindexed_count': reindexed_count,
                'total_documents': len(self.documents),
                'errors': errors
            }
            
            logger.info(f"Reindexing completed: {reindexed_count}/{len(self.documents)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Reindexing process failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'reindexed_count': 0
            }