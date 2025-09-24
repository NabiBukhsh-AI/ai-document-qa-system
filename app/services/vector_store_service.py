import faiss
import numpy as np
import json
import os
import logging
from typing import List, Tuple, Dict, Any
from ..core.config import settings
from ..utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.index_path = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
        self.metadata_path = os.path.join(settings.VECTOR_STORE_PATH, "metadata.json")
        self.dimension = settings.EMBEDDING_DIMENSION
        
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
        
        self.index = None
        self.metadata = {}
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)  # inner product for cosine similarity
                self.metadata = {}
                self._save_index()
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise VectorStoreError(f"Failed to load vector store: {str(e)}")
    
    def _save_index(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise VectorStoreError(f"Failed to save vector store: {str(e)}")
    
    def add_documents(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Add document embeddings with metadata"""
        try:
            if not embeddings or len(embeddings) != len(metadatas):
                raise ValueError("Embeddings and metadata lists must have same length")

            embeddings_array = np.array(embeddings, dtype=np.float32)

            # if index is empty, or dimension mismatch, recreate it
            if self.index is None or self.index.d != embeddings_array.shape[1]:
                self.dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)

            faiss.normalize_L2(embeddings_array)

            start_idx = self.index.ntotal
            self.index.add(embeddings_array)

            for i, metadata in enumerate(metadatas):
                self.metadata[str(start_idx + i)] = metadata

            self._save_index()
            return start_idx

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise VectorStoreError(f"Failed to add documents: {str(e)}")

    
    def search(self, query_embedding: List[float], k: int = 5, threshold: float = 0.7) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        try:
            if self.index.ntotal == 0:
                return []
            
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and str(idx) in self.metadata:
                    results.append((self.metadata[str(idx)], float(score)))
            
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def delete_document(self, doc_id: str) -> int:
        """Delete all vectors for a document (requires rebuilding index) - FIXED VERSION"""
        try:
            logger.info(f"Attempting to delete document: {doc_id}")
            
            # Find indices to remove
            indices_to_remove = []
            for idx_str, metadata in self.metadata.items():
                if metadata.get('doc_id') == doc_id:
                    indices_to_remove.append(int(idx_str))
            
            if not indices_to_remove:
                logger.warning(f"No vectors found for document: {doc_id}")
                return 0
            
            logger.info(f"Found {len(indices_to_remove)} vectors to delete")
            
            # If we're deleting all vectors, just reset everything
            if len(indices_to_remove) >= self.index.ntotal:
                logger.info("Deleting all vectors, resetting index")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = {}
                self._save_index()
                return len(indices_to_remove)
            
            # Otherwise rebuild the index
            all_embeddings = []
            new_metadata = {}
            new_idx = 0
            failed_reconstructions = 0
            
            for idx in range(self.index.ntotal):
                if idx not in indices_to_remove:
                    try:
                        # Try to reconstruct the embedding
                        embedding = self.index.reconstruct(idx)
                        
                        # Validate the embedding
                        if embedding is None or len(embedding) != self.dimension:
                            logger.warning(f"Invalid embedding at index {idx}, skipping")
                            failed_reconstructions += 1
                            continue
                        
                        all_embeddings.append(embedding)
                        
                        # Copy metadata if it exists
                        if str(idx) in self.metadata:
                            new_metadata[str(new_idx)] = self.metadata[str(idx)]
                        else:
                            logger.warning(f"No metadata found for index {idx}")
                        
                        new_idx += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to reconstruct embedding at index {idx}: {str(e)}")
                        failed_reconstructions += 1
                        continue
            
            if failed_reconstructions > 0:
                logger.warning(f"Failed to reconstruct {failed_reconstructions} embeddings during deletion")
            
            # Create new index
            logger.info(f"Rebuilding index with {len(all_embeddings)} embeddings")
            self.index = faiss.IndexFlatIP(self.dimension)
            
            if all_embeddings:
                try:
                    embeddings_array = np.array(all_embeddings, dtype=np.float32)
                    self.index.add(embeddings_array)
                except Exception as e:
                    logger.error(f"Failed to add embeddings to new index: {str(e)}")
                    # Reset to empty index rather than corrupt state
                    self.index = faiss.IndexFlatIP(self.dimension)
                    new_metadata = {}
            
            self.metadata = new_metadata
            self._save_index()
            
            deleted_count = len(indices_to_remove)
            logger.info(f"Successfully deleted {deleted_count} vectors for document {doc_id}")
            return deleted_count
        
        except Exception as e:
            error_msg = f"Failed to delete document {doc_id}: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        chunks = []
        for metadata in self.metadata.values():
            if metadata.get('doc_id') == doc_id:
                chunks.append(metadata)
        return sorted(chunks, key=lambda x: x.get('chunk_index', 0))
    
    def validate_index_consistency(self) -> Dict[str, Any]:
        """Check if index and metadata are consistent"""
        try:
            issues = []
            
            # Check vector count vs metadata count
            if self.index.ntotal != len(self.metadata):
                issues.append(f"Vector count ({self.index.ntotal}) != metadata count ({len(self.metadata)})")
            
            # Check for missing metadata
            for i in range(self.index.ntotal):
                if str(i) not in self.metadata:
                    issues.append(f"Missing metadata for vector {i}")
            
            # Check for orphaned metadata
            for idx_str in self.metadata.keys():
                try:
                    idx = int(idx_str)
                    if idx >= self.index.ntotal:
                        issues.append(f"Orphaned metadata for index {idx}")
                except ValueError:
                    issues.append(f"Invalid metadata key: {idx_str}")
            
            return {
                'consistent': len(issues) == 0,
                'issues': issues,
                'vector_count': self.index.ntotal,
                'metadata_count': len(self.metadata)
            }
            
        except Exception as e:
            logger.error(f"Failed to validate index consistency: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_orphaned_metadata(self) -> int:
        """Remove metadata for non-existent vectors"""
        try:
            original_count = len(self.metadata)
            valid_metadata = {}
            
            for i in range(self.index.ntotal):
                idx_str = str(i)
                if idx_str in self.metadata:
                    valid_metadata[idx_str] = self.metadata[idx_str]
            
            self.metadata = valid_metadata
            self._save_index()
            
            cleaned_count = original_count - len(valid_metadata)
            logger.info(f"Cleaned up {cleaned_count} orphaned metadata entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup metadata: {str(e)}")
            return 0