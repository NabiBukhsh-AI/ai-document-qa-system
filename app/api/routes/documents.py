from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
import os
import aiofiles
import logging
from ...models.schemas import DocumentInput, DocumentResponse, DocumentListResponse
from ...services.document_service import DocumentService
from ...core.security import get_current_user
from ...core.config import settings
from ...utils.exceptions import DocumentQAException
from ..dependencies import get_document_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/", response_model=DocumentResponse)
async def create_document(
    document: DocumentInput,
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(get_current_user)
):
    """Add a new document to the system"""
    try:
        logger.info(f"User {current_user.get('username', 'unknown')} adding document: {document.title}")
        
        result = await document_service.add_document(document)
        
        logger.info(f"Document added successfully: {result.id}")
        return result
        
    except DocumentQAException as e:
        logger.error(f"Document creation failed for user {current_user.get('username', 'unknown')}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(get_current_user)
):
    """Upload and process a document file"""
    try:
        username = current_user.get('username', 'unknown')
        logger.info(f"User {username} uploading file: {file.filename}")
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_ext} not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )
        
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        import uuid
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        result = await document_service.add_document_from_file(file_path, file.filename)
        
        background_tasks.add_task(cleanup_file, file_path)
        
        logger.info(f"File uploaded successfully by user {username}: {result.id}")
        return result
    
    except DocumentQAException as e:
        logger.error(f"File upload failed for user {current_user.get('username', 'unknown')}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload for user {current_user.get('username', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def cleanup_file(file_path: str):
    """Background task to cleanup uploaded file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")

@router.get("/", response_model=List[DocumentListResponse])
async def list_documents(
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(get_current_user)
):
    """Get list of all indexed documents"""
    try:
        logger.info(f"User {current_user.get('username', 'unknown')} listing documents")
        
        documents = document_service.get_documents()
        
        # document filtration can be implemented for authenticated users
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to list documents for user {current_user.get('username', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(get_current_user)
):
    """Get specific document by ID"""
    try:
        logger.info(f"User {current_user.get('username', 'unknown')} retrieving document: {doc_id}")
        
        result = document_service.get_document(doc_id)
        
        # permission check can be implemented for authenticated users here too
        
        return result
        
    except DocumentQAException as e:
        logger.error(f"Document retrieval failed for user {current_user.get('username', 'unknown')}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error retrieving document for user {current_user.get('username', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(get_current_user)
):
    """Delete a document and all its vectors"""
    try:
        username = current_user.get('username', 'unknown')
        logger.info(f"User {username} attempting to delete document: {doc_id}")
        
        # can add things like user can only delete their own documents, once user filtration or authentication is properly implemented
        
        success = document_service.delete_document(doc_id)
        
        if success:
            logger.info(f"Document {doc_id} deleted successfully by user {username}")
            return {
                "message": f"Document {doc_id} deleted successfully",
                "deleted_by": username,
                "success": True
            }
        else:
            logger.warning(f"Document {doc_id} had no vectors to delete for user {username}")
            return {
                "message": f"Document {doc_id} had no vectors to delete",
                "deleted_by": username,
                "success": False
            }
            
    except DocumentQAException as e:
        logger.error(f"Document deletion failed for user {current_user.get('username', 'unknown')}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error during document deletion for user {current_user.get('username', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.get("/health/status")
async def documents_health(
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(get_current_user)
):
    """Check health of document service"""
    try:
        documents = document_service.get_documents()
        document_count = len(documents)
        
        vector_store_health = document_service.vector_store.index.ntotal if hasattr(document_service.vector_store, 'index') else 0
        
        return {
            "status": "healthy",
            "document_count": document_count,
            "vector_count": vector_store_health,
            "checked_by": current_user.get('username', 'unknown')
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "checked_by": current_user.get('username', 'unknown')
        }