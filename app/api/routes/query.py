from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import time
import json
import logging
from typing import AsyncGenerator
from ...models.schemas import QueryRequest, QueryResponse, SourceDocument
from ...services.document_service import DocumentService
from ...services.llm_service import LLMService
from ...core.security import get_current_user
from ...utils.exceptions import DocumentQAException
from ..dependencies import get_document_service, get_llm_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query_documents(
    query: QueryRequest,
    document_service: DocumentService = Depends(get_document_service),
    llm_service: LLMService = Depends(get_llm_service),
    current_user: dict = Depends(get_current_user)
):
    """Answer a question using indexed documents"""
    try:
        start_time = time.time()
        username = current_user.get('username', 'unknown')
        
        logger.info(f"User {username} querying: '{query.question[:100]}...' "
                   f"(max_chunks={query.max_chunks}, threshold={query.similarity_threshold})")
        
        embedding_service = document_service.embedding_service
        query_embedding = await embedding_service.generate_embedding(query.question)

        search_results = document_service.vector_store.search(
            query_embedding, 
            k=query.max_chunks, 
            threshold=query.similarity_threshold
        )
        
        if not search_results:
            logger.info(f"No relevant documents found for user {username}'s query")
            return QueryResponse(
                answer="I couldn't find any relevant information in the indexed documents to answer your question.",
                sources=[],
                query=query.question,
                response_time_ms=int((time.time() - start_time) * 1000)
            )
        
        context_docs = []
        sources = []
        
        logger.info(f"Found {len(search_results)} relevant documents for user {username}")
        
        for metadata, score in search_results:
            context_docs.append({
                'title': metadata.get('doc_title', 'Unknown'),
                'content': metadata.get('content', '')
            })
            
            sources.append(SourceDocument(
                doc_id=metadata.get('doc_id', ''),
                title=metadata.get('doc_title', 'Unknown'),
                content=metadata.get('content', '')[:500] + "..." if len(metadata.get('content', '')) > 500 else metadata.get('content', ''),
                similarity_score=score,
                metadata={k: v for k, v in metadata.items() if k not in ['content', 'doc_title', 'doc_id']}
            ))
        
        logger.info(f"Generating answer for user {username} using {len(context_docs)} context documents")
        answer = await llm_service.generate_answer(query.question, context_docs)
        
        response_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Query completed for user {username} in {response_time}ms, "
                   f"answer length: {len(answer)} chars")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=query.question,
            response_time_ms=response_time
        )
    
    except DocumentQAException as e:
        logger.error(f"Query failed for user {current_user.get('username', 'unknown')}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in query for user {current_user.get('username', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# just fall back mechanism, using astream in llm_services.py from langchain which directly calls the ainvoke function for LLM invocation for responses from langchain.
@router.get("/stream")
async def stream_query(
    question: str = Query(..., description="The question to ask"),
    max_chunks: int = Query(default=5, ge=1, le=20, description="Maximum number of chunks to retrieve"),
    similarity_threshold: float = Query(default=0.7, ge=0.0, le=1.0, description="Similarity threshold for chunk selection"),
    token: str = Query(..., description="Bearer token for authentication"),
    document_service: DocumentService = Depends(get_document_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Stream answer for long responses using Server-Sent Events"""
    from jose import JWTError, jwt
    from ...core.security import settings
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=403, detail="Invalid authentication token")
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid authentication token")
    
    from ...core.security import settings, fake_users_db
    if username not in fake_users_db:
        raise HTTPException(status_code=403, detail="User not found")
    
    current_user = {"username": username}
    
    logger.info(f"User {username} starting streaming query: '{question[:100]}...'")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            start_time = time.time()
            
            yield f"data: {json.dumps({'type': 'status', 'content': f'Processing query for user {username}...', 'timestamp': time.time()})}\n\n"
            
            embedding_service = document_service.embedding_service
            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating query embedding...', 'timestamp': time.time()})}\n\n"
            
            query_embedding = await embedding_service.generate_embedding(question)
            
            yield f"data: {json.dumps({'type': 'status', 'content': 'Searching relevant documents...', 'timestamp': time.time()})}\n\n"
            
            search_results = document_service.vector_store.search(
                query_embedding, 
                k=max_chunks, 
                threshold=similarity_threshold
            )
            
            if not search_results:
                logger.info(f"No relevant documents found for streaming query from user {username}")
                yield f"data: {json.dumps({'type': 'answer', 'content': 'I could not find any relevant information in the indexed documents to answer your question.', 'user': username})}\n\n"
                yield f"data: {json.dumps({'type': 'end', 'response_time_ms': int((time.time() - start_time) * 1000), 'user': username, 'total_sources': 0})}\n\n"
                return
            
            sources = []
            context_docs = []
            
            for metadata, score in search_results:
                context_docs.append({
                    'title': metadata.get('doc_title', 'Unknown'),
                    'content': metadata.get('content', '')
                })
                
                source = {
                    'doc_id': metadata.get('doc_id', ''),
                    'title': metadata.get('doc_title', 'Unknown'),
                    'similarity_score': score
                }
                sources.append(source)
            
            logger.info(f"Streaming {len(sources)} sources to user {username}")
            yield f"data: {json.dumps({'type': 'sources', 'content': sources, 'count': len(sources)})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating answer...', 'timestamp': time.time()})}\n\n"
            
            if hasattr(llm_service, 'generate_streaming_answer'):
                chunk_count = 0
                async for chunk in llm_service.generate_streaming_answer(question, context_docs):
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk, 'chunk_index': chunk_count})}\n\n"
                    chunk_count += 1
            else:
                answer = await llm_service.generate_answer(question, context_docs)
                
                words = answer.split()
                chunk_size = 3
                total_chunks = len(words) // chunk_size + (1 if len(words) % chunk_size else 0)
                
                logger.info(f"Simulating streaming of {len(words)} words in {total_chunks} chunks to user {username}")
                
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    if i + chunk_size < len(words):
                        chunk += " "
                    
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk, 'chunk_index': i // chunk_size, 'total_chunks': total_chunks})}\n\n"
                    
                    import asyncio
                    await asyncio.sleep(0.05)
            
            response_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Streaming completed for user {username} in {response_time}ms")
            yield f"data: {json.dumps({'type': 'end', 'response_time_ms': response_time, 'user': username, 'total_sources': len(sources)})}\n\n"
            
        except DocumentQAException as e:
            logger.error(f"Streaming query failed for user {username}: {e.message}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Query error: {e.message}', 'user': username})}\n\n"
        except Exception as e:
            logger.error(f"Unexpected error in streaming query for user {username}: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Unexpected error: {str(e)}', 'user': username})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-User": username,
        }
    )

@router.get("/history")
async def get_query_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get query history for the current user (placeholder for future implementation)"""
    username = current_user.get('username', 'unknown')
    logger.info(f"User {username} requested query history (limit: {limit})")
    
    # placeholder - in a real implementation, you would store and retrieve using db.
    return {
        "message": "Query history feature not yet implemented",
        "user": username,
        "requested_limit": limit,
        "history": []
    }

@router.get("/stats")
async def get_query_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get query statistics for the current user"""
    username = current_user.get('username', 'unknown')
    logger.info(f"User {username} requested query statistics")
    
    # placeholder - same as above.
    return {
        "user": username,
        "total_queries": 0,
        "avg_response_time_ms": 0,
        "most_queried_topics": [],
        "success_rate": 100.0
    }

@router.get("/health")
async def query_service_health(
    document_service: DocumentService = Depends(get_document_service),
    llm_service: LLMService = Depends(get_llm_service),
    current_user: dict = Depends(get_current_user)
):
    """Check health of query-related services"""
    try:
        username = current_user.get('username', 'unknown')
        
        test_embedding = await document_service.embedding_service.generate_embedding("test")
        embedding_healthy = len(test_embedding) > 0
        
        vector_store_healthy = hasattr(document_service.vector_store, 'index')
        
        llm_healthy = hasattr(llm_service, 'model') or hasattr(llm_service, 'llm')
        
        health_status = {
            "status": "healthy" if all([embedding_healthy, vector_store_healthy, llm_healthy]) else "degraded",
            "services": {
                "embedding": "healthy" if embedding_healthy else "unhealthy",
                "vector_store": "healthy" if vector_store_healthy else "unhealthy", 
                "llm": "healthy" if llm_healthy else "unhealthy"
            },
            "checked_by": username,
            "timestamp": time.time()
        }
        
        logger.info(f"Query service health check by user {username}: {health_status['status']}")
        return health_status
        
    except Exception as e:
        logger.error(f"Query service health check failed for user {current_user.get('username', 'unknown')}: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "checked_by": current_user.get('username', 'unknown'),
            "timestamp": time.time()
        }