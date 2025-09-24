from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from fastapi.responses import HTMLResponse
from datetime import timedelta
import os
import logging

from .core.config import settings
from .core.security import authenticate_user, create_access_token
from .models.schemas import Token
from .api.routes import documents, query
from .utils.exceptions import DocumentQAException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Professional Document-based Question Answering API with Streaming Support",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

@app.exception_handler(DocumentQAException)
async def document_qa_exception_handler(request, exc: DocumentQAException):
    return HTTPException(status_code=exc.status_code, detail=exc.message)

@app.post(f"{settings.API_V1_STR}/token", response_model=Token)
async def login_for_access_token(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate and get access token"""
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

app.include_router(
    documents.router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["documents"]
)

app.include_router(
    query.router,
    prefix=f"{settings.API_V1_STR}/query",
    tags=["query"]
)

@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "message": "Document QA API with Streaming is running"
    }

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve enhanced frontend UI with streaming support"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document QA API - Enhanced with Streaming</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px; margin: 0 auto; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                background: white; border-radius: 12px; padding: 30px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header { text-align: center; margin-bottom: 40px; }
            .header h1 { color: #2d3748; margin-bottom: 10px; }
            .header p { color: #718096; font-size: 1.1em; }
            .section { margin-bottom: 30px; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; }
            .section h3 { color: #2d3748; margin-top: 0; }
            .form-group { margin-bottom: 20px; }
            .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #4a5568; }
            .form-control { 
                width: 100%; padding: 12px; border: 2px solid #e2e8f0; border-radius: 6px; 
                font-size: 16px; transition: border-color 0.2s;
            }
            .form-control:focus { outline: none; border-color: #667eea; }
            textarea.form-control { min-height: 120px; resize: vertical; }
            .btn { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 12px 24px; border: none; border-radius: 6px; 
                cursor: pointer; font-size: 16px; font-weight: 600; transition: transform 0.2s;
                margin-right: 10px;
            }
            .btn:hover { transform: translateY(-1px); }
            .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
            .btn-secondary { 
                background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
            }
            .result { 
                margin-top: 20px; padding: 15px; background: #f7fafc; 
                border-radius: 6px; border-left: 4px solid #667eea; 
            }
            .error { border-left-color: #e53e3e; background: #fed7d7; }
            .success { border-left-color: #38a169; background: #c6f6d5; }
            .streaming { border-left-color: #ed8936; background: #fef5e7; }
            .loading { display: inline-block; margin-left: 10px; }
            .source { 
                margin: 10px 0; padding: 12px; background: #edf2f7; 
                border-radius: 6px; font-size: 14px; 
            }
            .token-info {
                background: #bee3f8; padding: 15px; border-radius: 6px; margin-bottom: 20px;
                border-left: 4px solid #3182ce;
            }
            .streaming-content {
                min-height: 50px;
                max-height: 400px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
                white-space: pre-wrap;
            }
            .progress-bar {
                width: 100%;
                height: 4px;
                background: #e2e8f0;
                border-radius: 2px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 2px;
                transition: width 0.3s ease;
            }
            .status-indicator {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                margin-right: 8px;
            }
            .status-processing { background: #fed7d7; color: #c53030; }
            .status-streaming { background: #fef5e7; color: #dd6b20; }
            .status-complete { background: #c6f6d5; color: #2f855a; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Document QA API - Enhanced</h1>
                <p>Upload documents and ask questions with AI-powered search & real-time streaming</p>
                <div class="token-info">
                    <strong>Authentication Required:</strong> Use username: <code>admin</code>, password: <code>secret</code> to get your API token
                </div>
            </div>

            <!-- Authentication Section -->
            <div class="section">
                <h3>🔐 Authentication</h3>
                <div class="form-group">
                    <label>Username:</label>
                    <input type="text" id="username" class="form-control" value="admin">
                </div>
                <div class="form-group">
                    <label>Password:</label>
                    <input type="password" id="password" class="form-control" value="secret">
                </div>
                <button onclick="getToken()" class="btn">Get API Token</button>
                <div id="tokenResult"></div>
            </div>

            <!-- File Upload Section -->
            <div class="section">
                <h3>📄 Upload Document</h3>
                <div class="form-group">
                    <label>Choose File (PDF, TXT, MD):</label>
                    <input type="file" id="fileInput" class="form-control" accept=".pdf,.txt,.md">
                </div>
                <button onclick="uploadFile()" class="btn">Upload Document</button>
                <div id="uploadResult"></div>
            </div>

            <!-- Manual Document Addition -->
            <div class="section">
                <h3>✍️ Add Document Manually</h3>
                <div class="form-group">
                    <label>Document Title:</label>
                    <input type="text" id="docTitle" class="form-control" placeholder="Enter document title">
                </div>
                <div class="form-group">
                    <label>Document Content:</label>
                    <textarea id="docContent" class="form-control" placeholder="Paste your document content here..."></textarea>
                </div>
                <button onclick="addDocument()" class="btn">Add Document</button>
                <div id="addResult"></div>
            </div>

            <!-- Enhanced Query Section with Streaming -->
            <div class="section">
                <h3>❓ Ask Question</h3>
                <div class="form-group">
                    <label>Your Question:</label>
                    <input type="text" id="question" class="form-control" placeholder="What would you like to know?" onkeypress="handleQuestionKeyPress(event)">
                </div>
                <div class="form-group">
                    <button onclick="askQuestion()" class="btn">Search & Answer</button>
                    <button onclick="askQuestionStreaming()" class="btn btn-secondary">🔴 Stream Response</button>
                    <button onclick="stopStreaming()" class="btn" id="stopBtn" style="display: none; background: #e53e3e;">Stop Streaming</button>
                </div>
                <div id="queryResult"></div>
            </div>

            <!-- Documents List -->
            <div class="section">
                <h3>📚 Your Documents</h3>
                <button onclick="loadDocuments()" class="btn">Refresh List</button>
                <div id="documentsList"></div>
            </div>
        </div>

        <script>
            let authToken = '';
            let currentEventSource = null;
            let isStreaming = false;
            const API_BASE = '/api/v1';

            function handleQuestionKeyPress(event) {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    askQuestion();
                }
            }

            // Authentication
            async function getToken() {
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                const resultDiv = document.getElementById('tokenResult');
                
                try {
                    const credentials = btoa(username + ':' + password);
                    const response = await fetch(API_BASE + '/token', {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Basic ' + credentials,
                            'Content-Type': 'application/x-www-form-urlencoded',
                        }
                    });

                    if (response.ok) {
                        const data = await response.json();
                        authToken = data.access_token;
                        resultDiv.innerHTML = '<div class="result success">✅ Authentication successful! Token obtained.</div>';
                        loadDocuments(); // Auto-load documents after auth
                    } else {
                        const error = await response.json();
                        resultDiv.innerHTML = `<div class="result error">❌ Authentication failed: ${error.detail}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">❌ Error: ${error.message}</div>`;
                }
            }

            // File Upload
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('uploadResult');
                
                if (!authToken) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please authenticate first</div>';
                    return;
                }
                
                if (!fileInput.files[0]) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please select a file</div>';
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    resultDiv.innerHTML = '<div class="result">⏳ Uploading and processing file...</div>';
                    
                    const response = await fetch(API_BASE + '/documents/upload', {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Bearer ' + authToken,
                        },
                        body: formData
                    });

                    if (response.ok) {
                        const data = await response.json();
                        resultDiv.innerHTML = `<div class="result success">✅ File uploaded successfully!<br>
                            <strong>Title:</strong> ${data.title}<br>
                            <strong>Chunks:</strong> ${data.chunk_count}<br>
                            <strong>ID:</strong> ${data.id}</div>`;
                        loadDocuments(); // Refresh document list
                    } else {
                        const error = await response.json();
                        resultDiv.innerHTML = `<div class="result error">❌ Upload failed: ${error.detail}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">❌ Error: ${error.message}</div>`;
                }
            }

            // Manual Document Addition
            async function addDocument() {
                const title = document.getElementById('docTitle').value;
                const content = document.getElementById('docContent').value;
                const resultDiv = document.getElementById('addResult');
                
                if (!authToken) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please authenticate first</div>';
                    return;
                }
                
                if (!title || !content) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please provide both title and content</div>';
                    return;
                }

                try {
                    resultDiv.innerHTML = '<div class="result">⏳ Adding document...</div>';
                    
                    const response = await fetch(API_BASE + '/documents/', {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Bearer ' + authToken,
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ title, content })
                    });

                    if (response.ok) {
                        const data = await response.json();
                        resultDiv.innerHTML = `<div class="result success">✅ Document added successfully!<br>
                            <strong>Title:</strong> ${data.title}<br>
                            <strong>Chunks:</strong> ${data.chunk_count}<br>
                            <strong>ID:</strong> ${data.id}</div>`;
                        document.getElementById('docTitle').value = '';
                        document.getElementById('docContent').value = '';
                        loadDocuments(); // Refresh document list
                    } else {
                        const error = await response.json();
                        resultDiv.innerHTML = `<div class="result error">❌ Failed to add document: ${error.detail}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">❌ Error: ${error.message}</div>`;
                }
            }

            // Regular Query (Non-streaming)
            async function askQuestion() {
                const question = document.getElementById('question').value;
                const resultDiv = document.getElementById('queryResult');
                
                if (!authToken) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please authenticate first</div>';
                    return;
                }
                
                if (!question) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please enter a question</div>';
                    return;
                }

                try {
                    resultDiv.innerHTML = '<div class="result">⏳ Searching documents and generating answer...</div>';
                    
                    const response = await fetch(API_BASE + '/query/', {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Bearer ' + authToken,
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question })
                    });

                    if (response.ok) {
                        const data = await response.json();
                        let sourcesHtml = '';
                        data.sources.forEach((source, index) => {
                            sourcesHtml += `<div class="source">
                                <strong>Source ${index + 1}:</strong> ${source.title} (Score: ${source.similarity_score.toFixed(3)})<br>
                                ${source.content}
                            </div>`;
                        });
                        
                        resultDiv.innerHTML = `<div class="result success">
                            <h4>🤖 Answer:</h4>
                            <p>${data.answer}</p>
                            <h4>📖 Sources Used:</h4>
                            ${sourcesHtml}
                            <small><strong>Response time:</strong> ${data.response_time_ms}ms</small>
                        </div>`;
                    } else {
                        const error = await response.json();
                        resultDiv.innerHTML = `<div class="result error">❌ Query failed: ${error.detail}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">❌ Error: ${error.message}</div>`;
                }
            }

            // Streaming Query
            async function askQuestionStreaming() {
                const question = document.getElementById('question').value;
                const resultDiv = document.getElementById('queryResult');
                const stopBtn = document.getElementById('stopBtn');
                
                if (!authToken) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please authenticate first</div>';
                    return;
                }
                
                if (!question) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please enter a question</div>';
                    return;
                }

                // Stop any existing stream
                stopStreaming();

                try {
                    isStreaming = true;
                    stopBtn.style.display = 'inline-block';
                    
                    // Initialize streaming UI
                    resultDiv.innerHTML = `
                        <div class="result streaming">
                            <h4>🔴 Live Response Stream:</h4>
                            <div class="status-indicator status-processing">Processing...</div>
                            <div class="progress-bar">
                                <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                            </div>
                            <div class="streaming-content" id="streamingContent"></div>
                            <div id="sourcesContainer"></div>
                            <div id="statusContainer"></div>
                        </div>
                    `;

                    const streamingContent = document.getElementById('streamingContent');
                    const sourcesContainer = document.getElementById('sourcesContainer');
                    const statusContainer = document.getElementById('statusContainer');
                    const progressFill = document.getElementById('progressFill');
                    
                    // Build streaming URL with proper authentication via token parameter
                    const streamUrl = new URL(`${window.location.origin}${API_BASE}/query/stream`);
                    streamUrl.searchParams.set('question', question);
                    streamUrl.searchParams.set('max_chunks', '5');
                    streamUrl.searchParams.set('similarity_threshold', '0.7');
                    streamUrl.searchParams.set('token', authToken); // Pass token as query parameter
                    
                    // Create EventSource for streaming
                    const eventSource = new EventSource(streamUrl.toString());
                    
                    // Set authorization header (Note: EventSource doesn't support custom headers directly)
                    // We'll handle auth through the dependency injection in the backend
                    
                    currentEventSource = eventSource;
                    
                    let currentProgress = 0;
                    let totalChunks = 1;
                    let receivedChunks = 0;

                    eventSource.onopen = function(event) {
                        console.log('Streaming connection opened');
                        updateStatus('Streaming connection established...', 'processing');
                        progressFill.style.width = '10%';
                    };

                    eventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            
                            switch(data.type) {
                                case 'status':
                                    updateStatus(data.content, 'processing');
                                    if (data.content.includes('embedding')) {
                                        progressFill.style.width = '20%';
                                    } else if (data.content.includes('Searching')) {
                                        progressFill.style.width = '30%';
                                    } else if (data.content.includes('Generating')) {
                                        progressFill.style.width = '40%';
                                    }
                                    break;
                                    
                                case 'sources':
                                    displaySources(data.content, data.count || 0);
                                    progressFill.style.width = '50%';
                                    updateStatus(`Found ${data.count} relevant sources`, 'streaming');
                                    break;
                                    
                                case 'answer':
                                    streamingContent.textContent += data.content;
                                    streamingContent.scrollTop = streamingContent.scrollHeight;
                                    
                                    if (data.total_chunks) {
                                        totalChunks = data.total_chunks;
                                        receivedChunks = data.chunk_index + 1;
                                        currentProgress = 50 + (receivedChunks / totalChunks) * 40;
                                        progressFill.style.width = currentProgress + '%';
                                    } else {
                                        // Incremental progress for word-based streaming
                                        currentProgress = Math.min(90, currentProgress + 1);
                                        progressFill.style.width = currentProgress + '%';
                                    }
                                    updateStatus('Streaming response...', 'streaming');
                                    break;
                                    
                                case 'end':
                                    progressFill.style.width = '100%';
                                    updateStatus(`Completed in ${data.response_time_ms}ms with ${data.total_sources} sources`, 'complete');
                                    stopStreaming();
                                    break;
                                    
                                case 'error':
                                    streamingContent.textContent += `\n\nError: ${data.content}`;
                                    updateStatus('Error occurred during streaming', 'error');
                                    stopStreaming();
                                    break;
                            }
                        } catch (e) {
                            console.error('Error parsing streaming data:', e);
                            updateStatus('Error parsing response data', 'error');
                        }
                    };

                    eventSource.onerror = function(event) {
                        console.error('Streaming error:', event);
                        if (event.target.readyState === EventSource.CONNECTING) {
                            updateStatus('Reconnecting to stream...', 'processing');
                        } else if (event.target.readyState === EventSource.CLOSED) {
                            updateStatus('Stream connection closed', 'error');
                            stopStreaming();
                        } else {
                            updateStatus('Streaming connection error', 'error');
                            stopStreaming();
                        }
                    };

                    function updateStatus(message, type) {
                        const statusIndicator = resultDiv.querySelector('.status-indicator');
                        if (statusIndicator) {
                            statusIndicator.textContent = message;
                            statusIndicator.className = `status-indicator status-${type}`;
                        }
                    }

                    function displaySources(sources, count) {
                        let sourcesHtml = `<h4>📖 Sources Found (${count}):</h4>`;
                        sources.forEach((source, index) => {
                            sourcesHtml += `<div class="source">
                                <strong>Source ${index + 1}:</strong> ${source.title} (Score: ${source.similarity_score.toFixed(3)})
                            </div>`;
                        });
                        sourcesContainer.innerHTML = sourcesHtml;
                    }

                } catch (error) {
                    console.error('Streaming setup error:', error);
                    resultDiv.innerHTML = `<div class="result error">❌ Streaming error: ${error.message}</div>`;
                    stopStreaming();
                }
            }

            function stopStreaming() {
                if (currentEventSource) {
                    currentEventSource.close();
                    currentEventSource = null;
                }
                
                isStreaming = false;
                document.getElementById('stopBtn').style.display = 'none';
                
                if (isStreaming) {
                    const resultDiv = document.getElementById('queryResult');
                    const statusIndicator = resultDiv.querySelector('.status-indicator');
                    if (statusIndicator) {
                        statusIndicator.textContent = 'Streaming stopped by user';
                        statusIndicator.className = 'status-indicator status-error';
                    }
                }
            }

            // Load Documents List
            async function loadDocuments() {
                const resultDiv = document.getElementById('documentsList');
                
                if (!authToken) {
                    resultDiv.innerHTML = '<div class="result error">❌ Please authenticate first</div>';
                    return;
                }

                try {
                    const response = await fetch(API_BASE + '/documents/', {
                        headers: {
                            'Authorization': 'Bearer ' + authToken,
                        }
                    });

                    if (response.ok) {
                        const documents = await response.json();
                        if (documents.length === 0) {
                            resultDiv.innerHTML = '<div class="result">📭 No documents uploaded yet</div>';
                        } else {
                            let docsHtml = '<div class="result success"><h4>📚 Indexed Documents:</h4>';
                            documents.forEach(doc => {
                                docsHtml += `<div class="source">
                                    <strong>${doc.title}</strong> (${doc.chunk_count} chunks)<br>
                                    <small>ID: ${doc.id} | Created: ${new Date(doc.created_at).toLocaleDateString()}</small>
                                    <button onclick="deleteDocument('${doc.id}')" style="float: right; background: #e53e3e; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer;">Delete</button>
                                </div>`;
                            });
                            docsHtml += '</div>';
                            resultDiv.innerHTML = docsHtml;
                        }
                    } else {
                        const error = await response.json();
                        resultDiv.innerHTML = `<div class="result error">❌ Failed to load documents: ${error.detail}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">❌ Error: ${error.message}</div>`;
                }
            }

            // Delete Document
            async function deleteDocument(docId) {
                if (!confirm('Are you sure you want to delete this document?')) return;
                
                try {
                    const response = await fetch(API_BASE + '/documents/' + docId, {
                        method: 'DELETE',
                        headers: {
                            'Authorization': 'Bearer ' + authToken,
                        }
                    });

                    if (response.ok) {
                        loadDocuments(); // Refresh list
                    } else {
                        const error = await response.json();
                        alert('Delete failed: ' + error.detail);
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }

            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {
                stopStreaming();
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )