import os
import hashlib
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import markdown
from ..core.config import settings

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def generate_doc_id(self, title: str, content: str) -> str:
        """Generate unique document ID based on title and content hash"""
        content_hash = hashlib.md5((title + content).encode()).hexdigest()
        return f"doc_{content_hash[:16]}"
    
    def split_document(self, content: str, title: str) -> List[Tuple[str, dict]]:
        """Split document into chunks with metadata"""
        chunks = self.text_splitter.split_text(content)
        doc_id = self.generate_doc_id(title, content)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "doc_id": doc_id,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            chunk_data.append((chunk, metadata))
        
        return chunk_data
    
    def process_file(self, file_path: str, filename: str) -> Tuple[str, str]:
        """Process uploaded file and extract content"""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            title = os.path.splitext(filename)[0]
        
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = "\n\n".join([page.page_content for page in pages])
            title = os.path.splitext(filename)[0]
        
        elif ext == ".md":
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            content = markdown.markdown(md_content)
            title = os.path.splitext(filename)[0]
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return title, content