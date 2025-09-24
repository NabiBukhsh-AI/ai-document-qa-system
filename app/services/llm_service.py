import google.generativeai as genai
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import time
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate, 
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.output_parsers import (
    PydanticOutputParser,
    OutputFixingParser
)
from langchain_core.runnables import RunnableSequence
from langchain_core.caches import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import LLMResult

try:
    from langchain_community.chat_models import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pydantic import BaseModel, Field

from ..core.config import settings
from ..utils.exceptions import LLMServiceError

logger = logging.getLogger(__name__)

class DocumentAnswer(BaseModel):
    """Structured answer with citations"""
    answer: str = Field(description="The main answer to the question")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)
    citations: List[str] = Field(description="List of document titles/sources used")
    reasoning: Optional[str] = Field(description="Brief explanation of reasoning", default=None)
    limitations: Optional[str] = Field(description="Any limitations or uncertainties", default=None)

class StreamingCallback(AsyncCallbackHandler):
    """Callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
        self.current_response = ""
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated"""
        self.tokens.append(token)
        self.current_response += token
    
    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes"""
        logger.info(f"LLM completed with {len(self.tokens)} tokens")

class LangChainLLMService:
    """
    Enhanced LLM service using LangChain with Google Gemini
    """
    
    def __init__(self):
        set_llm_cache(InMemoryCache())
        
        self.provider = getattr(settings, 'LLM_PROVIDER', 'google').lower()
        self._init_llm()
        
        self._init_prompts()
        
        self.answer_parser = PydanticOutputParser(pydantic_object=DocumentAnswer)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.answer_parser, llm=self.llm)
        
        self.request_count = 0
        self.total_response_time = 0
        self.cache_hits = 0
        
        logger.info(f"Initialized LangChain LLM service with {self.provider} provider")
    
    def _init_llm(self):
        """Initialize LLM based on provider"""
        try:
            if self.provider == "google":
                self._init_google_llm()
            elif self.provider == "openai":
                self._init_openai_llm()
            else:
                raise LLMServiceError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} LLM: {str(e)}")
            raise LLMServiceError(f"LLM initialization failed: {str(e)}")
    
    def _init_google_llm(self):
        """Initialize Google Gemini LLM"""
        if not settings.GOOGLE_API_KEY:
            raise LLMServiceError("Google API key not configured")
        
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40
        )
        
        self.direct_model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        logger.info(f"Initialized Google Gemini LLM: {settings.GEMINI_MODEL}")
    
    def _init_openai_llm(self):
        """Initialize OpenAI LLM as fallback"""
        if not OPENAI_AVAILABLE:
            raise LLMServiceError("OpenAI package not installed")
        
        if not settings.OPENAI_API_KEY:
            raise LLMServiceError("OpenAI API key not configured")
        
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=getattr(settings, 'OPENAI_MODEL', 'gpt-3.5-turbo'),
            temperature=0.1,
            max_tokens=2048
        )
        
        logger.info(f"Initialized OpenAI LLM: {getattr(settings, 'OPENAI_MODEL', 'gpt-3.5-turbo')}")
    
    def _init_prompts(self):
        """Initialize prompt templates"""

        self.system_template = SystemMessagePromptTemplate.from_template(
            """You are a helpful AI assistant that answers questions based on provided context documents.

Instructions:
- Answer the question using only the information provided in the context documents
- If the context doesn't contain enough information to answer the question, clearly state this
- Be concise but comprehensive in your answers
- Cite specific parts of the context when relevant using document titles
- If multiple documents provide relevant information, synthesize them coherently
- Provide a confidence score for your answer
- Note any limitations or uncertainties in your response

Context Documents:
{context}"""
        )

        self.human_template = HumanMessagePromptTemplate.from_template(
            "Question: {question}\n\nPlease provide a structured answer."
        )
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_template,
            self.human_template
        ])
        
        self.simple_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Context Documents:
{context}

Question: {question}

Based on the provided context documents, please answer the question. If the context doesn't contain enough information, please say so clearly.

Answer:"""
        )
        
        self.chat_chain = self.chat_prompt | self.llm
        self.simple_chain = self.simple_prompt | self.llm
    
    async def generate_answer(
        self, 
        question: str, 
        context_documents: List[Dict[str, Any]], 
        structured: bool = False,
        use_citations: bool = True
    ) -> str:
        """Generate answer based on context documents"""
        try:
            start_time = time.time()
            self.request_count += 1
            
            logger.info(f"Generating answer for question: '{question[:100]}...' "
                       f"with {len(context_documents)} context documents")
            
            context_text = self._format_context(context_documents, use_citations)
            
            if structured:
                response = await self._generate_structured_answer(question, context_text, context_documents)
            else:
                response = await self._generate_simple_answer(question, context_text)
            
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            logger.info(f"Generated answer in {response_time:.2f}s, "
                       f"response length: {len(response)} chars")
            
            return response
        
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            raise LLMServiceError(f"Failed to generate answer: {str(e)}")
    
    async def _generate_simple_answer(self, question: str, context_text: str) -> str:
        """Generate simple text answer"""
        try:
            result = await self.simple_chain.ainvoke({
                "context": context_text,
                "question": question
            })
            return result.content.strip()
            #return result.strip()
        except Exception as e:
            logger.error(f"Simple answer generation failed: {str(e)}")
            raise LLMServiceError(f"Answer generation failed: {str(e)}")
    
    async def _generate_structured_answer(
        self, 
        question: str, 
        context_text: str, 
        context_documents: List[Dict[str, Any]]
    ) -> str:
        """Generate structured answer with citations"""
        try:
            format_instructions = self.answer_parser.get_format_instructions()
            
            messages = self.chat_prompt.format_prompt(
                context=context_text + f"\n\nOutput Format:\n{format_instructions}",
                question=question
            ).to_messages()
            
            result = await self.llm.ainvoke(messages)

            try:
                parsed_answer = self.answer_parser.parse(result.content)
                return self._format_structured_response(parsed_answer)
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured response: {parse_error}")
                try:
                    fixed_answer = await self.fixing_parser.aparse(result.content)
                    return self._format_structured_response(fixed_answer)
                except Exception:
                    return result.content.strip()
        
        except Exception as e:
            logger.error(f"Structured answer generation failed: {str(e)}")
            raise LLMServiceError(f"Structured answer generation failed: {str(e)}")
    
    def _format_structured_response(self, answer: DocumentAnswer) -> str:
        """Format structured answer for display"""
        response_parts = [answer.answer]
        
        if answer.confidence < 1.0:
            response_parts.append(f"\n\n**Confidence:** {answer.confidence:.1%}")
        
        if answer.citations:
            citations_text = ", ".join(answer.citations)
            response_parts.append(f"\n\n**Sources:** {citations_text}")
        
        if answer.reasoning:
            response_parts.append(f"\n\n**Reasoning:** {answer.reasoning}")
        
        if answer.limitations:
            response_parts.append(f"\n\n**Limitations:** {answer.limitations}")
        
        return "".join(response_parts)
    
    async def generate_streaming_answer(
        self, 
        question: str, 
        context_documents: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Generate streaming answer with token-level streaming"""
        try:
            logger.info(f"Starting streaming answer for: '{question[:100]}...'")
            
            context_text = self._format_context(context_documents)

            async for chunk in self.simple_chain.astream(
                {"context": context_text, "question": question}
            ):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Streaming answer generation failed: {str(e)}")
            yield f"Error: {str(e)}"

    
    def _format_context(self, context_documents: List[Dict[str, Any]], use_citations: bool = True) -> str:
        """Format context documents for the prompt"""
        context_parts = []
        
        for i, doc in enumerate(context_documents, 1):
            title = doc.get('title', f'Unknown Document {i}')
            content = doc.get('content', '')
            
            if use_citations:
                context_parts.append(f"Document {i} - {title}:\n{content}")
            else:
                context_parts.append(f"{content}")
        
        return "\n\n".join(context_parts)
    
    async def summarize_documents(self, documents: List[Dict[str, Any]], max_length: int = 500) -> str:
        """Summarize multiple documents"""
        try:
            context_text = self._format_context(documents)

            summary_prompt = PromptTemplate(
                input_variables=["context", "max_length"],
                template="""Please provide a concise summary of the following documents in no more than {max_length} words:

    {context}

    Summary:"""
            )

            summary_chain = summary_prompt | self.llm

            result = await summary_chain.ainvoke({
                "context": context_text,
                "max_length": max_length
            })

            return result.content.strip()
            #return result.strip()

        except Exception as e:
            logger.error(f"Document summarization failed: {str(e)}")
            raise LLMServiceError(f"Summarization failed: {str(e)}")

    
    async def extract_key_points(self, documents: List[Dict[str, Any]], num_points: int = 5) -> List[str]:
        """Extract key points from documents"""
        try:
            context_text = self._format_context(documents)

            extraction_prompt = PromptTemplate(
                input_variables=["context", "num_points"],
                template="""Extract the {num_points} most important key points from the following documents:

    {context}

    Please format your response as a numbered list of key points.

    Key Points:"""
            )

            extraction_chain = extraction_prompt | self.llm

            result = await extraction_chain.ainvoke({
                "context": context_text,
                "num_points": num_points
            })

            lines = result.content.split('\n')
            #lines = result.strip().split('\n')
            key_points = []

            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    cleaned = line.lstrip('0123456789.-• ').strip()
                    if cleaned:
                        key_points.append(cleaned)

            return key_points[:num_points]

        except Exception as e:
            logger.error(f"Key point extraction failed: {str(e)}")
            raise LLMServiceError(f"Key point extraction failed: {str(e)}")

    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the LLM service connection"""
        try:
            test_question = "What is artificial intelligence?"
            test_context = [{"title": "Test Document", "content": "Artificial intelligence is a field of computer science."}]
            
            start_time = time.time()
            result = await self.generate_answer(test_question, test_context)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'provider': self.provider,
                'model': getattr(self.llm, 'model_name', 'unknown'),
                'response_time_ms': response_time,
                'response_length': len(result),
                'test_successful': True
            }
        except Exception as e:
            logger.error(f"LLM service test failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'provider': self.provider,
                'error': str(e),
                'test_successful': False
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        avg_response_time = self.total_response_time / self.request_count if self.request_count > 0 else 0
        
        return {
            'provider': self.provider,
            'model': getattr(self.llm, 'model_name', settings.GEMINI_MODEL),
            'total_requests': self.request_count,
            'cache_hits': self.cache_hits,
            'avg_response_time_seconds': avg_response_time,
            'cache_hit_rate': self.cache_hits / self.request_count if self.request_count > 0 else 0
        }
    
    async def validate_answer_quality(self, question: str, answer: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of generated answer"""
        try:
            validation_prompt = PromptTemplate(
                input_variables=["question", "answer", "context"],
                template="""Please evaluate the quality of the following answer:

    Question: {question}

    Answer: {answer}

    Context Documents:
    {context}

    Please evaluate on a scale of 1-10:
    1. Accuracy (based on context)
    2. Completeness
    3. Clarity
    4. Relevance

    Format your response as JSON with scores and brief explanations.
    """
            )

            context_text = self._format_context(context_docs)

            validation_chain = validation_prompt | self.llm

            result = await validation_chain.ainvoke({
                "question": question,
                "answer": answer,
                "context": context_text
            })
            try:
                return json.loads(result.content)
            except json.JSONDecodeError:
                return {
                    "validation_text": result.content,
                    "parseable": False
                }

        except Exception as e:
            logger.error(f"Answer validation failed: {str(e)}")
            return {
                "error": str(e),
                "validation_successful": False
            }

LLMService = LangChainLLMService