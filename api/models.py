# api/models.py
# Data models for FetiiGPT Transportation Chatbot API

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    """Enumeration for the role of the message sender."""
    USER = "user"
    ASSISTANT = "assistant"

class QueryType(str, Enum):
    """Types of queries supported by FetiiGPT."""
    SQL = "SQL"
    RAG = "RAG"
    HYBRID = "HYBRID"

class ProcessingType(str, Enum):
    """Processing types for FetiiGPT responses."""
    SQL_ONLY = "SQL"
    RAG_ONLY = "RAG"
    HYBRID = "HYBRID"

class ChatMessage(BaseModel):
    """Data model for a single chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    query_type: Optional[QueryType] = None
    processing_time: Optional[float] = None

class ChatHistory(BaseModel):
    """In-memory representation of a chat session's history."""
    messages: List[ChatMessage] = Field(default_factory=list)
    
    def add_message(self, role: MessageRole, content: str, query_type: QueryType = None, processing_time: float = None):
        """Adds a new message to the in-memory chat history."""
        message = ChatMessage(
            role=role, 
            content=content, 
            query_type=query_type,
            processing_time=processing_time
        )
        self.messages.append(message)
    
    def get_llm_context(self, limit: int = 6) -> List[Dict[str, str]]:
        """Formats the most recent messages for the LLM's context window."""
        recent_messages = self.messages[-limit:]
        return [{"role": msg.role.value, "content": msg.content} for msg in recent_messages]

class QueryRequest(BaseModel):
    """Request model for a user's transportation data query."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question about Fetii transportation data")
    session_id: Optional[str] = Field(None, description="Unique ID for the conversation session")
    show_debug: Optional[bool] = Field(False, description="Include debug information in response")

class SQLResult(BaseModel):
    """SQL query execution result."""
    sql_query: Optional[str] = None
    data: Optional[Union[List[Dict], Dict, str]] = None
    row_count: Optional[int] = None
    execution_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None

class RAGResult(BaseModel):
    """RAG search result."""
    context_snippets: Optional[List[Dict[str, Any]]] = None
    total_results: Optional[int] = None
    search_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None

class ProcessingResult(BaseModel):
    """Complete processing result from FetiiGPT."""
    processing_type: ProcessingType
    sql_result: Optional[SQLResult] = None
    rag_result: Optional[RAGResult] = None
    classification: Optional[Dict[str, Any]] = None
    processing_time: float
    success: bool
    error: Optional[str] = None

class DebugInfo(BaseModel):
    """Debug information for the query processing."""
    classification_details: Optional[Dict[str, Any]] = None
    sql_query: Optional[str] = None
    rag_search_terms: Optional[List[str]] = None
    component_status: Optional[Dict[str, str]] = None
    processing_steps: Optional[List[str]] = None

class QueryResponse(BaseModel):
    """Response model for a FetiiGPT query."""
    response: str = Field(..., description="The assistant's generated response")
    session_id: str = Field(..., description="The session ID for the ongoing conversation")
    processing_type: ProcessingType = Field(..., description="How the query was processed")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    debug_info: Optional[DebugInfo] = Field(None, description="Debug information (if requested)")

class SystemStatus(BaseModel):
    """System status for FetiiGPT components."""
    overall_status: str
    components: Dict[str, Dict[str, Any]]
    database_stats: Optional[Dict[str, Any]] = None
    embeddings_stats: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    service: str = "FetiiGPT - Austin Transportation Chatbot API"
    system_status: Optional[SystemStatus] = None

class ErrorResponse(BaseModel):
    """Standard model for returning API errors."""
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class SessionInfo(BaseModel):
    """Data model for summarizing a single chat session."""
    session_id: str
    created_at: datetime
    message_count: int
    last_activity: Optional[datetime] = None
    total_queries: Optional[int] = None
    avg_processing_time: Optional[float] = None

class SessionListResponse(BaseModel):
    """Response model for the list of all chat sessions."""
    sessions: List[SessionInfo]
    total_sessions: int = Field(default=0)

class DatabaseStats(BaseModel):
    """Database statistics for FetiiGPT."""
    total_users: Optional[int] = None
    total_trips: Optional[int] = None
    total_checkins: Optional[int] = None
    date_range: Optional[Dict[str, str]] = None
    table_counts: Optional[Dict[str, int]] = None

class EmbeddingsStats(BaseModel):
    """Embeddings statistics for FetiiGPT."""
    total_vectors: Optional[int] = None
    index_name: Optional[str] = None
    embedding_model: Optional[str] = None
    last_updated: Optional[datetime] = None

class StatsResponse(BaseModel):
    """Response model for system statistics."""
    database_stats: Optional[DatabaseStats] = None
    embeddings_stats: Optional[EmbeddingsStats] = None
    system_uptime: Optional[float] = None
    total_queries_processed: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)