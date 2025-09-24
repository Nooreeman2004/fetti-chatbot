# api/routes.py
# FastAPI routes for FetiiGPT Transportation Chatbot API

import sys
import os
from pathlib import Path
import uuid
import logging
import sqlite3
from datetime import datetime
from typing import Dict
import asyncio
import traceback

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

# Import Path Setup
project_root = Path(__file__).resolve().parents[1]  # Go up one level from api/
sys.path.insert(0, str(project_root))

# Import FetiiGPT components
try:
    from chatbot.hybrid_controller import HybridController
    from chatbot.response_generator import ResponseGenerator
    from database.seed_data import DataPreparationLayer
except ImportError as e:
    logging.error(f"Failed to import FetiiGPT components: {e}")
    HybridController = None
    ResponseGenerator = None
    DataPreparationLayer = None

# Import API models
from api.models import (
    QueryRequest, QueryResponse, HealthResponse, ChatHistory, MessageRole,
    SessionInfo, SessionListResponse, ErrorResponse, SystemStatus,
    ProcessingResult, DebugInfo, ProcessingType, QueryType,
    DatabaseStats, EmbeddingsStats, StatsResponse
)

# Configuration
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["FetiiGPT"])
DB_PATH = project_root / 'chat_sessions.db'

# Global components
hybrid_controller = None
response_generator = None
startup_time = datetime.now()
query_counter = 0

# In-Memory Session Cache
sessions: Dict[str, ChatHistory] = {}

# Database Initialization
def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Table to store session metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_activity TEXT,
                    total_queries INTEGER DEFAULT 0,
                    avg_processing_time REAL DEFAULT 0.0
                )
            ''')
            # Table to store all messages with FetiiGPT-specific fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    query_type TEXT,
                    processing_time REAL,
                    processing_type TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            conn.commit()
        logger.info(f"Database initialized successfully at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise

# Component Initialization
async def init_components():
    """Initialize FetiiGPT components."""
    global hybrid_controller, response_generator
    
    try:
        logger.info("Initializing FetiiGPT components...")
        
        # Initialize Hybrid Controller
        if HybridController:
            hybrid_controller = HybridController()
            logger.info("Hybrid Controller initialized successfully")
        else:
            logger.error("HybridController class not available")
            
        # Initialize Response Generator
        if ResponseGenerator:
    response_generator = ResponseGenerator()
            logger.info("Response Generator initialized successfully")
        else:
            logger.error("ResponseGenerator class not available")
            
except Exception as e:
        logger.error(f"Failed to initialize FetiiGPT components: {e}", exc_info=True)

# Session and Database Functions
def load_session_from_db(session_id: str) -> ChatHistory:
    """Loads a session's entire chat history from the database."""
    logger.info(f"Loading session {session_id} from database.")
    history = ChatHistory()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, timestamp, query_type, processing_time FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            for row in cursor.fetchall():
                query_type = QueryType(row['query_type']) if row['query_type'] else None
                history.add_message(
                    role=MessageRole(row['role']), 
                    content=row['content'],
                    query_type=query_type,
                    processing_time=row['processing_time']
                )
        sessions[session_id] = history  # Cache the loaded session
        return history
    except Exception as e:
        logger.error(f"Failed to load session {session_id} from DB: {e}")
        return history

def get_or_create_session(session_id: str = None) -> tuple[str, ChatHistory]:
    """Retrieves an existing session or creates a new one, syncing with the database."""
    if session_id and session_id in sessions:
        logger.info(f"Found active session in memory: {session_id}")
        return session_id, sessions[session_id]

    if session_id:
        history = load_session_from_db(session_id)
        return session_id, history

    new_session_id = str(uuid.uuid4())
    logger.info(f"Creating new session: {new_session_id}")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, created_at, last_activity) VALUES (?, ?, ?)",
                (new_session_id, datetime.now().isoformat(), datetime.now().isoformat())
            )
            conn.commit()
        sessions[new_session_id] = ChatHistory()
        return new_session_id, sessions[new_session_id]
    except Exception as e:
        logger.error(f"Failed to create new session in DB: {e}")
        raise HTTPException(status_code=500, detail="Could not create a new chat session.")

def save_message_to_db(session_id: str, role: MessageRole, content: str, query_type: QueryType = None, processing_time: float = None, processing_type: ProcessingType = None):
    """Saves a single message to the SQLite database permanently."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, query_type, processing_time, processing_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, role.value, content, datetime.now().isoformat(), 
                 query_type.value if query_type else None, processing_time, 
                 processing_type.value if processing_type else None)
            )
            
            # Update session statistics
            if role == MessageRole.USER:
                cursor.execute(
                    "UPDATE sessions SET last_activity = ?, total_queries = total_queries + 1 WHERE session_id = ?",
                    (datetime.now().isoformat(), session_id)
                )
            
            conn.commit()
        logger.info(f"Saved message for session {session_id} to database.")
    except Exception as e:
        logger.error(f"Failed to save message for session {session_id}: {e}")

def update_session_stats(session_id: str, processing_time: float):
    """Update session processing time statistics."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Calculate new average processing time
            cursor.execute(
                "SELECT total_queries, avg_processing_time FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            result = cursor.fetchone()
            if result:
                total_queries, current_avg = result
                new_avg = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
                cursor.execute(
                    "UPDATE sessions SET avg_processing_time = ? WHERE session_id = ?",
                    (new_avg, session_id)
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to update session stats for {session_id}: {e}")

# API Endpoints
@router.on_event("startup")
async def startup_event():
    """Initializes the database and components when the API starts."""
    init_db()
    await init_components()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint to verify that the API is running and check component status."""
    try:
        system_status = None
        if hybrid_controller:
            system_status = SystemStatus(
                overall_status=hybrid_controller.get_system_status().get('overall_status', 'unknown'),
                components=hybrid_controller.get_system_status().get('components', {})
            )
        
        return HealthResponse(system_status=system_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy")

@router.post("/chat", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    """Main endpoint to handle user transportation data queries."""
    global query_counter
    query_counter += 1
    
    if not hybrid_controller or not response_generator:
        raise HTTPException(status_code=503, detail="FetiiGPT service is currently unavailable.")

    try:
        session_id, chat_history = get_or_create_session(request.session_id)
        
        # Save user message immediately
        save_message_to_db(session_id, MessageRole.USER, request.query)
        chat_history.add_message(MessageRole.USER, request.query)
        
        # Process query with FetiiGPT hybrid controller
        start_time = datetime.now()
        processing_result = hybrid_controller.process_query(request.query)
        
        # Get chat history for context
        llm_context = chat_history.get_llm_context()
        
        # Generate natural language response
        response_result = response_generator.generate_response(
            request.query, processing_result, llm_context
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Determine processing type
        processing_type = ProcessingType(processing_result.get('processing_type', 'SQL'))
        
        # Get the response text
        response_text = response_result.get('response', 'Sorry, I could not generate a response.')
        if not response_result.get('success'):
            # Use fallback response if generation failed
            response_text = response_generator.generate_fallback_response(request.query, processing_result)
        
        # Save assistant response
        query_type = QueryType(processing_result.get('processing_type', 'SQL'))
        save_message_to_db(
            session_id, MessageRole.ASSISTANT, response_text, 
            query_type, processing_time, processing_type
        )
        chat_history.add_message(
            MessageRole.ASSISTANT, response_text, 
            query_type, processing_time
        )
        
        # Update session statistics
        update_session_stats(session_id, processing_time)
        
        # Prepare debug info if requested
        debug_info = None
        if request.show_debug:
            debug_info = DebugInfo(
                classification_details=processing_result.get('classification', {}),
                sql_query=processing_result.get('results', {}).get('sql_query'),
                component_status=hybrid_controller.get_system_status().get('components', {}),
                processing_steps=[f"Query classified as: {processing_type.value}"]
            )
        
        return QueryResponse(
            response=response_text,
            session_id=session_id,
            processing_type=processing_type,
            processing_time=processing_time,
            debug_info=debug_info
        )
        
    except Exception as e:
        logger.error(f"Error during chat processing for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.get("/chat/history/{session_id}", response_model=ChatHistory)
async def get_chat_history(session_id: str):
    """Endpoint to retrieve the entire chat history for a given session_id."""
    logger.info(f"Fetching full chat history for session: {session_id}")
    history = load_session_from_db(session_id)
    if not history.messages:
        raise HTTPException(status_code=404, detail="Session ID not found or session has no messages.")
    return history

@router.get("/sessions", response_model=SessionListResponse, tags=["Admin"])
async def list_all_sessions():
    """Admin endpoint to list all chat sessions stored in the database."""
    logger.info("Admin request to list all sessions.")
    sessions_info = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = """
                SELECT
                    s.session_id,
                    s.created_at,
                    s.last_activity,
                    s.total_queries,
                    s.avg_processing_time,
                    (SELECT COUNT(m.message_id) FROM messages m WHERE m.session_id = s.session_id) as message_count
                FROM
                    sessions s
                ORDER BY
                    s.last_activity DESC;
            """
            cursor.execute(query)
            for row in cursor.fetchall():
                sessions_info.append(
                    SessionInfo(
                        session_id=row['session_id'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        message_count=row['message_count'],
                        last_activity=datetime.fromisoformat(row['last_activity']) if row['last_activity'] else None,
                        total_queries=row['total_queries'],
                        avg_processing_time=row['avg_processing_time']
                    )
                )
        logger.info(f"Found {len(sessions_info)} sessions to list.")
        return SessionListResponse(sessions=sessions_info, total_sessions=len(sessions_info))
    except Exception as e:
        logger.error(f"Failed to list sessions from database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve session list from the database.")

@router.get("/stats", response_model=StatsResponse, tags=["Admin"])
async def get_system_stats():
    """Get system statistics including database and embeddings info."""
    try:
        database_stats = None
        embeddings_stats = None
        
        # Get database statistics
        if hybrid_controller and hasattr(hybrid_controller, 'sql_engine'):
            try:
                sql_engine = hybrid_controller.sql_engine
                if hasattr(sql_engine, 'get_data_quality_assessment'):
                    stats = sql_engine.get_data_quality_assessment()
                    database_stats = DatabaseStats(
                        total_users=stats.get('demographics', {}).get('total_users'),
                        total_trips=stats.get('trips', {}).get('total_trips'),
                        total_checkins=stats.get('checkins', {}).get('total_checkins'),
                        date_range=stats.get('trips', {}).get('date_range'),
                        table_counts={
                            'demographics': stats.get('demographics', {}).get('total_users', 0),
                            'trips': stats.get('trips', {}).get('total_trips', 0),
                            'checkins': stats.get('checkins', {}).get('total_checkins', 0)
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to get database stats: {e}")
        
        # Get embeddings statistics
        if hybrid_controller and hasattr(hybrid_controller, 'rag_engine'):
            try:
                rag_engine = hybrid_controller.rag_engine
                if hasattr(rag_engine, 'get_index_stats'):
                    stats = rag_engine.get_index_stats()
                    embeddings_stats = EmbeddingsStats(
                        total_vectors=stats.get('total_vectors'),
                        index_name=stats.get('index_name', 'fetii-chatbot'),
                        embedding_model='text-embedding-ada-002'
                    )
            except Exception as e:
                logger.error(f"Failed to get embeddings stats: {e}")
        
        # Calculate uptime
        uptime = (datetime.now() - startup_time).total_seconds()
        
        return StatsResponse(
            database_stats=database_stats,
            embeddings_stats=embeddings_stats,
            system_uptime=uptime,
            total_queries_processed=query_counter
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve system statistics.")

@router.get("/system/status", response_model=SystemStatus, tags=["Admin"])
async def get_system_status():
    """Get detailed system component status."""
    try:
        if hybrid_controller:
            status = hybrid_controller.get_system_status()
            return SystemStatus(
                overall_status=status.get('overall_status', 'unknown'),
                components=status.get('components', {})
            )
        else:
            return SystemStatus(
                overall_status='unhealthy',
                components={'hybrid_controller': {'status': 'failed', 'error': 'Not initialized'}}
            )
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve system status.")

# Error handlers
@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for the API."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred. Please try again later."
        ).dict()
    )