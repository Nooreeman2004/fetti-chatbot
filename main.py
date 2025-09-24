# main.py
# FastAPI main application for FetiiGPT Transportation Chatbot

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import API components
from api.routes import router
from api.models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fetii_chatbot.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables for application state
app_startup_time = None
app_shutdown_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global app_startup_time
    app_startup_time = __import__('datetime').datetime.now()
    
    # Startup
    logger.info("🚀 Starting FetiiGPT Transportation Chatbot API...")
    logger.info(f"📁 Project root: {project_root}")
    logger.info(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    # Check for required environment variables
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
        logger.warning("Some features may not work properly without these variables.")
    else:
        logger.info("✅ All required environment variables are set")
    
    # Check if database exists
    db_path = project_root / 'database' / 'transportation.db'
    if db_path.exists():
        logger.info(f"✅ Database found at: {db_path}")
    else:
        logger.warning(f"⚠️ Database not found at: {db_path}")
        logger.warning("The application will attempt to initialize the database on first use.")
    
    # Check if embeddings exist
    embeddings_config = project_root / 'embeddings' / 'openai_pinecone_config.json'
    if embeddings_config.exists():
        logger.info("✅ Pinecone embeddings configuration found")
    else:
        logger.warning("⚠️ Pinecone embeddings configuration not found")
        logger.warning("RAG functionality may be limited without embeddings.")
    
    logger.info("🎉 FetiiGPT API startup completed successfully!")
    
    yield
    
    # Shutdown
    global app_shutdown_time
    app_shutdown_time = __import__('datetime').datetime.now()
    logger.info("🛑 Shutting down FetiiGPT Transportation Chatbot API...")
    
    if app_startup_time and app_shutdown_time:
        uptime = (app_shutdown_time - app_startup_time).total_seconds()
        logger.info(f"⏱️ Total uptime: {uptime:.2f} seconds")
    
    logger.info("👋 FetiiGPT API shutdown completed")

# Create FastAPI application
app = FastAPI(
    title="FetiiGPT - Austin Transportation Chatbot API",
    description="""
    ## FetiiGPT Transportation Data Analytics API
    
    A sophisticated hybrid chatbot system that combines SQL query processing, RAG (Retrieval-Augmented Generation), 
    and natural language understanding to provide intelligent insights from Fetii's group rideshare data in Austin, Texas.
    
    ### Features:
    - **Hybrid Query Processing**: Automatically classifies and routes queries to SQL, RAG, or hybrid processing
    - **Natural Language to SQL**: Converts natural language questions into SQL queries using OpenAI GPT
    - **Semantic Search**: RAG-based retrieval for contextual information and entity descriptions
    - **Interactive Chat**: Real-time chat interface with session management
    - **Comprehensive Analytics**: Support for complex transportation data analysis
    - **Robust Error Handling**: Graceful fallbacks and detailed error reporting
    
    ### Data Schema:
    The system works with three main data tables:
    - **CustomerDemographics**: User information (UserID, Age, Gender, Name, CreatedAt)
    - **TripData**: Trip details (TripID, BookingUserID, Pickup/Dropoff locations, TripDate, etc.)
    - **CheckedInUsers**: Check-in records (CheckInID, UserID, TripID, CheckInTime, CheckInStatus)
    
    ### Sample Queries:
    - "How many trips were taken last month?"
    - "What's the average trip duration?"
    - "Show me the top 5 users by trip count"
    - "Which locations are most popular for 18-24 year-olds?"
    - "How many groups went to Moody Center last month?"
    """,
    version="1.0.0",
    contact={
        "name": "FetiiGPT Support",
        "email": "support@fetii.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:8501",  # Streamlit development server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "https://yourdomain.com",  # Add your production domain here
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to FetiiGPT - Austin Transportation Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "features": [
            "Hybrid Query Processing (SQL + RAG)",
            "Natural Language to SQL Translation",
            "Semantic Search with Pinecone",
            "Session Management",
            "Real-time Chat Interface",
            "Comprehensive Analytics"
        ],
        "sample_queries": [
            "How many trips were taken last month?",
            "What's the average trip duration?",
            "Show me the top 5 users by trip count",
            "Which locations are most popular for 18-24 year-olds?"
        ]
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred. Please try again later."
        ).dict()
    )

# Health check endpoint (additional to the one in routes)
@app.get("/health", tags=["Health"])
async def simple_health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "FetiiGPT Transportation Chatbot API",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

# Development server configuration
if __name__ == "__main__":
    # Configuration for development
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"🚀 Starting FetiiGPT API server...")
    logger.info(f"🌐 Host: {host}")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"🔄 Reload: {reload}")
    logger.info(f"📊 Log Level: {log_level}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logger.info(f"🔍 Health Check: http://{host}:{port}/health")
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)
