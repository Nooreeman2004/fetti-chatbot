#!/usr/bin/env python3
"""
FetiiGPT FastAPI Server Startup Script
This script provides an easy way to start the FetiiGPT API server with proper configuration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fetii_api.log')
        ]
    )

def check_environment():
    """Check if required environment variables are set."""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ Warning: Missing environment variables: {missing_vars}")
        print("Some features may not work properly without these variables.")
        return False
    
    print("✅ All required environment variables are set")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ FastAPI dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install requirements: pip install -r requirements-api.txt")
        return False

def main():
    """Main function to start the FetiiGPT API server."""
    parser = argparse.ArgumentParser(description="Start FetiiGPT Transportation Chatbot API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], 
                       help="Log level (default: info)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--check-only", action="store_true", help="Only check environment and dependencies")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("🚀 FetiiGPT Transportation Chatbot API")
    print("=" * 50)
    
    # Check environment
    env_ok = check_environment()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if args.check_only:
        if env_ok and deps_ok:
            print("✅ All checks passed!")
            sys.exit(0)
        else:
            print("❌ Some checks failed!")
            sys.exit(1)
    
    if not deps_ok:
        print("❌ Cannot start server due to missing dependencies")
        sys.exit(1)
    
    # Set environment variables
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    os.environ["RELOAD"] = str(args.reload).lower()
    os.environ["LOG_LEVEL"] = args.log_level
    
    print(f"🌐 Starting server on {args.host}:{args.port}")
    print(f"🔄 Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print(f"📊 Log level: {args.log_level}")
    print(f"👥 Workers: {args.workers}")
    print("=" * 50)
    
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers if not args.reload else 1,  # Workers don't work with reload
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
