#!/usr/bin/env python3
"""
Development server runner for HuggingFace Model Inference Server.

This script provides a convenient way to start the FastAPI server
for development and testing purposes. It uses uvicorn with
development-friendly settings.

Usage:
    python run_server.py

The server will start on http://localhost:8000 with auto-reload enabled.
"""

import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to start the development server.
    
    This function configures and starts the uvicorn server with
    development-friendly settings including auto-reload and
    detailed logging.
    """
    logger.info("🚀 Starting HuggingFace Model Inference Server in development mode")
    logger.info("📖 API Documentation available at: http://localhost:8000/docs")
    logger.info("📚 Alternative docs available at: http://localhost:8000/redoc")
    logger.info("🔧 Server will auto-reload on code changes")
    
    # Start the server with development settings
    uvicorn.run(
        "main:app",                    # Import path to FastAPI app
        host="0.0.0.0",               # Listen on all interfaces
        port=8000,                    # Development port
        reload=True,                  # Auto-reload on file changes
        reload_dirs=["."],            # Watch current directory for changes
        log_level="info",             # Logging level
        access_log=True,              # Enable access logging
        use_colors=True,              # Colored log output
        app_dir=".",                  # Application directory
    )

if __name__ == "__main__":
    main()