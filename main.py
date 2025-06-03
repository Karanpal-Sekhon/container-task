"""
Main FastAPI application for HuggingFace model inference server.

This module sets up a basic FastAPI server with health check endpoints
and establishes the foundation for ML model inference capabilities.
The server is designed to handle async requests efficiently using FastAPI's
built-in async support and Uvicorn's event loop.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application with metadata
app = FastAPI(
    title="HuggingFace Model Inference Server",
    description="A FastAPI server for processing inference requests using HuggingFace models",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI documentation endpoint
    redoc_url="/redoc"  # ReDoc documentation endpoint
)


# Pydantic models for request/response validation
class HealthResponse(BaseModel):
    """Response model for health check endpoints."""
    status: str
    timestamp: str
    message: str


class ServerInfo(BaseModel):
    """Response model for server information."""
    name: str
    version: str
    description: str
    endpoints: list


# Application startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    
    This function runs when the FastAPI application starts up.
    It's where we'll initialize resources, load models, and set up
    any necessary background tasks.
    """
    logger.info("🚀 Starting HuggingFace Model Inference Server")
    logger.info("Server initialization complete")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    
    This function runs when the FastAPI application shuts down.
    It's where we'll clean up resources, save state, and perform
    any necessary cleanup tasks.
    """
    logger.info("🛑 Shutting down HuggingFace Model Inference Server")
    logger.info("Cleanup complete")


# Basic health check endpoints
@app.get("/", response_model=ServerInfo)
async def root():
    """
    Root endpoint providing basic server information.
    
    Returns:
        ServerInfo: Basic information about the server and available endpoints
    """
    return ServerInfo(
        name="HuggingFace Model Inference Server",
        version="1.0.0",
        description="FastAPI server for ML model inference using HuggingFace transformers",
        endpoints=["/", "/health", "/health/ready", "/health/live"]
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    This endpoint provides a simple health status of the server.
    It can be used by load balancers and monitoring systems to
    determine if the server is responding to requests.
    
    Returns:
        HealthResponse: Current health status of the server
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        message="Server is running and accepting requests"
    )


@app.get("/health/ready", response_model=HealthResponse)
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/container orchestration.
    
    This endpoint indicates whether the server is ready to accept traffic.
    In future iterations, this will check if models are loaded and
    all dependencies are available.
    
    Returns:
        HealthResponse: Readiness status of the server
    """
    # In future PRs, we'll add actual readiness checks here
    # For now, we assume the server is ready if it's responding
    
    try:
        # Simulate a quick async operation to test server responsiveness
        await asyncio.sleep(0.001)
        
        return HealthResponse(
            status="ready",
            timestamp=datetime.utcnow().isoformat(),
            message="Server is ready to accept requests"
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Server is not ready to accept requests"
        )


@app.get("/health/live", response_model=HealthResponse)
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes/container orchestration.
    
    This endpoint indicates whether the server process is alive and functioning.
    If this endpoint fails, it typically means the container should be restarted.
    
    Returns:
        HealthResponse: Liveness status of the server
    """
    try:
        # Basic liveness check - test that the server can perform async operations
        start_time = datetime.utcnow()
        await asyncio.sleep(0.001)
        
        return HealthResponse(
            status="alive",
            timestamp=start_time.isoformat(),
            message="Server process is alive and functioning"
        )
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Server liveness check failed"
        )


# Global exception handler for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    
    This handler catches any unhandled exceptions and returns a
    standardized error response instead of exposing internal errors.
    
    Args:
        request: The FastAPI request object
        exc: The exception that was raised
        
    Returns:
        JSONResponse: Standardized error response
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Development server runner (for local testing)
if __name__ == "__main__":
    import uvicorn
    
    # Run the server with uvicorn for development
    # In production, this will be handled by the Docker container
    logger.info("Starting development server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )