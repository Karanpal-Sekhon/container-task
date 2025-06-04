"""
Main FastAPI application for HuggingFace model inference server.

This module sets up a basic FastAPI server with health check endpoints
and establishes the foundation for ML model inference capabilities.
The server is designed to handle async requests efficiently using FastAPI's
built-in async support and Uvicorn's event loop.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import logging
from datetime import datetime
import time

# Import our custom modules
from models import HealthResponse, ServerInfo, TextGenerationRequest, TextGenerationResponse, ModelStatus
from model_service import T5ModelService

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the T5 model service
model_service = T5ModelService()

# Initialize FastAPI application with metadata
app = FastAPI(
    title="HuggingFace Model Inference Server",
    description="A FastAPI server for processing inference requests using HuggingFace models",
    version="2.0.0",
    docs_url="/docs",  # Swagger UI documentation endpoint
    redoc_url="/redoc"  # ReDoc documentation endpoint
)


# Application startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    
    This function runs when the FastAPI application starts up.
    It loads the T5-small model and initializes all necessary resources.
    """
    logger.info("🚀 Starting HuggingFace Model Inference Server")
    
    try:
        # Load the T5-small model asynchronously
        await model_service.load_model()
        logger.info("✅ Model loading completed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model during startup: {str(e)}")
        # Continue startup even if model loading fails
        # The model status endpoint will show the error state
    
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
        version="2.0.0",
        description="FastAPI server for ML model inference using HuggingFace transformers",
        endpoints=["/", "/health", "/health/ready", "/health/live", "/model/status", "/generate"]
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
    It checks if the model is loaded and ready for inference.
    
    Returns:
        HealthResponse: Readiness status of the server
    """
    try:
        # Check if the model is loaded and ready
        if model_service.is_model_ready():
            return HealthResponse(
                status="ready",
                timestamp=datetime.utcnow().isoformat(),
                message="Server and model are ready to accept requests"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded yet. Server is not ready for inference requests."
            )
    except HTTPException:
        raise
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


# Model-specific endpoints
@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """
    Get the current status of the T5-small model.
    
    This endpoint provides information about whether the model is loaded
    and ready for inference operations.
    
    Returns:
        ModelStatus: Current status of the model
    """
    model_info = model_service.get_model_info()
    
    status = "ready" if model_info["model_ready"] else "loading" if not model_info["is_loaded"] else "error"
    
    return ModelStatus(
        model_name=model_info["model_name"],
        is_loaded=model_info["is_loaded"],
        status=status,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """
    Generate text using the T5-small model.
    
    This endpoint accepts text input and generates output using the T5-small
    model for text-to-text generation tasks such as translation, summarization,
    and question answering.
    
    Args:
        request: TextGenerationRequest containing input text and generation parameters
        
    Returns:
        TextGenerationResponse: Generated text with metadata
        
    Raises:
        HTTPException: If model is not ready or generation fails
    """
    logger.info(f"Received text generation request: '{request.text[:50]}...'")
    
    try:
        # Generate text using the model service
        generated_text, generation_time = await model_service.generate_text(
            input_text=request.text,
            max_length=request.max_length,
            temperature=request.temperature,
            num_beams=request.num_beams
        )
        
        # Create response
        response = TextGenerationResponse(
            generated_text=generated_text,
            input_text=request.text,
            model_name=model_service.model_name,
            generation_time_seconds=generation_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Successfully generated text in {generation_time:.3f}s")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (from model_service)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in text generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
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