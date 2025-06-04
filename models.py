"""
Pydantic models for the HuggingFace Model Inference Server.

This module contains all the request and response models used throughout
the application for data validation, serialization, and API documentation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


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


class TextGenerationRequest(BaseModel):
    """
    Request model for text generation.
    
    This model validates incoming requests for the T5-small text generation
    endpoint. It includes the input text and optional generation parameters.
    """
    text: str = Field(
        ..., 
        min_length=1,
        max_length=512,
        description="Input text for text-to-text generation",
        json_schema_extra={"example": "translate English to German: The house is wonderful."}
    )
    max_length: Optional[int] = Field(
        default=150,
        ge=10,
        le=512,
        description="Maximum length of generated text"
    )
    temperature: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Temperature for text generation (higher = more creative)"
    )
    num_beams: Optional[int] = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of beams for beam search"
    )


class TextGenerationResponse(BaseModel):
    """
    Response model for text generation.
    
    This model structures the response from the T5-small text generation
    endpoint, including the generated text and metadata about the generation.
    """
    generated_text: str
    input_text: str
    model_name: str
    generation_time_seconds: float
    timestamp: str


class ModelStatus(BaseModel):
    """Response model for model status information."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    is_loaded: bool
    status: str
    timestamp: str