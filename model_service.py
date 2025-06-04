"""
Model service for T5-small text generation.

This module contains the T5ModelService class which handles loading,
initialization, and inference operations for the HuggingFace T5-small model.
"""

import asyncio
import logging
import time
from typing import Optional
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class T5ModelService:
    """
    Service class for T5-small model operations.
    
    This class handles model loading, tokenization, and text generation
    using the HuggingFace T5-small model. It provides a clean interface
    for text-to-text generation tasks.
    """
    
    def __init__(self):
        self.model_name = "t5-small"
        self.model: Optional[T5ForConditionalGeneration] = None
        self.tokenizer: Optional[T5Tokenizer] = None
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    async def load_model(self):
        """
        Load the T5-small model and tokenizer.
        
        This method initializes the T5ForConditionalGeneration model
        and T5Tokenizer from HuggingFace transformers. Loading is done
        asynchronously to avoid blocking the server startup.
        """
        try:
            logger.info(f"Loading {self.model_name} model on {self.device}...")
            
            # Load tokenizer and model from HuggingFace
            # Using asyncio.to_thread to run the blocking operations in a thread pool
            self.tokenizer = await asyncio.to_thread(
                T5Tokenizer.from_pretrained, self.model_name
            )
            self.model = await asyncio.to_thread(
                T5ForConditionalGeneration.from_pretrained, self.model_name
            )
            
            # Set model to evaluation mode for inference
            self.model.eval()
            
            # Move to GPU if available
            self.model = self.model.to(self.device)
            
            self.is_loaded = True
            logger.info(f"✅ {self.model_name} model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_name} model: {str(e)}")
            self.is_loaded = False
            raise e
    
    async def generate_text(
        self, 
        input_text: str, 
        max_length: int = 150,
        temperature: float = 1.0,
        num_beams: int = 4
    ) -> tuple[str, float]:
        """
        Generate text using the T5-small model.
        
        Args:
            input_text: Input text for generation
            max_length: Maximum length of generated text
            temperature: Temperature for generation (higher = more creative)
            num_beams: Number of beams for beam search
            
        Returns:
            tuple: (generated_text, generation_time_seconds)
            
        Raises:
            HTTPException: If model is not loaded or generation fails
        """
        if not self.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded yet. Please wait for model initialization."
            )
        
        start_time = time.time()
        
        try:
            # Tokenize input text
            input_ids = await asyncio.to_thread(
                self.tokenizer.encode,
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Move input to same device as model
            input_ids = input_ids.to(self.device)
            
            # Generate text using the model
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = await asyncio.to_thread(
                self.tokenizer.decode,
                outputs[0],
                skip_special_tokens=True
            )
            
            generation_time = time.time() - start_time
            
            logger.info(f"Generated text in {generation_time:.3f}s: '{generated_text[:50]}...'")
            
            return generated_text, generation_time
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Text generation failed: {str(e)}"
            )
    
    def is_model_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.is_loaded and self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "model_ready": self.is_model_ready()
        }