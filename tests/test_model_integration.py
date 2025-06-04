"""
Test module for T5-small model integration.

This module contains tests for:
- Model service functionality
- Text generation endpoints
- Model status endpoints
- Error handling for model operations
- Integration between FastAPI and model service

Tests use pytest with async support and may include mocked model operations
for faster test execution.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import asyncio
from datetime import datetime

# Import the FastAPI app and model service
from main import app
from model_service import T5ModelService


# Create a test client for the FastAPI application
client = TestClient(app)


class TestModelService:
    """Test class for T5ModelService functionality."""
    
    def test_model_service_initialization(self):
        """
        Test that T5ModelService initializes correctly.
        
        This test verifies the initial state of the model service
        before any models are loaded.
        """
        service = T5ModelService()
        
        # Check initial state
        assert service.model_name == "t5-small"
        assert service.model is None
        assert service.tokenizer is None
        assert service.is_loaded is False
        assert service.device in ["cuda", "cpu"]
        
        # Check model readiness
        assert not service.is_model_ready()
        
        # Check model info
        info = service.get_model_info()
        assert info["model_name"] == "t5-small"
        assert info["is_loaded"] is False
        assert info["model_ready"] is False
    
    @pytest.mark.asyncio
    async def test_model_loading_success(self):
        """
        Test successful model loading.
        
        This test mocks the HuggingFace model loading to verify
        the model service handles successful loading correctly.
        """
        # Create mock objects
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Configure mock_model.to() to return itself (like real PyTorch models do)
        mock_model.to.return_value = mock_model
        
        # Patch the actual transformers imports at the module level
        with patch('model_service.T5Tokenizer') as mock_tokenizer_class, \
             patch('model_service.T5ForConditionalGeneration') as mock_model_class:
            
            # Setup the from_pretrained methods to return our mocks
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Create service and load model
            service = T5ModelService()
            await service.load_model()
            
            # Verify model loading calls were made
            mock_tokenizer_class.from_pretrained.assert_called_once_with("t5-small")
            mock_model_class.from_pretrained.assert_called_once_with("t5-small")
            
            # Verify service state
            assert service.is_loaded is True
            assert service.tokenizer == mock_tokenizer
            assert service.model == mock_model  # Now this should work!
            assert service.is_model_ready() is True
            
            # Verify model setup methods were called
            mock_model.eval.assert_called_once()
            mock_model.to.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self):
        """
        Test model loading failure handling.
        
        This test verifies that the model service handles
        loading failures gracefully.
        """
        # Patch the transformers imports to raise exception
        with patch('model_service.T5Tokenizer') as mock_tokenizer_class:
            # Setup mock to raise exception during tokenizer loading
            mock_tokenizer_class.from_pretrained.side_effect = Exception("Model loading failed")
            
            service = T5ModelService()
            
            # Verify exception is raised
            with pytest.raises(Exception, match="Model loading failed"):
                await service.load_model()
            
            # Verify service state after failure
            assert service.is_loaded is False
            assert service.model is None
            assert service.tokenizer is None
            assert not service.is_model_ready()
    
    @pytest.mark.asyncio
    @patch('model_service.T5ModelService.is_model_ready')
    async def test_generate_text_model_not_ready(self, mock_is_ready):
        """
        Test text generation when model is not ready.
        
        This test verifies proper error handling when attempting
        to generate text before the model is loaded.
        """
        mock_is_ready.return_value = False
        
        service = T5ModelService()
        
        # Attempt text generation should raise HTTPException
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await service.generate_text("test input")
        
        assert exc_info.value.status_code == 503
        assert "not loaded yet" in exc_info.value.detail


class TestModelEndpoints:
    """Test class for model-related API endpoints."""
    
    def test_model_status_endpoint(self):
        """
        Test the model status endpoint.
        
        This test verifies that the /model/status endpoint
        returns correct information about the model state.
        """
        response = client.get("/model/status")
        
        # Check HTTP status code
        assert response.status_code == 200
        
        # Parse JSON response
        data = response.json()
        
        # Verify response structure
        assert "model_name" in data
        assert "is_loaded" in data
        assert "status" in data
        assert "timestamp" in data
        
        # Verify specific values
        assert data["model_name"] == "t5-small"
        assert isinstance(data["is_loaded"], bool)
        assert data["status"] in ["ready", "loading", "error"]
        
        # Verify timestamp format
        timestamp = data["timestamp"]
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Timestamp {timestamp} is not in valid ISO format")
    
    @pytest.mark.asyncio
    @patch('main.model_service.generate_text')
    async def test_generate_text_endpoint_success(self, mock_generate):
        """
        Test successful text generation endpoint.
        
        This test mocks the model service to verify the
        /generate endpoint works correctly with valid input.
        """
        # Setup mock response
        mock_generate.return_value = ("Generated text output", 0.5)
        
        # Make request to generate endpoint
        request_data = {
            "text": "translate English to German: Hello world",
            "max_length": 50,
            "temperature": 1.0,
            "num_beams": 4
        }
        
        response = client.post("/generate", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_text" in data
        assert "input_text" in data
        assert "model_name" in data
        assert "generation_time_seconds" in data
        assert "timestamp" in data
        
        # Verify specific values
        assert data["generated_text"] == "Generated text output"
        assert data["input_text"] == request_data["text"]
        assert data["model_name"] == "t5-small"
        assert data["generation_time_seconds"] == 0.5
    
    def test_generate_text_endpoint_validation(self):
        """
        Test input validation for the text generation endpoint.
        
        This test verifies that the endpoint properly validates
        request data and rejects invalid inputs.
        """
        # Test with empty text
        response = client.post("/generate", json={"text": ""})
        assert response.status_code == 422  # Validation error
        
        # Test with text too long
        long_text = "a" * 600  # Exceeds max_length of 512
        response = client.post("/generate", json={"text": long_text})
        assert response.status_code == 422  # Validation error
        
        # Test with invalid temperature
        response = client.post("/generate", json={
            "text": "test", 
            "temperature": 3.0  # Exceeds max of 2.0
        })
        assert response.status_code == 422  # Validation error
        
        # Test with invalid max_length
        response = client.post("/generate", json={
            "text": "test", 
            "max_length": 1000  # Exceeds max of 512
        })
        assert response.status_code == 422  # Validation error
        
        # Test with invalid num_beams
        response = client.post("/generate", json={
            "text": "test", 
            "num_beams": 15  # Exceeds max of 10
        })
        assert response.status_code == 422  # Validation error
    
    def test_generate_text_endpoint_missing_fields(self):
        """
        Test text generation endpoint with missing required fields.
        
        This test verifies proper error handling when required
        fields are missing from the request.
        """
        # Test with missing text field
        response = client.post("/generate", json={})
        assert response.status_code == 422  # Validation error
        
        # Test with null text
        response = client.post("/generate", json={"text": None})
        assert response.status_code == 422  # Validation error
    
    def test_generate_text_with_default_parameters(self):
        """
        Test text generation with default parameters.
        
        This test verifies that the endpoint works correctly
        when only required fields are provided.
        """
        # This test will depend on whether the model is actually loaded
        # For now, we'll test the request structure
        request_data = {"text": "test input"}
        
        response = client.post("/generate", json=request_data)
        
        # Response could be 200 (if model loaded) or 503 (if not loaded)
        assert response.status_code in [200, 503]
        
        if response.status_code == 503:
            # Model not loaded - expected during testing
            data = response.json()
            assert "not loaded yet" in data["detail"]


class TestUpdatedHealthEndpoints:
    """Test class for updated health endpoints with model awareness."""
    
    @patch('main.model_service.is_model_ready')
    def test_readiness_check_with_model_ready(self, mock_is_ready):
        """
        Test readiness check when model is ready.
        
        This test verifies that the readiness endpoint returns
        success when the model is loaded and ready.
        """
        mock_is_ready.return_value = True
        
        response = client.get("/health/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ready"
        assert "model are ready" in data["message"]
    
    @patch('main.model_service.is_model_ready')
    def test_readiness_check_with_model_not_ready(self, mock_is_ready):
        """
        Test readiness check when model is not ready.
        
        This test verifies that the readiness endpoint returns
        503 when the model is not loaded.
        """
        mock_is_ready.return_value = False
        
        response = client.get("/health/ready")
        assert response.status_code == 503
        
        data = response.json()
        assert "not loaded yet" in data["detail"]
    
    def test_updated_root_endpoint(self):
        """
        Test that the root endpoint includes new model endpoints.
        
        This test verifies that the server information includes
        the newly added model-related endpoints.
        """
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify updated endpoints list
        expected_endpoints = ["/", "/health", "/health/ready", "/health/live", "/model/status", "/generate"]
        assert data["endpoints"] == expected_endpoints
        
        # Verify updated version
        assert data["version"] == "2.0.0"


class TestModelIntegrationFlow:
    """Test class for end-to-end model integration workflows."""
    
    def test_server_startup_sequence(self):
        """
        Test the server startup sequence with model loading.
        
        This test verifies that the server can start up properly
        and handle the model loading process.
        """
        # Test that server is responsive
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test model status endpoint
        response = client.get("/model/status")
        assert response.status_code == 200
        
        # Model might or might not be loaded depending on test environment
        data = response.json()
        assert data["status"] in ["ready", "loading", "error"]
    
    @patch('main.model_service.generate_text')
    def test_complete_generation_workflow(self, mock_generate):
        """
        Test a complete text generation workflow.
        
        This test simulates a full workflow from request to response
        for the text generation feature.
        """
        # Setup mock
        mock_generate.return_value = ("Translated: Hallo Welt", 0.3)
        
        # Step 1: Check server is running
        response = client.get("/health")
        assert response.status_code == 200
        
        # Step 2: Check model status
        response = client.get("/model/status")
        assert response.status_code == 200
        
        # Step 3: Make generation request
        request_data = {
            "text": "translate English to German: Hello world",
            "max_length": 50
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["generated_text"] == "Translated: Hallo Welt"
        assert data["input_text"] == request_data["text"]
        assert isinstance(data["generation_time_seconds"], float)
    
    def test_concurrent_model_requests(self):
        """
        Test handling of concurrent requests to model endpoints.
        
        This test verifies that the server can handle multiple
        simultaneous requests to model-related endpoints.
        """
        import concurrent.futures
        
        def make_status_request():
            """Helper function to make a model status request."""
            return client.get("/model/status")
        
        # Make concurrent requests to model status endpoint
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_status_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all requests succeeded
        assert len(responses) == 10
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "model_name" in data
            assert "status" in data