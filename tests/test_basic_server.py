"""
Test module for basic FastAPI server functionality.

This module contains tests for:
- Server startup and basic functionality
- Health check endpoints
- Root endpoint information
- Error handling
- Response validation

Tests use pytest with async support and FastAPI's TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import asyncio
from datetime import datetime

# Import the FastAPI app from our main module
from main import app


# Create a test client for the FastAPI application
client = TestClient(app)


class TestServerBasics:
    """Test class for basic server functionality."""
    
    def test_server_startup(self):
        """
        Test that the server starts up successfully.
        
        This test verifies that the FastAPI application can be instantiated
        and is ready to handle requests.
        """
        # The fact that we can create a TestClient means the app loaded successfully
        assert app is not None
        assert app.title == "HuggingFace Model Inference Server"
        assert app.version == "1.0.0"
    
    def test_root_endpoint(self):
        """
        Test the root endpoint (/) returns correct server information.
        
        This test verifies:
        - The endpoint returns HTTP 200
        - Response contains expected server metadata
        - Response structure matches ServerInfo model
        """
        response = client.get("/")
        
        # Check HTTP status code
        assert response.status_code == 200
        
        # Parse JSON response
        data = response.json()
        
        # Verify response structure and content
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data
        
        # Verify specific values
        assert data["name"] == "HuggingFace Model Inference Server"
        assert data["version"] == "1.0.0"
        assert isinstance(data["endpoints"], list)
        assert len(data["endpoints"]) > 0
        
        # Verify expected endpoints are listed
        expected_endpoints = ["/", "/health", "/health/ready", "/health/live"]
        for endpoint in expected_endpoints:
            assert endpoint in data["endpoints"]


class TestHealthEndpoints:
    """Test class for health check endpoints."""
    
    def test_basic_health_check(self):
        """
        Test the basic health check endpoint (/health).
        
        This test verifies:
        - Endpoint returns HTTP 200
        - Response contains required health information
        - Response structure matches HealthResponse model
        """
        response = client.get("/health")
        
        # Check HTTP status code
        assert response.status_code == 200
        
        # Parse JSON response
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "timestamp" in data
        assert "message" in data
        
        # Verify content
        assert data["status"] == "healthy"
        assert "Server is running" in data["message"]
        
        # Verify timestamp is a valid ISO format
        timestamp = data["timestamp"]
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Timestamp {timestamp} is not in valid ISO format")
    
    def test_readiness_check(self):
        """
        Test the readiness check endpoint (/health/ready).
        
        This test verifies:
        - Endpoint returns HTTP 200 when server is ready
        - Response indicates readiness status
        - Async operations are working correctly
        """
        response = client.get("/health/ready")
        
        # Check HTTP status code
        assert response.status_code == 200
        
        # Parse JSON response
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "timestamp" in data
        assert "message" in data
        
        # Verify readiness status
        assert data["status"] == "ready"
        assert "ready to accept requests" in data["message"]
    
    def test_liveness_check(self):
        """
        Test the liveness check endpoint (/health/live).
        
        This test verifies:
        - Endpoint returns HTTP 200 when server is alive
        - Response indicates liveness status
        - Server can handle async operations
        """
        response = client.get("/health/live")
        
        # Check HTTP status code
        assert response.status_code == 200
        
        # Parse JSON response
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "timestamp" in data
        assert "message" in data
        
        # Verify liveness status
        assert data["status"] == "alive"
        assert "alive and functioning" in data["message"]
    
    @patch('main.asyncio.sleep')
    def test_readiness_check_with_exception(self, mock_sleep):
        """
        Test readiness check behavior when an exception occurs.
        
        This test simulates an error during the readiness check
        to verify proper error handling.
        """
        # Configure mock to raise an exception
        mock_sleep.side_effect = Exception("Simulated error")
        
        response = client.get("/health/ready")
        
        # Should return HTTP 503 (Service Unavailable) on error
        assert response.status_code == 503
        
        # Parse JSON response
        data = response.json()
        
        # Verify error response structure
        assert "detail" in data
        assert "not ready" in data["detail"]
    
    @patch('main.asyncio.sleep')
    def test_liveness_check_with_exception(self, mock_sleep):
        """
        Test liveness check behavior when an exception occurs.
        
        This test simulates an error during the liveness check
        to verify proper error handling.
        """
        # Configure mock to raise an exception
        mock_sleep.side_effect = Exception("Simulated error")
        
        response = client.get("/health/live")
        
        # Should return HTTP 500 (Internal Server Error) on error
        assert response.status_code == 500
        
        # Parse JSON response
        data = response.json()
        
        # Verify error response structure
        assert "detail" in data
        assert "liveness check failed" in data["detail"]


class TestResponseValidation:
    """Test class for response validation and data models."""
    
    def test_health_response_model_validation(self):
        """
        Test that health endpoint responses conform to the HealthResponse model.
        
        This test verifies that all health endpoints return data that
        matches the expected Pydantic model structure.
        """
        endpoints_to_test = ["/health", "/health/ready", "/health/live"]
        
        for endpoint in endpoints_to_test:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            data = response.json()
            
            # All health responses should have these fields
            required_fields = ["status", "timestamp", "message"]
            for field in required_fields:
                assert field in data, f"Field '{field}' missing from {endpoint} response"
                assert isinstance(data[field], str), f"Field '{field}' should be string in {endpoint}"
    
    def test_server_info_response_model_validation(self):
        """
        Test that the root endpoint response conforms to the ServerInfo model.
        
        This test verifies that the root endpoint returns data that
        matches the expected Pydantic model structure.
        """
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify required fields and types
        assert isinstance(data["name"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["description"], str)
        assert isinstance(data["endpoints"], list)
        
        # Verify endpoints list contains strings
        for endpoint in data["endpoints"]:
            assert isinstance(endpoint, str)
            assert endpoint.startswith("/")


class TestAsyncFunctionality:
    """Test class for async functionality and concurrency."""
    
    def test_concurrent_health_checks(self):
        """
        Test that multiple concurrent requests to health endpoints work correctly.
        
        This test verifies that the server can handle multiple simultaneous
        requests without issues, demonstrating async capability.
        """
        import concurrent.futures
        import threading
        
        def make_health_request():
            """Helper function to make a health check request."""
            return client.get("/health")
        
        # Create multiple threads to make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit 10 concurrent requests
            futures = [executor.submit(make_health_request) for _ in range(10)]
            
            # Collect all responses
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all requests succeeded
        assert len(responses) == 10
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    def test_mixed_endpoint_concurrent_access(self):
        """
        Test concurrent access to different endpoints.
        
        This test verifies that the server can handle requests to different
        endpoints simultaneously without conflicts.
        """
        import concurrent.futures
        
        endpoints = ["/", "/health", "/health/ready", "/health/live"]
        
        def make_request(endpoint):
            """Helper function to make a request to a specific endpoint."""
            return client.get(endpoint)
        
        # Make concurrent requests to different endpoints
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(make_request, endpoint) for endpoint in endpoints]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all requests succeeded
        assert len(responses) == 4
        for response in responses:
            assert response.status_code == 200


class TestErrorHandling:
    """Test class for error handling and edge cases."""
    
    def test_nonexistent_endpoint(self):
        """
        Test that requests to non-existent endpoints return 404.
        
        This test verifies proper handling of requests to endpoints
        that don't exist on the server.
        """
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_http_methods(self):
        """
        Test that invalid HTTP methods return appropriate errors.
        
        This test verifies that endpoints only respond to their
        designated HTTP methods.
        """
        # Health endpoints should only accept GET requests
        response = client.post("/health")
        assert response.status_code == 405  # Method Not Allowed
        
        response = client.put("/health")
        assert response.status_code == 405
        
        response = client.delete("/health")
        assert response.status_code == 405


class TestDocumentationEndpoints:
    """Test class for API documentation endpoints."""
    
    def test_swagger_docs_accessible(self):
        """
        Test that Swagger UI documentation is accessible.
        
        This test verifies that the FastAPI automatic documentation
        is available at the /docs endpoint.
        """
        response = client.get("/docs")
        assert response.status_code == 200
        # Should return HTML content for Swagger UI
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_docs_accessible(self):
        """
        Test that ReDoc documentation is accessible.
        
        This test verifies that the alternative documentation
        is available at the /redoc endpoint.
        """
        response = client.get("/redoc")
        assert response.status_code == 200
        # Should return HTML content for ReDoc
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_schema_accessible(self):
        """
        Test that OpenAPI schema is accessible.
        
        This test verifies that the OpenAPI schema JSON is available,
        which is used by documentation tools and API clients.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        # Should return JSON content
        assert "application/json" in response.headers.get("content-type", "")
        
        # Parse and verify basic schema structure
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Verify our endpoints are documented
        paths = schema["paths"]
        assert "/" in paths
        assert "/health" in paths
        assert "/health/ready" in paths
        assert "/health/live" in paths