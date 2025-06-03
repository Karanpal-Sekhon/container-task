# HuggingFace Model Inference Server

A high-performance FastAPI-based server for processing inference requests from pre-trained models in the HuggingFace model hub. This server is designed to handle multiple parallel requests efficiently using async architecture and will be containerized for production deployment.

## 🎯 Project Overview

This application provides a REST API for running inference on HuggingFace transformer models, starting with the T5-small model for text-to-text generation. The server is built with production-ready features including health checks, error handling, logging, and comprehensive testing.

## 🚀 Key Features

- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Async Architecture**: Built for handling concurrent requests efficiently
- **Health Monitoring**: Kubernetes-ready health check endpoints
- **Production Ready**: Comprehensive error handling, logging, and validation
- **Containerized Deployment**: Docker support with nginx reverse proxy
- **Comprehensive Testing**: Full test suite with pytest

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd container-task
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

**Development Mode:**
```bash
python run_server.py
```

**Direct with uvicorn:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

## 📚 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔍 API Endpoints

### Health Check Endpoints

| Endpoint | Purpose | Use Case |
|----------|---------|----------|
| `GET /` | Server information | Basic server metadata and available endpoints |
| `GET /health` | Basic health check | Load balancer health monitoring |
| `GET /health/ready` | Readiness probe | Kubernetes readiness check |
| `GET /health/live` | Liveness probe | Kubernetes liveness check |

### Example Requests

**Server Information:**
```bash
curl http://localhost:8000/
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Readiness Check:**
```bash
curl http://localhost:8000/health/ready
```

## 🧪 Testing

The project includes a comprehensive test suite using pytest with async support. Tests are organized by functionality and include unit tests, integration tests, and error handling scenarios.

### Test Categories

#### 🔧 **Unit Tests**
- **Server Basics**: Application startup, configuration validation
- **Health Endpoints**: Individual endpoint functionality and response validation
- **Response Models**: Pydantic model validation and data structure verification
- **Error Handling**: Exception handling and error response formatting

#### 🔗 **Integration Tests**
- **Async Functionality**: Concurrent request handling and performance
- **API Documentation**: Swagger UI, ReDoc, and OpenAPI schema accessibility
- **End-to-End Workflows**: Complete request/response cycles

#### 🚨 **Error Scenario Tests**
- **Invalid Endpoints**: 404 error handling
- **Invalid Methods**: HTTP method validation
- **Exception Handling**: Simulated failures and recovery
- **Edge Cases**: Boundary conditions and unexpected inputs

### Running Tests

**Run All Tests:**
```bash
pytest
```

**Run with Verbose Output:**
```bash
pytest -v
```

**Run Specific Test File:**
```bash
pytest tests/test_basic_server.py -v
```

**Run Specific Test Class:**
```bash
pytest tests/test_basic_server.py::TestHealthEndpoints -v
```

**Run Specific Test Method:**
```bash
pytest tests/test_basic_server.py::TestHealthEndpoints::test_basic_health_check -v
```

**Run Tests with Coverage:**
```bash
pytest --cov=main --cov-report=html
```

**Run Tests by Marker:**
```bash
pytest -m "health" -v      # Run only health-related tests
pytest -m "api" -v         # Run only API endpoint tests
```

### Test Structure

The test suite includes:

- **TestServerBasics**: Basic server functionality and startup tests
- **TestHealthEndpoints**: Health check endpoint validation
- **TestResponseValidation**: Response model and data structure verification
- **TestAsyncFunctionality**: Concurrent request handling tests
- **TestErrorHandling**: Error scenarios and exception handling
- **TestDocumentationEndpoints**: API documentation accessibility tests

### Test Execution Examples

**Basic Test Run:**
```bash
$ pytest tests/test_basic_server.py -v

tests/test_basic_server.py::TestServerBasics::test_server_startup PASSED
tests/test_basic_server.py::TestServerBasics::test_root_endpoint PASSED
tests/test_basic_server.py::TestHealthEndpoints::test_basic_health_check PASSED
tests/test_basic_server.py::TestHealthEndpoints::test_readiness_check PASSED
tests/test_basic_server.py::TestHealthEndpoints::test_liveness_check PASSED
...
```

**Concurrent Request Test:**
```bash
$ pytest tests/test_basic_server.py::TestAsyncFunctionality::test_concurrent_health_checks -v
```

## 🔧 Troubleshooting

### Common Issues

**Server Won't Start:**
- Ensure virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify port 8000 is not in use

**Tests Failing:**
- Run tests in isolation: `pytest tests/test_basic_server.py::test_name -v`
- Check that the server can start: `python run_server.py`
- Ensure all dependencies are installed for testing

**Import Errors:**
- Verify you're in the correct directory
- Check that the virtual environment is activated
- Ensure the project structure matches the expected layout