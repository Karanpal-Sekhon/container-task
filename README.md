# HuggingFace Model Inference Server

A production-ready containerized inference server for processing requests from any pre-trained model in the HuggingFace model hub. The system uses nginx load balancing with multiple FastAPI workers to handle parallel requests efficiently, featuring T5-small for versatile text-to-text generation tasks.

## 🎯 Project Overview

This application provides a REST API for running inference on HuggingFace transformer models, specifically demonstrating T5-small for translation, summarization, question answering, and text completion. The server is built with enterprise-grade features including Docker containerization, nginx load balancing, health monitoring, comprehensive testing, and a complete parallel inference demonstration.

## 🏗️ Architecture

The system consists of four key components:

1. **Docker Multi-Stage Build**: Optimized container with pre-cached T5-small model
2. **nginx Load Balancer**: Handles multiple concurrent connections with rate limiting and health checks
3. **FastAPI Application**: Async Python server with Kubernetes-ready monitoring endpoints
4. **uvicorn Workers**: Multiple worker processes for parallel request processing (2 workers by default)

## 🚀 Key Features

- **Containerized Deployment**: Full Docker stack with nginx reverse proxy
- **Parallel Processing**: Multiple uvicorn workers behind nginx load balancer
- **Model Pre-loading**: T5-small cached in container for instant startup
- **Production Ready**: Health checks, rate limiting, error handling, and logging
- **Async Architecture**: Built for handling concurrent requests efficiently
- **Comprehensive Demo**: Jupyter notebook demonstrating parallel inference across multiple NLP tasks
- **Extensive Testing**: Full test suite with pytest for both basic functionality and model integration

## 📋 Prerequisites

- **Docker & Docker Compose**: For containerized deployment
- **Python 3.10+**: For local development
- **Virtual environment**: Recommended for local testing
- **4GB+ RAM**: For T5-small model loading

## 🛠️ Installation & Setup

### Option 1: Containerized Deployment (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd container-task
   ```

2. **Build and start the containers**
   ```bash
   docker compose build
   docker compose up
   ```

3. **Access the service**
   - **API Server**: http://localhost (nginx proxy)
   - **API Documentation**: http://localhost/docs
   - **Health Check**: http://localhost/health

### Option 2: Local Development

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

4. **Configure Jupyter notebook kernel (for demo)**
   ```bash
   pip install jupyter ipykernel
   python -m ipykernel install --user --name=container-task-venv --display-name="Container Task (venv)"
   ```

## 🚀 How to Run

### Production Deployment (Docker)
```bash
# Build containers (first time only)
docker compose build

# Start the full stack
docker compose up

# Stop the containers
docker compose down
```

### Development Mode (Local)
```bash
# Activate virtual environment
source venv/bin/activate

# Start development server
python run_server.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Demo Notebook
```bash
# Start Jupyter (after setting up kernel)
source venv/bin/activate
jupyter notebook demo_notebook.ipynb
```

## 📚 API Documentation

### Containerized Deployment (nginx proxy)
- **Swagger UI**: http://localhost/docs
- **ReDoc**: http://localhost/redoc
- **OpenAPI Schema**: http://localhost/openapi.json

### Local Development
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔍 API Endpoints

### Core Endpoints

| Endpoint | Method | Purpose | Use Case |
|----------|--------|---------|----------|
| `GET /` | GET | Server information | Basic server metadata and available endpoints |
| `GET /health` | GET | Basic health check | Load balancer health monitoring |
| `GET /health/ready` | GET | Readiness probe | Kubernetes readiness check |
| `GET /health/live` | GET | Liveness probe | Kubernetes liveness check |
| `GET /model/status` | GET | Model status | Check if T5-small model is loaded and ready |
| `POST /generate` | POST | Text generation | Main inference endpoint for all NLP tasks |

### Example Requests

**Server Information (containerized):**
```bash
curl http://localhost/
```

**Health Check (containerized):**
```bash
curl http://localhost/health
```

**Model Status:**
```bash
curl http://localhost/model/status
```

**Text Generation (Translation):**
```bash
curl -X POST http://localhost/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "translate English to German: Hello world",
    "max_length": 50,
    "temperature": 1.0,
    "num_beams": 4
  }'
```

**Text Generation (Summarization):**
```bash
curl -X POST http://localhost/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "summarize: Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
    "max_length": 25
  }'
```

**Text Generation (Question Answering):**
```bash
curl -X POST http://localhost/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "question: What is the capital of France? context: Paris is the capital and largest city of France.",
    "max_length": 20
  }'
```

## 🎯 Model Choice: T5-small

**Why T5-small was selected for this demonstration:**

1. **Computational Efficiency**: 60M parameters provide optimal inference speed for containerized deployment without GPU requirements
2. **Architectural Versatility**: T5's text-to-text transformer architecture handles multiple NLP tasks through a unified interface:
   - **Translation**: English ↔ German, French, Spanish
   - **Summarization**: Text compression and key point extraction  
   - **Question Answering**: Information extraction from context
   - **Text Completion**: Generative text tasks
3. **Production Maturity**: Google's T5 is extensively tested, well-documented, and maintained in HuggingFace transformers
4. **Resource Optimization**: Memory footprint under 250MB fits comfortably in container constraints while maintaining quality output

This unified approach means one model serves multiple use cases, demonstrating real-world deployment scenarios where diverse NLP capabilities are required.

## 📊 Parallel Inference Demo

The `demo_notebook.ipynb` demonstrates the system's ability to handle concurrent requests efficiently:

- **8 Simultaneous Requests**: Different NLP tasks executed in parallel
- **Performance Metrics**: Comparison of parallel vs sequential execution times
- **Load Balancing Verification**: nginx distributing requests across multiple workers
- **Real-time Results**: Live demonstration of concurrent inference processing

### Running the Demo

1. **Start the containerized system:**
   ```bash
   docker compose up
   ```

2. **Open the notebook:**
   ```bash
   jupyter notebook demo_notebook.ipynb
   ```

3. **Execute all cells** to see parallel request processing in action

## 🧪 Testing

The project includes a comprehensive test suite using pytest with async support. Tests are organized by functionality and include unit tests, integration tests, and error handling scenarios.

### Test Categories

#### 🔧 **Unit Tests** (`test_basic_server.py`)
- **Server Basics**: Application startup, configuration validation
- **Health Endpoints**: Individual endpoint functionality and response validation
- **Response Models**: Pydantic model validation and data structure verification
- **Error Handling**: Exception handling and error response formatting
- **Async Functionality**: Concurrent request handling and performance

#### 🤖 **Model Integration Tests** (`test_model_integration.py`)
- **Model Service**: T5ModelService functionality and lifecycle
- **Text Generation**: Inference endpoint testing with mocked responses
- **Model Status**: Health check integration with model state
- **Error Scenarios**: Model loading failures and timeout handling
- **Concurrent Model Access**: Multiple simultaneous inference requests

### Running Tests

**Activate virtual environment first:**
```bash
source venv/bin/activate
```

**Run All Tests:**
```bash
pytest -v
```

**Run Specific Test Files:**
```bash
# Basic server functionality
pytest tests/test_basic_server.py -v

# Model integration tests
pytest tests/test_model_integration.py -v
```

**Run Tests with Coverage:**
```bash
pytest --cov=main --cov=model_service --cov-report=html
```

### Test Execution Example

**Full Test Suite:**
```bash
$ pytest -v

tests/test_basic_server.py::TestServerBasics::test_server_startup PASSED
tests/test_basic_server.py::TestHealthEndpoints::test_basic_health_check PASSED
tests/test_basic_server.py::TestAsyncFunctionality::test_concurrent_health_checks PASSED
tests/test_model_integration.py::TestModelService::test_model_loading_success PASSED
tests/test_model_integration.py::TestModelEndpoints::test_generate_text_endpoint_success PASSED
...
```

## 🔧 Troubleshooting

### Docker Issues

**Build Failures:**
```bash
# Clean up and rebuild
docker compose down
docker system prune -f
docker compose build --no-cache
```

**Container Won't Start:**
```bash
# Check container logs
docker compose logs app
docker compose logs nginx

# Verify container status
docker compose ps
```

**Permission Denied (Docker on WSL):**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Then restart WSL or logout/login

# Alternative: Use sudo
sudo docker compose build
sudo docker compose up
```

### Local Development Issues

**Server Won't Start:**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Check dependencies: `pip install -r requirements.txt`
- Verify port availability: `netstat -tulpn | grep 8000`

**Model Loading Errors:**
- Ensure sufficient RAM (4GB+ recommended)
- Check internet connectivity for first-time model download
- Verify transformers version: `pip show transformers`

**Jupyter Kernel Issues:**
```bash
# Reinstall kernel
source venv/bin/activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=container-task-venv --display-name="Container Task (venv)"
```

**Tests Failing:**
```bash
# Run specific failing test
pytest tests/test_basic_server.py::test_name -v

# Check dependencies
pip install -r requirements.txt

# Verify project structure
ls -la  # Should show main.py, model_service.py, etc.
```

### Network Issues

**Can't Access nginx Proxy:**
- Verify containers are running: `docker compose ps`
- Check nginx logs: `docker compose logs nginx`
- Test direct FastAPI: `curl http://localhost:8000/health` (if running locally)

**Demo Notebook Connection Errors:**
- Ensure containers are running on port 80
- Update notebook SERVER_URL if needed
- Test endpoints manually with curl first

### Performance Issues

**Slow Inference:**
- T5-small is CPU-optimized; consider GPU version for production
- Increase Docker memory allocation if using Docker Desktop
- Monitor container resources: `docker stats`

**Memory Errors:**
- Increase Docker memory limit
- Reduce number of uvicorn workers in docker-compose.yml
- Check system RAM availability

## 🚀 Production Deployment

### Kubernetes Deployment

The system is designed for Kubernetes with:
- Health check endpoints (`/health/ready`, `/health/live`)
- Non-root container user
- Configurable environment variables
- Horizontal pod autoscaling support

### Scaling Considerations

- **nginx**: Configure upstream servers for multiple app instances
- **Workers**: Adjust uvicorn worker count based on CPU cores
- **Memory**: Plan for ~1GB per worker including model cache
- **Storage**: Consider persistent volumes for model cache in production

### Monitoring

- **Metrics**: nginx access logs with timing information
- **Health**: Kubernetes probes for automatic restart/scaling
- **Logs**: Structured JSON logging for observability platforms