# Multi-stage Dockerfile for HuggingFace Model Inference Server
# Stage 1: Build dependencies and download models
# Stage 2: Production runtime

# ==============================================================================
# Stage 1: Builder Stage
# ==============================================================================
FROM python:3.10-slim as builder

# Set build arguments
ARG MODEL_NAME=t5-small

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY . .

# Pre-download the model to cache it in the image
# This prevents download delays during container startup
RUN python -c "\
from transformers import T5ForConditionalGeneration, T5Tokenizer; \
import os; \
model_name = os.getenv('MODEL_NAME', 't5-small'); \
print(f'Pre-downloading {model_name} model...'); \
tokenizer = T5Tokenizer.from_pretrained(model_name); \
model = T5ForConditionalGeneration.from_pretrained(model_name); \
print(f'✅ {model_name} model cached successfully'); \
"

# ==============================================================================
# Stage 2: Production Runtime
# ==============================================================================
FROM python:3.10-slim as production

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code and pre-downloaded models
COPY --from=builder /app .
COPY --from=builder /root/.cache /home/appuser/.cache

# Change ownership to appuser
RUN chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Add local Python packages to PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_NAME=t5-small
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=INFO

# Expose the application port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# Default command - can be overridden in docker-compose
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]