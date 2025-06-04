#!/bin/bash
# Production startup script for ML Inference Server
# This script demonstrates production deployment practices

set -e  # Exit on any error

echo "🚀 Starting ML Inference Server in Production Mode"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
print_status "Checking Docker availability..."
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi
print_success "Docker is running"

# Check if docker-compose is available
print_status "Checking docker-compose availability..."
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed or not in PATH"
    exit 1
fi
print_success "docker-compose is available"

# Create logs directory if it doesn't exist
print_status "Creating logs directory..."
mkdir -p logs/nginx
print_success "Logs directory ready"

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true
print_success "Existing containers stopped"

# Build and start services
print_status "Building Docker images..."
docker-compose build --no-cache

print_status "Starting services..."
docker-compose up -d

# Wait for services to be healthy
print_status "Waiting for services to become healthy..."
echo "This may take 60+ seconds for model loading..."

# Function to check health
check_health() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "$url" > /dev/null 2>&1; then
            return 0
        fi
        printf "."
        sleep 2
        ((attempt++))
    done
    return 1
}

# Check app health
print_status "Checking FastAPI app health..."
if check_health "app" "http://localhost:8000/health/ready"; then
    print_success "FastAPI app is healthy"
else
    print_error "FastAPI app failed to become healthy"
    docker-compose logs app
    exit 1
fi

# Check nginx health  
print_status "Checking nginx health..."
if check_health "nginx" "http://localhost:80/health"; then
    print_success "Nginx is healthy"
else
    print_error "Nginx failed to become healthy"
    docker-compose logs nginx
    exit 1
fi

# Display service status
print_success "🎉 ML Inference Server is running successfully!"
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🌐 Access Points:"
echo "  • API Documentation: http://localhost/docs"
echo "  • Health Check:      http://localhost/health"
echo "  • Text Generation:   http://localhost/generate"
echo "  • Alternative Docs:  http://localhost/redoc"

echo ""
echo "📋 Quick Test Commands:"
echo "  • curl http://localhost/health"
echo "  • curl -X POST http://localhost/generate -H 'Content-Type: application/json' -d '{\"text\":\"translate English to German: Hello world\"}'"

echo ""
echo "🔍 Monitoring Commands:"
echo "  • View logs:         docker-compose logs -f"
echo "  • Check status:      docker-compose ps"
echo "  • Stop services:     docker-compose down"

echo ""
print_warning "Note: First text generation request may be slower due to model initialization"