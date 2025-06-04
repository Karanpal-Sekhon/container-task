#!/bin/bash
# Production monitoring script for ML Inference Server
# Shows real-time status and performance metrics

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}$1${NC}"
    echo "$(printf '=%.0s' {1..60})"
}

print_metric() {
    printf "%-25s: %s\n" "$1" "$2"
}

# Function to get container status
get_container_status() {
    local container=$1
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container"; then
        echo -e "${GREEN}Running${NC}"
    else
        echo -e "${RED}Stopped${NC}"
    fi
}

# Function to check health endpoint
check_endpoint_health() {
    local url=$1
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [ "$response_code" -eq 200 ]; then
        echo -e "${GREEN}Healthy (200)${NC}"
    elif [ "$response_code" -eq 503 ]; then
        echo -e "${YELLOW}Service Unavailable (503)${NC}"
    elif [ "$response_code" -eq 000 ]; then
        echo -e "${RED}Unreachable${NC}"
    else
        echo -e "${YELLOW}HTTP $response_code${NC}"
    fi
}

# Function to get response time
get_response_time() {
    local url=$1
    local time=$(curl -s -o /dev/null -w "%{time_total}" "$url" 2>/dev/null || echo "0.000")
    echo "${time}s"
}

# Function to test text generation
test_generation() {
    local start_time=$(date +%s.%3N)
    local response=$(curl -s -X POST "http://localhost/generate" \
        -H "Content-Type: application/json" \
        -d '{"text":"translate English to German: Hello", "max_length":50}' 2>/dev/null || echo "error")
    local end_time=$(date +%s.%3N)
    local duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
    
    if [[ "$response" =~ "generated_text" ]]; then
        echo -e "${GREEN}Success (${duration}s)${NC}"
    else
        echo -e "${RED}Failed${NC}"
    fi
}

# Main monitoring loop
while true; do
    clear
    
    print_header "🔍 ML Inference Server - Real-time Monitoring"
    echo "Last updated: $(date)"
    echo ""
    
    # Container Status
    print_header "📦 Container Status"
    print_metric "FastAPI App" "$(get_container_status "ml-inference-app")"
    print_metric "Nginx Proxy" "$(get_container_status "ml-inference-nginx")"
    echo ""
    
    # Health Checks
    print_header "🏥 Health Endpoints"
    print_metric "App Health" "$(check_endpoint_health "http://localhost:8000/health")"
    print_metric "App Readiness" "$(check_endpoint_health "http://localhost:8000/health/ready")"
    print_metric "Nginx Health" "$(check_endpoint_health "http://localhost/health")"
    print_metric "Model Status" "$(check_endpoint_health "http://localhost/model/status")"
    echo ""
    
    # Response Times
    print_header "⏱️  Response Times"
    print_metric "Health Check" "$(get_response_time "http://localhost/health")"
    print_metric "Model Status" "$(get_response_time "http://localhost/model/status")"
    echo ""
    
    # Generation Test
    print_header "🤖 ML Generation Test"
    print_metric "Text Generation" "$(test_generation)"
    echo ""
    
    # Docker Stats
    print_header "📊 Resource Usage"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null | grep -E "(CONTAINER|ml-inference)" || echo "Containers not running"
    echo ""
    
    # Recent Logs
    print_header "📝 Recent Activity (Last 5 lines)"
    echo "--- FastAPI Logs ---"
    docker-compose logs --tail=3 app 2>/dev/null | tail -3 || echo "No logs available"
    echo ""
    echo "--- Nginx Logs ---"
    docker-compose logs --tail=2 nginx 2>/dev/null | tail -2 || echo "No logs available"
    echo ""
    
    # Footer
    echo -e "${BLUE}Commands:${NC}"
    echo "  • Ctrl+C to exit monitoring"
    echo "  • View full logs: docker-compose logs -f"
    echo "  • Test API: curl http://localhost/docs"
    
    # Wait before next update
    sleep 5
done