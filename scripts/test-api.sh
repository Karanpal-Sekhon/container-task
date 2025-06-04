#!/bin/bash
# API testing script for ML Inference Server
# Tests all endpoints to verify functionality

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_URL="http://localhost"
PASSED=0
FAILED=0

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local expected_code=$3
    local data=$4
    local description=$5
    
    print_test "$description"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$BASE_URL$endpoint" \
                  -H "Content-Type: application/json" -d "$data" 2>/dev/null)
    fi
    
    # Extract HTTP code (last line)
    http_code=$(echo "$response" | tail -n1)
    # Extract body (all but last line)
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" -eq "$expected_code" ]; then
        print_pass "HTTP $http_code - $description"
        if [ ! -z "$body" ] && [ "$body" != "null" ]; then
            echo "  Response: $(echo "$body" | jq -r '.' 2>/dev/null | head -3 || echo "$body" | head -50)"
        fi
    else
        print_fail "Expected HTTP $expected_code, got $http_code - $description"
        echo "  Response: $body"
    fi
    echo ""
}

echo "🧪 Testing ML Inference Server API"
echo "=================================="
echo ""

# Health checks
test_endpoint "GET" "/health" 200 "" "Basic health check"
test_endpoint "GET" "/health/live" 200 "" "Liveness probe"
test_endpoint "GET" "/health/ready" 200 "" "Readiness probe (may be 503 if model loading)"

# Server info
test_endpoint "GET" "/" 200 "" "Server information"
test_endpoint "GET" "/model/status" 200 "" "Model status"

# Documentation
test_endpoint "GET" "/docs" 200 "" "Swagger UI documentation"
test_endpoint "GET" "/redoc" 200 "" "ReDoc documentation"
test_endpoint "GET" "/openapi.json" 200 "" "OpenAPI schema"

# Text generation tests
echo -e "${YELLOW}Testing Text Generation (may take 10-30 seconds)...${NC}"

# Valid generation request
test_endpoint "POST" "/generate" 200 '{"text":"translate English to German: Hello world"}' "Basic text generation"

# Generation with parameters
test_endpoint "POST" "/generate" 200 '{"text":"summarize: This is a long text that needs to be summarized", "max_length":50, "temperature":0.8}' "Generation with custom parameters"

# Invalid requests (should return 422)
test_endpoint "POST" "/generate" 422 '{"text":""}' "Empty text (should fail validation)"
test_endpoint "POST" "/generate" 422 '{"text":"test", "max_length":1000}' "Invalid max_length (should fail validation)"
test_endpoint "POST" "/generate" 422 '{}' "Missing required field (should fail validation)"

# Non-existent endpoint
test_endpoint "GET" "/nonexistent" 404 "" "Non-existent endpoint (should return 404)"

# Summary
echo "🏁 Test Results Summary"
echo "======================"
echo -e "Tests Passed: ${GREEN}$PASSED${NC}"
echo -e "Tests Failed: ${RED}$FAILED${NC}"
echo -e "Total Tests:  $((PASSED + FAILED))"

if [ $FAILED -eq 0 ]; then
    echo -e "\n🎉 ${GREEN}All tests passed! API is working correctly.${NC}"
    exit 0
else
    echo -e "\n❌ ${RED}Some tests failed. Check the output above.${NC}"
    exit 1
fi