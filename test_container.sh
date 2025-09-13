#!/bin/bash

# Container Test Script for pdfkb-mcp
# This script comprehensively tests the built container image
#
# Usage: ./test_container.sh [IMAGE_TAG]
# Default IMAGE_TAG: pdfkb-mcp:test

set -euo pipefail

# Configuration
IMAGE_TAG="${1:-pdfkb-mcp:test}"
CONTAINER_NAME="pdfkb-mcp-test-$$"
TEST_PORT="8899"
HEALTH_CHECK_TIMEOUT=60

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up test container..."
    podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    podman rm "${CONTAINER_NAME}" 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Test 1: Image exists and basic inspection
test_image_exists() {
    log_info "Test 1: Checking if image exists..."

    if ! podman image exists "${IMAGE_TAG}"; then
        log_error "Image '${IMAGE_TAG}' does not exist!"
        return 1
    fi

    log_success "Image '${IMAGE_TAG}' exists"

    # Show image details
    log_info "Image details:"
    podman inspect "${IMAGE_TAG}" --format "Size: {{.Size}} bytes" || true
    podman inspect "${IMAGE_TAG}" --format "Created: {{.Created}}" || true

    return 0
}

# Test 2: Container starts successfully
test_container_start() {
    log_info "Test 2: Starting container..."

    # Start container in detached mode with health check port
    podman run -d \
        --name "${CONTAINER_NAME}" \
        -p "${TEST_PORT}:8000" \
        -e "PDFKB_SERVER_PORT=8000" \
        -e "PDFKB_WEB_ENABLE=false" \
        -e "PDFKB_LOG_LEVEL=DEBUG" \
        -e "PDFKB_TRANSPORT=http" \
        "${IMAGE_TAG}" \
        pdfkb-mcp --transport http --server-port 8000 || return 1

    log_success "Container started successfully"
    return 0
}

# Test 3: Container is running and healthy
test_container_health() {
    log_info "Test 3: Checking container health..."

    local attempts=0
    local max_attempts=12
    local sleep_interval=5

    while [ ${attempts} -lt ${max_attempts} ]; do
        if podman ps --filter "name=${CONTAINER_NAME}" --filter "status=running" --quiet | grep -q .; then
            log_success "Container is running"

            # Check if port is accessible
            log_info "Waiting for service to be ready on port ${TEST_PORT}..."
            sleep ${sleep_interval}

            # Try to connect to the service
            if curl -f -s -m 5 "http://localhost:${TEST_PORT}/" > /dev/null 2>&1; then
                log_success "Service is responding on port ${TEST_PORT}"
                return 0
            elif curl -f -s -m 5 "http://localhost:${TEST_PORT}/health" > /dev/null 2>&1; then
                log_success "Health endpoint is responding"
                return 0
            else
                log_warning "Service not yet ready, retrying... (attempt $((attempts + 1))/${max_attempts})"
            fi
        else
            log_warning "Container not running, retrying... (attempt $((attempts + 1))/${max_attempts})"
        fi

        attempts=$((attempts + 1))
        sleep ${sleep_interval}
    done

    log_error "Container failed to become healthy within ${HEALTH_CHECK_TIMEOUT} seconds"

    # Show logs for debugging
    log_info "Container logs:"
    podman logs "${CONTAINER_NAME}" || true

    return 1
}

# Test 4: MCP server responds to basic requests
test_mcp_functionality() {
    log_info "Test 4: Testing MCP functionality..."

    # Test if MCP server is responding
    local mcp_url="http://localhost:${TEST_PORT}"

    # Try different MCP endpoints that should be available
    local endpoints=(
        "/"
        "/mcp/"
        "/mcp/tools"
    )

    local success=false
    for endpoint in "${endpoints[@]}"; do
        log_info "Testing endpoint: ${endpoint}"
        if curl -f -s -m 10 "${mcp_url}${endpoint}" > /dev/null 2>&1; then
            log_success "Endpoint ${endpoint} is accessible"
            success=true
            break
        else
            log_warning "Endpoint ${endpoint} not accessible"
        fi
    done

    if [ "$success" = true ]; then
        log_success "MCP server is responding"
        return 0
    else
        log_error "No MCP endpoints are accessible"
        return 1
    fi
}

# Test 5: Check container resource usage
test_container_resources() {
    log_info "Test 5: Checking container resource usage..."

    # Get container stats
    local stats
    stats=$(podman stats "${CONTAINER_NAME}" --no-stream --format "table {{.MemUsage}}\t{{.CPUPerc}}" 2>/dev/null || echo "Unable to get stats")

    if [ "$stats" != "Unable to get stats" ]; then
        log_info "Container resource usage:"
        echo "$stats"
        log_success "Resource usage check completed"
    else
        log_warning "Could not retrieve container resource statistics"
    fi

    return 0
}

# Test 6: Validate environment variables and configuration
test_container_config() {
    log_info "Test 6: Validating container configuration..."

    # Check key environment variables
    local env_check
    env_check=$(podman exec "${CONTAINER_NAME}" env | grep "PDFKB_" | head -10 || echo "No PDFKB env vars")

    if [ "$env_check" != "No PDFKB env vars" ]; then
        log_success "Container has PDFKB environment variables configured"
        log_info "Sample environment variables:"
        echo "$env_check" | head -5
    else
        log_warning "No PDFKB environment variables found"
    fi

    # Check if pdfkb-mcp command is available
    if podman exec "${CONTAINER_NAME}" which pdfkb-mcp >/dev/null 2>&1; then
        log_success "pdfkb-mcp command is available in container"
    else
        log_error "pdfkb-mcp command not found in container"
        return 1
    fi

    return 0
}

# Test 7: Check Python packages installation
test_python_packages() {
    log_info "Test 7: Checking Python packages installation..."

    # Check if key packages are installed
    local packages=("fastmcp" "chromadb" "torch" "transformers" "pdfkb-mcp")
    local missing_packages=()

    for package in "${packages[@]}"; do
        if podman exec "${CONTAINER_NAME}" python -c "import ${package//-/_}" >/dev/null 2>&1; then
            log_success "Package '${package}' is installed and importable"
        else
            log_error "Package '${package}' is missing or not importable"
            missing_packages+=("${package}")
        fi
    done

    if [ ${#missing_packages[@]} -eq 0 ]; then
        log_success "All required Python packages are installed"
        return 0
    else
        log_error "Missing packages: ${missing_packages[*]}"
        return 1
    fi
}

# Main test execution
main() {
    log_info "Starting comprehensive container test for '${IMAGE_TAG}'"
    log_info "Container name: ${CONTAINER_NAME}"
    log_info "Test port: ${TEST_PORT}"
    echo "=================================================="

    local test_results=()
    local tests=(
        "test_image_exists"
        "test_container_start"
        "test_container_health"
        "test_mcp_functionality"
        "test_container_resources"
        "test_container_config"
        "test_python_packages"
    )

    for test in "${tests[@]}"; do
        echo
        if $test; then
            test_results+=("✅ $test: PASSED")
        else
            test_results+=("❌ $test: FAILED")
        fi
    done

    # Summary
    echo
    echo "=================================================="
    log_info "Test Summary:"

    local passed=0
    local failed=0

    for result in "${test_results[@]}"; do
        echo "$result"
        if [[ $result == *"PASSED"* ]]; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done

    echo
    if [ $failed -eq 0 ]; then
        log_success "All tests passed! ($passed/$((passed + failed)))"
        log_success "Container '${IMAGE_TAG}' is working correctly"
        return 0
    else
        log_error "$failed tests failed out of $((passed + failed)) total tests"
        log_error "Container '${IMAGE_TAG}' has issues that need to be addressed"
        return 1
    fi
}

# Run main function
main "$@"
