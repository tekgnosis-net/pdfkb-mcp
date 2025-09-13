#!/bin/bash
set -euo pipefail

# Basic Podman test script for pdfkb-mcp
# Tests container image build and basic functionality

echo "🐳 Testing pdfkb-mcp Container Implementation (Podman)"
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Test configuration
IMAGE_NAME="pdfkb-mcp:test"
CONTAINER_NAME="pdfkb-mcp-test"
TEST_PORT=8001

# Cleanup function
cleanup() {
    log_info "Cleaning up test environment..."

    # Stop and remove container
    if podman ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        log_info "Stopping container: ${CONTAINER_NAME}"
        podman stop ${CONTAINER_NAME} >/dev/null 2>&1 || true
    fi

    if podman ps -a -q -f name=${CONTAINER_NAME} | grep -q .; then
        log_info "Removing container: ${CONTAINER_NAME}"
        podman rm ${CONTAINER_NAME} >/dev/null 2>&1 || true
    fi

    # Remove test directories
    if [[ -d ./test-documents ]]; then
        rm -rf ./test-documents
        log_info "Removed test documents directory"
    fi

    if [[ -d ./test-cache ]]; then
        rm -rf ./test-cache
        log_info "Removed test cache directory"
    fi

    # Optionally remove test image
    if [[ "${REMOVE_IMAGE:-false}" == "true" ]]; then
        if podman images -q ${IMAGE_NAME} | grep -q .; then
            log_info "Removing test image: ${IMAGE_NAME}"
            podman rmi ${IMAGE_NAME} >/dev/null 2>&1 || true
        fi
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Test 1: Dockerfile syntax and build
log_info "Test 1: Building container image..."
if podman build -t ${IMAGE_NAME} .; then
    log_info "✅ Container image built successfully"
else
    log_error "❌ Container image build failed"
    exit 1
fi

# Test 2: Image inspection
log_info "Test 2: Inspecting container image..."
IMAGE_SIZE=$(podman images ${IMAGE_NAME} --format "table {{.Size}}" | tail -n 1)
log_info "Image size: ${IMAGE_SIZE}"

# Check image labels
if podman inspect ${IMAGE_NAME} --format '{{.Config.Labels}}' | grep -q "org.opencontainers.image"; then
    log_info "✅ Image labels present"
else
    log_warn "⚠️  Image labels missing"
fi

# Test 3: Create test environment
log_info "Test 3: Setting up test environment..."
mkdir -p ./test-documents ./test-cache

# Create a simple test markdown file
cat > ./test-documents/test.md << 'EOF'
# Test Document

This is a test markdown document for Docker testing.

## Section 1
Content for testing document processing.

## Section 2
More content to test chunking and embedding.
EOF

log_info "✅ Test environment created"

# Test 4: Container startup
log_info "Test 4: Starting container..."
if podman run -d \
    --name ${CONTAINER_NAME} \
    -p ${TEST_PORT}:8000 \
    -v "$(pwd)/test-documents:/app/documents:rw" \
    -v "$(pwd)/test-cache:/app/cache" \
    -e PDFKB_EMBEDDING_PROVIDER=local \
    -e PDFKB_TRANSPORT=http \
    -e PDFKB_LOG_LEVEL=INFO \
    ${IMAGE_NAME}; then
    log_info "✅ Container started successfully"
else
    log_error "❌ Container startup failed"
    exit 1
fi

# Test 5: Wait for container to be ready
log_info "Test 5: Waiting for container to be ready..."
max_attempts=30
attempt=1

while [[ $attempt -le $max_attempts ]]; do
    if curl -s -f http://localhost:${TEST_PORT}/health >/dev/null 2>&1; then
        log_info "✅ Container is healthy and responding"
        break
    fi

    if [[ $attempt -eq $max_attempts ]]; then
        log_error "❌ Container health check failed after ${max_attempts} attempts"
        log_info "Container logs:"
        podman logs ${CONTAINER_NAME} --tail 20
        exit 1
    fi

    log_info "Waiting for container... (attempt ${attempt}/${max_attempts})"
    sleep 2
    ((attempt++))
done

# Test 6: Check health endpoint
log_info "Test 6: Testing health endpoint..."
if health_response=$(curl -s http://localhost:${TEST_PORT}/health); then
    if echo "${health_response}" | grep -q '"status":"healthy"'; then
        log_info "✅ Health endpoint working correctly"
    else
        log_warn "⚠️  Health endpoint returned unexpected response: ${health_response}"
    fi
else
    log_error "❌ Health endpoint request failed"
    exit 1
fi

# Test 7: Check container logs for errors
log_info "Test 7: Checking container logs..."
if podman logs ${CONTAINER_NAME} 2>&1 | grep -i error | grep -v "test"; then
    log_warn "⚠️  Found potential errors in container logs"
else
    log_info "✅ No critical errors found in logs"
fi

# Test 8: Verify volume mounts
log_info "Test 8: Verifying volume mounts..."
if podman exec ${CONTAINER_NAME} ls -la /app/documents/test.md >/dev/null 2>&1; then
    log_info "✅ Document volume mounted correctly"
else
    log_error "❌ Document volume mount failed"
    exit 1
fi

if podman exec ${CONTAINER_NAME} test -w /app/cache; then
    log_info "✅ Cache volume is writable"
else
    log_error "❌ Cache volume is not writable"
    exit 1
fi

# Test 9: Check container user
log_info "Test 9: Verifying container security..."
container_user=$(podman exec ${CONTAINER_NAME} whoami)
if [[ "${container_user}" == "pdfkb" ]]; then
    log_info "✅ Container running as non-root user: ${container_user}"
else
    log_error "❌ Container running as: ${container_user} (should be pdfkb)"
    exit 1
fi

# Test 10: Configuration validation
log_info "Test 10: Testing configuration validation..."
# This should fail with invalid transport
if podman run --rm \
    -e PDFKB_TRANSPORT=invalid \
    -e PDFKB_EMBEDDING_PROVIDER=local \
    ${IMAGE_NAME} pdfkb-mcp --transport invalid 2>&1 | grep -q "Invalid transport mode"; then
    log_info "✅ Configuration validation working"
else
    log_warn "⚠️  Configuration validation may not be working properly"
fi

# Test Summary
log_info ""
log_info "🎉 Docker Test Summary"
log_info "===================="
log_info "✅ All critical tests passed!"
log_info "✅ Docker image builds correctly"
log_info "✅ Container starts and runs"
log_info "✅ Health checks work"
log_info "✅ Volume mounts function"
log_info "✅ Security configuration correct"
log_info ""
log_info "📝 Test completed successfully!"
log_info "Container: ${CONTAINER_NAME}"
log_info "Image: ${IMAGE_NAME}"
log_info "Test port: ${TEST_PORT}"

# Show basic usage information
log_info ""
log_info "🔧 To manually test the container:"
log_info "podman exec -it ${CONTAINER_NAME} bash"
log_info "curl http://localhost:${TEST_PORT}/health"
log_info ""
log_info "To clean up test image: REMOVE_IMAGE=true ./test-docker.sh"
