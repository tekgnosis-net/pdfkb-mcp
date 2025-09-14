#!/bin/bash
# Build script for pdfkb-mcp Docker image
# Uses podman to build optimized CPU-only PyTorch image

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="pdfkb-mcp"
IMAGE_TAG="latest"
DOCKERFILE="Dockerfile"

# Build arguments
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo -e "${BLUE}🐳 Building pdfkb-mcp Docker image...${NC}"
echo -e "${YELLOW}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "${YELLOW}Build Date: ${BUILD_DATE}${NC}"
echo -e "${YELLOW}VCS Ref: ${VCS_REF}${NC}"
echo

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    echo -e "${RED}❌ Error: $DOCKERFILE not found in current directory${NC}"
    exit 1
fi

# Build the image
echo -e "${BLUE}📦 Starting build...${NC}"
podman build \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --file "$DOCKERFILE" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --build-arg "VCS_REF=${VCS_REF}" \
    --build-arg "PDFKB_VERSION=${IMAGE_TAG}" \
    .

# Check if build was successful
if [[ $? -eq 0 ]]; then
    echo
    echo -e "${GREEN}✅ Build completed successfully!${NC}"

    # Show image info
    echo -e "${BLUE}📊 Image information:${NC}"
    podman images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.Created}}"

    echo
    echo -e "${GREEN}🚀 Image ready for deployment!${NC}"
    echo -e "${YELLOW}Run with: podman run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
