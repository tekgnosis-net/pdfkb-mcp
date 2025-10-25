#!/usr/bin/env bash
set -euo pipefail

# Local helper: build with buildx, use local cache dir, and stream output.
# Usage: ./scripts/local-build-and-monitor.sh [parser]
# Example: ./scripts/local-build-and-monitor.sh docling

PARSER=${1:-docling}
CACHE_DIR=".buildx-cache"
BUILDER_NAME="pdfkb-builder"
IMAGE_TAG="pdfkb-mcp:local-${PARSER}"

echo "Building image tag=${IMAGE_TAG} parser=${PARSER} using builder=${BUILDER_NAME}"

export DOCKER_BUILDKIT=1

# create builder if not exists
if ! docker buildx inspect ${BUILDER_NAME} >/dev/null 2>&1; then
  docker buildx create --use --name ${BUILDER_NAME} >/dev/null
else
  docker buildx use ${BUILDER_NAME}
fi

mkdir -p ${CACHE_DIR}

docker buildx build \
  --builder ${BUILDER_NAME} \
  --progress=plain \
  --load \
  --tag ${IMAGE_TAG} \
  --build-arg PDF_PARSER=${PARSER} \
  --cache-from=type=local,src=${CACHE_DIR} \
  --cache-to=type=local,dest=${CACHE_DIR},mode=max \
  .

echo "Build complete: ${IMAGE_TAG}"

echo "Listing built parsers in image (if any):"
docker run --rm --entrypoint sh ${IMAGE_TAG} -c 'ls -al /opt/parsers || echo "no /opt/parsers"'
