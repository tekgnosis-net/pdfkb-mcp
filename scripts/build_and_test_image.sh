#!/usr/bin/env bash
# Build and smoke-test the Docker image with parser extras (PDF_PARSER=all)
# Usage: ./scripts/build_and_test_image.sh
set -euo pipefail

IMAGE_TAG="pdfkb-mcp:all"

# Build image with extras. The Dockerfile should respect PDF_PARSER env.
PDF_PARSER=all docker build -t "$IMAGE_TAG" .

# Run container in background with necessary services (Redis), or use docker-compose
# For quick smoke test we start Redis using docker if not already running and then run the image.

# Start a Redis container for scoped-session persistence
REDIS_NAME="pdfkb_test_redis"
if [ "$(docker ps -q -f name=$REDIS_NAME)" = "" ]; then
  echo "Starting ephemeral Redis container: $REDIS_NAME"
  docker run -d --rm --name "$REDIS_NAME" -p 6379:6379 redis:7-alpine
  REDIS_STARTED=true
else
  REDIS_STARTED=false
fi

# Run the pdfkb-mcp container and expose ports
CONTAINER_NAME="pdfkb_mcp_test"
docker run -d --rm --name "$CONTAINER_NAME" \
  -e PDF_PARSER=all \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  -p 8000:8000 \
  "$IMAGE_TAG"

# Wait for the service to come up (simple loop)
echo "Waiting for service on http://localhost:8000..."
for i in {1..30}; do
  if curl -sSf http://localhost:8000/health >/dev/null 2>&1; then
    echo "Service is up"
    break
  fi
  sleep 1
done

# Run marker parsing and context-shift smoke checks (adjust endpoints as needed)
# 1) Upload a simple PDF and request parsing (replace with real endpoint if different)
# 2) Exercise scoped search endpoints to validate context shift persistence

# Placeholder: user may need to adjust endpoint paths
# Example POST to /parse?parser=marker
# curl -F "file=@tests/sample.pdf" "http://localhost:8000/parse?parser=marker"

# Example scoped search (POST search with session id)
# curl -X POST -H "Content-Type: application/json" -d '{"query":"test","session_id":"demo"}' http://localhost:8000/api/search/scoped

# Teardown
if [ "$REDIS_STARTED" = true ]; then
  echo "Stopping ephemeral Redis"
  docker stop "$REDIS_NAME"
fi

echo "Smoke test complete. Container logs (recent):"
docker logs --tail 200 "$CONTAINER_NAME" || true

echo "Cleaning up test container"
docker stop "$CONTAINER_NAME" || true

exit 0
