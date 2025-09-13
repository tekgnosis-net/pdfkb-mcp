# Multi-stage Dockerfile for pdfkb-mcp MCP Server
# Optimized for size, security, and performance
# Base image: python:3.11-slim for best compatibility with ML libraries

# Build arguments for customization
ARG PYTHON_VERSION=3.11
ARG PDFKB_VERSION=latest

# ============================================================================
# Stage 1: Builder - Install build dependencies and compile packages
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

# Build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Install build dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    # For some Python packages that need compilation
    python3-dev \
    # Git for VCS dependencies
    git \
    # SSL certificates for downloads
    ca-certificates \
    # For some native dependencies
    pkg-config \
    # Cleanup cache
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create build directory
WORKDIR /build

# Copy only requirements first for better caching
COPY pyproject.toml .
COPY src/ src/
COPY README.md .

# Install build dependencies and build the package
RUN pip install --upgrade pip setuptools wheel

# Install the package with all dependencies from pyproject.toml
# This eliminates the need to manually specify dependencies in the Dockerfile
RUN pip install --no-cache-dir -e .

# Build wheels for all dependencies to use in runtime stage
RUN pip freeze > requirements.txt && pip wheel --wheel-dir /build/wheels -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

# Build arguments
ARG PDFKB_VERSION
ARG BUILD_DATE
ARG VCS_REF

# Add labels for metadata
LABEL org.opencontainers.image.title="pdfkb-mcp" \
      org.opencontainers.image.description="PDF Knowledgebase MCP Server - Document search with vector embeddings" \
      org.opencontainers.image.version="${PDFKB_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/juanqui/pdfkb-mcp" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Juan Villa <juanqui@villafam.com>"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential runtime libraries
    libc6 \
    libgcc-s1 \
    libstdc++6 \
    # For HTTP health checks
    curl \
    # SSL/TLS certificates
    ca-certificates \
    # Clean up package cache
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*

# Create non-root user for security
RUN groupadd -r -g 1001 pdfkb && \
    useradd -r -g pdfkb -u 1001 -m -s /bin/false pdfkb

# Set up Python environment
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/home/pdfkb/.local/bin:$PATH"

# Application directories
ENV PDFKB_APP_DIR=/app \
    PDFKB_KNOWLEDGEBASE_PATH=/app/documents \
    PDFKB_CACHE_DIR=/app/cache \
    PDFKB_LOG_DIR=/app/logs \
    PDFKB_CONFIG_DIR=/app/config

# Create application directories with proper ownership
RUN mkdir -p ${PDFKB_APP_DIR} \
             ${PDFKB_KNOWLEDGEBASE_PATH} \
             ${PDFKB_CACHE_DIR} \
             ${PDFKB_LOG_DIR} \
             ${PDFKB_CONFIG_DIR} \
             /home/pdfkb/.local/bin && \
    chown -R pdfkb:pdfkb ${PDFKB_APP_DIR} /home/pdfkb

# Switch to non-root user
USER pdfkb

# Set working directory
WORKDIR ${PDFKB_APP_DIR}

# Copy wheels from builder stage
COPY --from=builder --chown=pdfkb:pdfkb /build/wheels /tmp/wheels

# Install Python packages from wheels (as non-root user)
RUN pip install --user --no-cache-dir --no-index --find-links /tmp/wheels \
    # Use the wheels we built in the previous stage
    pdfkb-mcp \
    && rm -rf /tmp/wheels

# Copy application source code
COPY --chown=pdfkb:pdfkb src/ src/
COPY --chown=pdfkb:pdfkb pyproject.toml .
COPY --chown=pdfkb:pdfkb README.md .

# Copy entrypoint script
COPY --chown=pdfkb:pdfkb docker-entrypoint.sh /app/docker-entrypoint.sh

# Make entrypoint executable
USER root
RUN chmod +x /app/docker-entrypoint.sh
USER pdfkb

# Default environment variables for container deployment
ENV PDFKB_EMBEDDING_PROVIDER=local \
    PDFKB_LOCAL_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B" \
    PDFKB_ENABLE_HYBRID_SEARCH=true \
    PDFKB_TRANSPORT=http \
    PDFKB_SERVER_HOST=0.0.0.0 \
    PDFKB_SERVER_PORT=8000 \
    PDFKB_WEB_ENABLE=false \
    PDFKB_LOG_LEVEL=INFO \
    PDFKB_MAX_PARALLEL_PARSING=1 \
    PDFKB_MAX_PARALLEL_EMBEDDING=1 \
    PDFKB_BACKGROUND_QUEUE_WORKERS=2 \
    PDFKB_PDF_PARSER=pymupdf4llm \
    PDFKB_DOCUMENT_CHUNKER=langchain \
    PDFKB_MODEL_CACHE_DIR=/app/cache/models

# Expose default ports
# 8000: Unified web + mcp port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PDFKB_SERVER_PORT}/health || exit 1

# Volume mount points for data persistence
VOLUME ["${PDFKB_KNOWLEDGEBASE_PATH}", "${PDFKB_CACHE_DIR}", "${PDFKB_LOG_DIR}"]

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command - can be overridden
CMD ["pdfkb-mcp", "--transport", "http"]
