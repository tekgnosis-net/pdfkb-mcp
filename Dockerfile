# Multi-stage Dockerfile for pdfkb-mcp MCP Server
# Optimized for size, security, and performance
# Base image: python:3.11-slim with CPU-only PyTorch for optimal size/compatibility balance

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

# Install UV (Rust-based Python package installer) for faster builds
RUN pip install --no-cache-dir --upgrade uv

# Install CPU-only PyTorch first to avoid CUDA dependencies
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# Install the package with remaining dependencies from pyproject.toml
# PyTorch is now already installed with CPU-only support
# Install all optional dependencies (use with caution - very large)
# Needed as optionals are not working in the docker image

RUN uv pip install --system --no-cache -e . \
    && uv pip install --system --no-cache -e ".[all-with-marker]" \
    && pip uninstall -y pip setuptools wheel uv  # Remove build tools to save space

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

# Build arguments
ARG PYTHON_VERSION
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

# Install only runtime system dependencies with aggressive cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential runtime libraries
    libc6 \
    libgcc-s1 \
    libstdc++6 \
    # For HTTP health checks
    curl \
    # SSL/TLS certificates
    ca-certificates \
    # Aggressive cleanup to minimize image size
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/cache/apt/archives/partial/* \
    && rm -rf /var/log/apt/* \
    && rm -rf /var/log/dpkg.log \
    && rm -rf /root/.cache \
    && find /usr/local -name "*.pyc" -delete \
    && find /usr/local -name "__pycache__" -type d -exec rm -rf {} + || true

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

# Copy Python packages and installed libraries from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY --chown=pdfkb:pdfkb src/ src/
COPY --chown=pdfkb:pdfkb pyproject.toml .
COPY --chown=pdfkb:pdfkb README.md .

# Final cleanup to minimize image size
USER root
RUN rm -rf /root/.cache \
    && rm -rf /tmp/* \
    && find /usr/local -name "*.pyc" -delete \
    && find /usr/local -name "__pycache__" -type d -exec rm -rf {} + || true

# Copy entrypoint script
COPY --chown=pdfkb:pdfkb docker-entrypoint.sh /app/docker-entrypoint.sh

# Make entrypoint executable
USER root
RUN chmod +x /app/docker-entrypoint.sh
USER pdfkb

# Default environment variables for container deployment
# Models are downloaded dynamically on first use (not pre-installed)
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
