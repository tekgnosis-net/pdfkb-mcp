# Multi-stage Dockerfile for pdfkb-mcp MCP Server
# Optimized for size, security, and performance
# Base image: python:3.11-slim with CPU-only PyTorch for optimal size/compatibility balance

# Build arguments for customization
ARG PYTHON_VERSION=3.11
ARG PDFKB_VERSION=latest
ARG PDF_PARSER=mineru  # Options: marker, mineru, pymupdf4llm
ARG USE_CUDA=false          # Build with CUDA-enabled PyTorch when "true"
ARG CUDA_PYTORCH_TAG=cu118  # PyTorch CUDA wheel tag (e.g. cu118, cu121)
ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime
ARG RUNTIME_BASE_IMAGE=pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

# ============================================================================
# Stage 1: Builder - Install build dependencies and compile packages
# ============================================================================
FROM ${BASE_IMAGE} AS builder

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
    # For HTTP requests (needed for font download)
    curl \
    # OpenCV system dependencies (needed for marker-pdf)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
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

# Install PyTorch: use CUDA wheels when requested, otherwise install CPU-only
RUN if [ "$USE_CUDA" = "true" ]; then \
        echo "Installing CUDA PyTorch wheels (tag: $CUDA_PYTORCH_TAG)"; \
        pip uninstall -y torch torchvision torchaudio || true; \
        uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/$CUDA_PYTORCH_TAG \
            --upgrade --force-reinstall torch torchvision torchaudio; \
    else \
        echo "Installing CPU-only PyTorch wheels"; \
        pip uninstall -y torch torchvision torchaudio || true; \
        uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu \
            --upgrade --force-reinstall torch torchvision torchaudio; \
    fi

# Build arguments
ARG PYTHON_VERSION
ARG BUILDPLATFORM
ARG PDF_PARSER

# Install the package with remaining dependencies from pyproject.toml
# PyTorch is now already installed with CPU-only support

# Install optional dependencies depending on PDF_PARSER. Support a special
# value `all` which installs both marker and mineru extras.
RUN uv pip install --system --no-cache -e . \
    && if [ "$PDF_PARSER" = "mineru" ]; then \
        uv pip install --system --no-cache -e ".[all-with-mineru]"; \
    elif [ "$PDF_PARSER" = "marker" ]; then \
        uv pip install --system --no-cache -e ".[all-with-marker]"; \
    elif [ "$PDF_PARSER" = "all" ]; then \
        # Install both marker and mineru extras
        uv pip install --system --no-cache -e ".[all-with-marker]" \
            -e ".[all-with-mineru]"; \
        # Also ensure the actual third-party packages are present (some installers
        # may not pull them into site-packages of the wheel layout). Installing
        # them explicitly avoids missing imports at runtime.
        pip install --no-cache-dir marker-pdf>=1.10.0 || true; \
        pip install --no-cache-dir "mineru[pipeline]>=2.1.10" || true; \
    else \
        uv pip install --system --no-cache -e ".[all-with-marker]"; \
    fi \
    && pip uninstall -y opencv-python || true  # Remove GUI version to avoid conflicts with headless \
    && pip install --no-cache opencv-python-headless==4.11.0.86  # Ensure headless version is properly installed

# Build a wheel for the application so the runtime image can install it
# This avoids building from source in the runtime (which would require build deps).
RUN python -m pip wheel --no-deps -w /build/dist . \
    && pip uninstall -y pip setuptools wheel uv || true  # Remove build tools to save space

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM ${RUNTIME_BASE_IMAGE} AS runtime

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
    # OpenCV runtime dependencies (needed for marker-pdf)
    libglib2.0-0 \
    libsm6 \
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1 \
    libglvnd0 \
    libglx0 \
    libglx-mesa0 \
    libglapi-mesa \
    libxcb-glx0 \
    libxxf86vm1 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    libx11-6 \
    libxrandr2 \
    libxcursor1 \
    libxi6 \
    libgomp1 \
    libgthread-2.0-0 \
    # Aggressive cleanup to minimize image size
    && rm -rf /var/lib/apt/lists/* \
    # Ensure dynamic linker cache is updated so libGL is discoverable
    && ldconfig || true \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/cache/apt/archives/partial/* \
    && rm -rf /var/log/apt/* \
    && rm -rf /var/log/dpkg.log \
    && rm -rf /root/.cache \
    && find /usr/local -name "*.pyc" -delete \
    && find /usr/local -name "__pycache__" -type d -exec rm -rf {} + || true

# Make build args available in runtime stage for conditional installs
ARG USE_CUDA
ARG CUDA_PYTORCH_TAG
ARG PDF_PARSER

# Prevent interactive prompts during apt operations in the runtime image
ENV DEBIAN_FRONTEND=noninteractive

# If building with CUDA support, ensure runtime has CUDA PyTorch wheels installed
RUN if [ "$USE_CUDA" = "true" ]; then \
        echo "Installing CUDA PyTorch wheels in runtime (tag: $CUDA_PYTORCH_TAG)"; \
        pip uninstall -y torch torchvision torchaudio || true; \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/$CUDA_PYTORCH_TAG \
            --upgrade --force-reinstall torch torchvision torchaudio || true; \
    else \
        echo "Skipping CUDA runtime torch install (USE_CUDA=$USE_CUDA)"; \
    fi
# Ensure opencv headless is available at runtime (some wheel layouts miss cv2 when copied)
RUN pip install --no-cache-dir opencv-python-headless==4.11.0.86 || true
# Ensure redis client is available in runtime for optional Redis-backed scopes
RUN pip install --no-cache-dir redis>=4.6.0 || true

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

# Set working directory
WORKDIR ${PDFKB_APP_DIR}

# Copy built wheel from the builder stage and install it in the runtime image.
# This produces deterministic binary artifacts for the final image instead of
# copying whole site-packages (which can lead to mismatched binary wheels).
COPY --from=builder /build/dist /build/dist
RUN if ls /build/dist/*.whl >/dev/null 2>&1; then \
        pip install --no-cache-dir /build/dist/*.whl && rm -rf /build/dist; \
    else \
        echo "No wheel found in /build/dist, falling back to editable install"; \
        pip install --no-cache-dir -e .; \
    fi

# Install parser extras at runtime if requested (marker, mineru or both).
RUN if [ "$PDF_PARSER" = "mineru" ]; then \
        pip install --no-cache-dir "mineru[pipeline]>=2.1.10"; \
    elif [ "$PDF_PARSER" = "marker" ]; then \
        pip install --no-cache-dir marker-pdf>=1.10.0; \
    elif [ "$PDF_PARSER" = "all" ]; then \
        pip install --no-cache-dir marker-pdf>=1.10.0 || true; \
        pip install --no-cache-dir "mineru[pipeline]>=2.1.10" || true; \
    else \
        echo "No PDF parser extras requested (PDF_PARSER=${PDF_PARSER})"; \
    fi

# Download Marker font file at runtime so the final image sets correct
# permissions for the non-root user that will run the server. Use Python
# for downloading to avoid depending on external CLI tools being available
# in every base image.
RUN if [ "$PDF_PARSER" = "marker" ] || [ "$PDF_PARSER" = "all" ]; then \
            mkdir -p /usr/local/lib/python3.11/site-packages/static/fonts && \
            python -c "import urllib.request; urllib.request.urlretrieve('https://models.datalab.to/artifacts/GoNotoCurrent-Regular.ttf','/usr/local/lib/python3.11/site-packages/static/fonts/GoNotoCurrent-Regular.ttf')" && \
            chmod 644 /usr/local/lib/python3.11/site-packages/static/fonts/GoNotoCurrent-Regular.ttf; \
    else \
        echo "Skipping font download (PDF_PARSER=${PDF_PARSER})"; \
    fi

# Ensure system GL libraries are present for opencv at runtime. Some base images
# may not include these by default; install them explicitly here so cv2 can load
# libGL.so.1 during import.
# (GL and related libraries are already installed in the runtime apt-get above.)

# Set ownership of static directory if it exists (for marker parser)
# Make this tolerant: try to chown, but don't fail the build if some files
# are not changeable (e.g., created by root in earlier stage). We'll also
# ensure fonts are world-readable when downloaded in the builder stage so
# the non-root `pdfkb` user can read them at runtime even if chown fails.
RUN if [ -d "/usr/local/lib/python3.11/site-packages/static" ]; then \
        chown -R pdfkb:pdfkb /usr/local/lib/python3.11/site-packages/static || true; \
    fi

# Ensure `/opt/conda` site-packages `static` directory exists and is writable
# by the runtime user. Some installations (conda-based runtimes) put
# packages under `/opt/conda/...`, and Marker may try to create a
# `/opt/conda/lib/python3.11/site-packages/static` directory at runtime.
# Create it now and chown to the non-root `pdfkb` user so runtime writes
# don't fail with PermissionError.
RUN mkdir -p /opt/conda/lib/python3.11/site-packages/static && \
    chown -R pdfkb:pdfkb /opt/conda/lib/python3.11/site-packages/static || true

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
    PDFKB_PDF_PARSER=marker \
    PDFKB_DOCUMENT_CHUNKER=langchain \
    PDFKB_MODEL_CACHE_DIR=/app/cache/models

# Default to CUDA device for embeddings when image is built with CUDA support
ENV PDFKB_EMBEDDING_DEVICE=cuda

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
