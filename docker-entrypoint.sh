#!/bin/bash
set -euo pipefail

# Docker entrypoint script for pdfkb-mcp MCP Server
# Handles environment setup, configuration validation, and service startup

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [[ "${PDFKB_LOG_LEVEL:-INFO}" == "DEBUG" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*" >&2
    fi
}

# Cleanup function for graceful shutdown
cleanup() {
    local exit_code=$?
    log_info "Received shutdown signal, cleaning up..."

    # Kill any background processes
    if [[ -n "${PDFKB_PID:-}" ]]; then
        log_info "Stopping pdfkb-mcp server (PID: ${PDFKB_PID})..."
        kill -TERM "${PDFKB_PID}" 2>/dev/null || true
        wait "${PDFKB_PID}" 2>/dev/null || true
    fi

    log_info "Cleanup completed"
    exit $exit_code
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT SIGHUP

# Environment variable validation
validate_environment() {
    log_info "Validating environment configuration..."

    # Check transport mode
    case "${PDFKB_TRANSPORT:-http}" in
        stdio|http|sse)
            log_debug "Transport mode: ${PDFKB_TRANSPORT:-http}"
            ;;
        *)
            log_error "Invalid transport mode: ${PDFKB_TRANSPORT}. Must be stdio, http, or sse"
            exit 1
            ;;
    esac

    # Validate embedding provider
    case "${PDFKB_EMBEDDING_PROVIDER:-local}" in
        local|openai|huggingface)
            log_debug "Embedding provider: ${PDFKB_EMBEDDING_PROVIDER:-local}"
            ;;
        *)
            log_error "Invalid embedding provider: ${PDFKB_EMBEDDING_PROVIDER}. Must be local, openai, or huggingface"
            exit 1
            ;;
    esac

    # Check OpenAI API key if using OpenAI embeddings
    if [[ "${PDFKB_EMBEDDING_PROVIDER:-local}" == "openai" ]]; then
        if [[ -z "${PDFKB_OPENAI_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
            log_error "PDFKB_OPENAI_API_KEY is required when using OpenAI embeddings"
            exit 1
        fi
    fi

    # Check HuggingFace token if using HuggingFace embeddings
    if [[ "${PDFKB_EMBEDDING_PROVIDER:-local}" == "huggingface" ]]; then
        if [[ -z "${PDFKB_HUGGINGFACE_API_KEY:-}" && -z "${HF_TOKEN:-}" ]]; then
            log_error "PDFKB_HUGGINGFACE_API_KEY or HF_TOKEN is required when using HuggingFace embeddings"
            exit 1
        fi
    fi

    # Validate unified server port
    if [[ "${PDFKB_WEB_PORT:-8000}" -lt 1 || "${PDFKB_WEB_PORT:-8000}" -gt 65535 ]]; then
        log_error "Invalid unified server port: ${PDFKB_WEB_PORT}. Must be between 1 and 65535"
        exit 1
    fi

    log_info "Environment validation completed successfully"
}

# Create and set permissions for required directories
setup_directories() {
    log_info "Setting up application directories..."

    # Array of directories to create with descriptions
    declare -A dirs=(
        ["${PDFKB_KNOWLEDGEBASE_PATH:-/app/documents}"]="Documents directory"
        ["${PDFKB_CACHE_DIR:-/app/cache}"]="Cache directory"
        ["${PDFKB_CACHE_DIR:-/app/cache}/chroma"]="ChromaDB directory"
        ["${PDFKB_CACHE_DIR:-/app/cache}/metadata"]="Metadata directory"
        ["${PDFKB_CACHE_DIR:-/app/cache}/processing"]="Processing directory"
        ["${PDFKB_CACHE_DIR:-/app/cache}/models"]="Models cache directory"
        ["${PDFKB_LOG_DIR:-/app/logs}"]="Logs directory"
    )

    # Create directories if they don't exist
    for dir in "${!dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_debug "Creating ${dirs[$dir]}: $dir"
            mkdir -p "$dir"
        else
            log_debug "${dirs[$dir]} already exists: $dir"
        fi

        # Ensure proper permissions
        if [[ ! -w "$dir" ]]; then
            log_warn "Directory not writable: $dir"
        fi
    done

    # Create hybrid search index directory if enabled
    if [[ "${PDFKB_ENABLE_HYBRID_SEARCH:-true}" == "true" ]]; then
        whoosh_dir="${PDFKB_CACHE_DIR:-/app/cache}/whoosh"
        if [[ ! -d "$whoosh_dir" ]]; then
            log_debug "Creating Whoosh index directory: $whoosh_dir"
            mkdir -p "$whoosh_dir"
        fi
    fi

    log_info "Directory setup completed"
}

# Display configuration summary
show_configuration() {
    log_info "Starting pdfkb-mcp with configuration:"
    echo "  📁 Documents: ${PDFKB_KNOWLEDGEBASE_PATH:-/app/documents}"
    echo "  💾 Cache: ${PDFKB_CACHE_DIR:-/app/cache}"
    echo "  📊 Logs: ${PDFKB_LOG_DIR:-/app/logs}"
    echo "  🔗 Transport: ${PDFKB_TRANSPORT:-http}"
    echo "  🧠 Embeddings: ${PDFKB_EMBEDDING_PROVIDER:-local}"
    echo "  📚 Model: ${PDFKB_LOCAL_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
    echo "  🔍 Hybrid Search: ${PDFKB_ENABLE_HYBRID_SEARCH:-true}"
    echo "  🌐 Web Interface: ${PDFKB_WEB_ENABLE:-false}"
    if [[ "${PDFKB_WEB_ENABLE:-false}" == "true" ]]; then
        echo "  🌍 Unified Server: ${PDFKB_WEB_HOST:-0.0.0.0}:${PDFKB_WEB_PORT:-8000} (Web + MCP endpoints)"
    fi
    echo "  📄 Parser: ${PDFKB_PDF_PARSER:-pymupdf4llm}"
    echo "  ✂️  Chunker: ${PDFKB_DOCUMENT_CHUNKER:-langchain}"
    echo "  📝 Log Level: ${PDFKB_LOG_LEVEL:-INFO}"
}

# Health check endpoint setup (for unified server)
setup_health_check() {
    if [[ "${PDFKB_WEB_ENABLE:-false}" == "true" ]]; then
        log_info "Health check will be available at: http://localhost:${PDFKB_WEB_PORT:-8000}/health"
    fi
}


# Ensure parser extras (marker/mineru) are installed when requested via
# the runtime environment variable `PDFKB_PDF_PARSER`.
# This is a best-effort installer: it will try to import the required
# package and, if missing, attempt a pip install. Failures are logged
# but do not stop container startup.
ensure_parser_extras_installed() {
    parser="${PDFKB_PDF_PARSER:-pymupdf4llm}"
    log_info "Ensuring parser extras for: ${parser} (this may install packages if missing)"

    # Check if 'timeout' is available to bound pip install time
    if command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD="timeout 120s"
    else
        TIMEOUT_CMD=""
    fi

    # Helper to attempt python import and pip install with retries if missing
    try_install() {
        modulename="$1"; pkg="$2"; max_retries=${3:-3};
        attempt=1
        while true; do
            if python - <<PY >/dev/null 2>&1
import sys
try:
    __import__("${modulename}")
except Exception:
    sys.exit(2)
PY
            then
                log_debug "${modulename} already importable"
                return 0
            fi

            log_warn "${modulename} not found (attempt ${attempt}/${max_retries}), attempting pip install: ${pkg}"

            # Use timeout if available to avoid hangs
            if ${TIMEOUT_CMD} pip install --no-cache-dir --upgrade "${pkg}"; then
                log_info "Successfully installed ${pkg}"
                return 0
            else
                log_warn "pip install failed for ${pkg} (attempt ${attempt})"
            fi

            attempt=$((attempt+1))
            if [[ ${attempt} -gt ${max_retries} ]]; then
                log_warn "Exceeded max retries (${max_retries}) installing ${pkg}; giving up"
                return 1
            fi
            # backoff before retrying
            sleep $((attempt * 2))
        done
    }

    # Install marker extras if requested
    if [[ "${parser}" == "marker" || "${parser}" == "all" || "${parser}" == *"marker"* ]]; then
        try_install marker "marker-pdf>=1.10.0" 3 || log_warn "Marker may still be unavailable"
    fi

    # Install mineru extras if requested
    if [[ "${parser}" == "mineru" || "${parser}" == "all" || "${parser}" == *"mineru"* ]]; then
        try_install mineru "mineru[pipeline]>=2.1.10" 3 || log_warn "MinerU may still be unavailable"
    fi

    # No-op for other parsers; they are included in the base package
}


# Main function
main() {
    log_info "Starting pdfkb-mcp Docker container..."
    log_debug "Entrypoint arguments: $*"

    # Validate environment
    validate_environment

    # Setup directories
    setup_directories

    # Show configuration
    show_configuration

    # Setup health check
    setup_health_check


    # Handle different startup modes
    if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
        log_info "Starting interactive shell..."
        exec "$@"
    elif [[ "${1:-}" == "pdfkb-mcp" ]]; then
        log_info "Starting pdfkb-mcp server..."

        # Try to ensure parser extras are present at runtime. This is
        # best-effort and will not prevent the server from starting if
        # installation fails (network or permissions may block installs).
        ensure_parser_extras_installed

        # Build command arguments
        cmd_args=()

        # Add transport mode
        if [[ -n "${PDFKB_TRANSPORT:-}" ]]; then
            cmd_args+=("--transport" "${PDFKB_TRANSPORT}")
        fi

        # No separate server configuration needed for unified server
        # All configuration is handled via environment variables

        # Add web interface flag if enabled
        if [[ "${PDFKB_WEB_ENABLE:-false}" == "true" ]]; then
            cmd_args+=("--enable-web")
        fi

        # Add log level if specified
        if [[ -n "${PDFKB_LOG_LEVEL:-}" ]]; then
            cmd_args+=("--log-level" "${PDFKB_LOG_LEVEL}")
        fi

        # Combine provided args with built args (skip first arg which is pdfkb-mcp)
        shift
        full_cmd=(pdfkb-mcp "${cmd_args[@]}" "$@")

    log_info "Executing: ${full_cmd[*]}"

    # Diagnostics & safety: enable Python faulthandler and allow core dumps so
    # crashes (native aborts) produce useful traces. Also limit OpenMP/MKL
    # thread counts to reduce risk of native allocator/threading interactions
    # that can lead to heap corruption in some native libraries.
    export PYTHONFAULTHANDLER=1
    ulimit -c unlimited || true
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
    export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

    # Start the server in background to handle signals
    "${full_cmd[@]}" &
    PDFKB_PID=$!

        # Wait for the process to complete
        wait $PDFKB_PID
        exit_code=$?

        log_info "pdfkb-mcp server stopped with exit code: $exit_code"
        exit $exit_code

    else
        # Execute custom command
        log_info "Executing custom command: $*"
        exec "$@"
    fi
}

# Only run main if this script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
