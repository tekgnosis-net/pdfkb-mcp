#!/usr/bin/env bash

# run-local.sh - Run PDF-KB MCP server locally using hatch
#
# This script runs the PDF-KB server locally using the same configuration
# as the Docker container, but adapted for local development.
#
# Usage:
#   ./run-local.sh                    # Run with default settings
#   ./run-local.sh --transport http   # Run with HTTP transport
#   ./run-local.sh --enable-web       # Force enable web interface
#   ./run-local.sh --config .env.dev  # Use custom config file

set -euo pipefail

# Get the script directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run PDF-KB MCP server locally using hatch with the same configuration as the container.

Options:
    --transport MODE    MCP transport mode (stdio, http, sse) [default: http]
    --enable-web        Force enable web interface
    --config FILE       Load configuration from custom file [default: .env]
    --log-level LEVEL   Override log level (DEBUG, INFO, WARNING, ERROR)
    --help             Show this help message

Examples:
    $0                           # Run with web interface on localhost:8000
    $0 --transport sse           # Run with SSE transport
    $0 --config .env.dev         # Use custom config file
    $0 --log-level DEBUG         # Enable debug logging

Endpoints (when web is enabled):
    Web Interface: http://localhost:8000/
    MCP (HTTP):    http://localhost:8000/mcp/
    MCP (SSE):     http://localhost:8000/sse/
    API Docs:      http://localhost:8000/docs

Note: This script automatically sets up local paths and enables the web interface
      by default for easier development testing.
EOF
}

# Initialize variables
TRANSPORT="http"
ENABLE_WEB="true"
CONFIG_FILE=".env"
LOG_LEVEL=""
ADDITIONAL_ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --transport)
            TRANSPORT="$2"
            shift 2
            ;;
        --enable-web)
            ENABLE_WEB="true"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        --*)
            # Pass through any other arguments to the main script
            ADDITIONAL_ARGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                ADDITIONAL_ARGS+=("$2")
                shift 2
            else
                shift
            fi
            ;;
        *)
            error "Unknown argument: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate transport option
if [[ ! "$TRANSPORT" =~ ^(stdio|http|sse)$ ]]; then
    error "Invalid transport mode: $TRANSPORT"
    error "Valid options: stdio, http, sse"
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/pdfkb" ]]; then
    error "Please run this script from the pdfkb-mcp project root directory"
    exit 1
fi

# Create documents directory if it doesn't exist
if [[ ! -d "documents" ]]; then
    info "Creating documents directory..."
    mkdir -p documents
fi

# Create cache directory if it doesn't exist
if [[ ! -d "documents/.cache" ]]; then
    info "Creating cache directory..."
    mkdir -p documents/.cache
fi

info "🚀 Starting PDF-KB MCP server locally..."
info "📁 Project root: $SCRIPT_DIR"
info "📄 Config file: $CONFIG_FILE"
info "🚌 Transport mode: $TRANSPORT"
info "🌐 Web interface: $ENABLE_WEB"

# Set up environment variables for local development
# These override/supplement what's in the .env file

# Export variables that need to be set for local development
export PDFKB_KNOWLEDGEBASE_PATH="$SCRIPT_DIR/documents"
export PDFKB_CACHE_DIR="$SCRIPT_DIR/documents/.cache"
export PDFKB_WEB_ENABLE="$ENABLE_WEB"
export PDFKB_WEB_HOST="localhost"  # Override container's 0.0.0.0 for local use
export PDFKB_TRANSPORT="$TRANSPORT"

# Set log level if specified
if [[ -n "$LOG_LEVEL" ]]; then
    export PDFKB_LOG_LEVEL="$LOG_LEVEL"
fi

# Load the specified config file
if [[ -f "$CONFIG_FILE" ]]; then
    info "📥 Loading configuration from: $CONFIG_FILE"
    set -a  # Export all variables
    source "$CONFIG_FILE"
    set +a  # Stop exporting
fi

# Show key configuration
info "Configuration overview:"
info "  📂 Documents path: $PDFKB_KNOWLEDGEBASE_PATH"
info "  🗃️  Cache directory: $PDFKB_CACHE_DIR"
info "  🌐 Web enabled: $PDFKB_WEB_ENABLE"
info "  🏠 Host: ${PDFKB_WEB_HOST:-localhost}"
info "  🔌 Port: ${PDFKB_WEB_PORT:-8000}"
info "  🚌 Transport: $PDFKB_TRANSPORT"

# Check for hatch
if ! command -v hatch &> /dev/null; then
    error "hatch is not installed or not in PATH"
    error "Please install hatch: pip install hatch"
    exit 1
fi

# Build command arguments
CMD_ARGS=()
CMD_ARGS+=("--transport" "$TRANSPORT")

if [[ "$ENABLE_WEB" == "true" ]]; then
    CMD_ARGS+=("--enable-web")
fi

if [[ -n "$LOG_LEVEL" ]]; then
    CMD_ARGS+=("--log-level" "$LOG_LEVEL")
fi

# Add any additional arguments passed through
CMD_ARGS+=("${ADDITIONAL_ARGS[@]:+${ADDITIONAL_ARGS[@]}}")

# Show the URLs where the service will be available
if [[ "$ENABLE_WEB" == "true" ]]; then
    PORT="${PDFKB_WEB_PORT:-8000}"
    HOST="${PDFKB_WEB_HOST:-localhost}"

    echo
    success "🌍 Service will be available at:"
    success "  📖 Web Interface: http://$HOST:$PORT/"
    success "  📡 API Documentation: http://$HOST:$PORT/docs"

    if [[ "$TRANSPORT" == "http" ]]; then
        success "  🔗 MCP HTTP Endpoint: http://$HOST:$PORT/mcp/"
    elif [[ "$TRANSPORT" == "sse" ]]; then
        success "  📡 MCP SSE Endpoint: http://$HOST:$PORT/sse/"
    fi
    echo
fi

info "🚀 Starting server with hatch..."

# Run the server with hatch
exec hatch run python -m pdfkb.main "${CMD_ARGS[@]}"
