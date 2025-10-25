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

# Optional, optimized runtime installer for parsers
# - Runs only when PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true
# - Installs one or more parsers into /opt/parsers/<parser>/venv
# - Uses a local wheels cache at /opt/parsers/wheels to reduce repeated downloads
# - Supports configurable concurrency via PDFKB_PARSER_INSTALL_CONCURRENCY (default 2)
# - Parsers to install: comma-separated PDFKB_PARSERS_TO_INSTALL or set PDFKB_INSTALL_ALL_PARSERS=true
install_parsers_runtime() {
    if [[ "${PDFKB_ALLOW_RUNTIME_PARSER_INSTALL:-false}" != "true" ]]; then
        log_debug "Runtime parser install disabled (PDFKB_ALLOW_RUNTIME_PARSER_INSTALL!=true)"
        return 0
    fi

    log_info "Runtime parser install enabled — preparing to install requested parsers"

    PARSERS_BASE_DIR="/opt/parsers"
    WHEELS_DIR="${PARSERS_BASE_DIR}/wheels"
    mkdir -p "${PARSERS_BASE_DIR}" "${WHEELS_DIR}"

    # Determine list of parsers to install
    if [[ -n "${PDFKB_PARSERS_TO_INSTALL:-}" ]]; then
        # comma or space separated
        IFS=',' read -r -a _tmp_parsers <<< "${PDFKB_PARSERS_TO_INSTALL}"
    elif [[ "${PDFKB_INSTALL_ALL_PARSERS:-false}" == "true" ]]; then
        _tmp_parsers=(marker docling mineru pymupdf4llm)
    else
        log_info "No parsers requested for runtime install. Set PDFKB_PARSERS_TO_INSTALL or PDFKB_INSTALL_ALL_PARSERS=true to enable."
        return 0
    fi

    # normalize parser list (trim spaces)
    parsers=()
    for p in "${_tmp_parsers[@]:-}"; do
        p_trimmed=$(echo "$p" | xargs)
        if [[ -n "$p_trimmed" ]]; then
            parsers+=("$p_trimmed")
        fi
    done

    concurrency=${PDFKB_PARSER_INSTALL_CONCURRENCY:-2}
    timeout_seconds=${PDFKB_PARSER_INSTALL_TIMEOUT_SEC:-1200}

    log_info "Installing parsers: ${parsers[*]} (concurrency=${concurrency}, timeout=${timeout_seconds}s)"

    declare -a pids=()

    # Ensure lock dir exists for cross-container coordination when /opt/parsers is a shared volume
    mkdir -p "${PARSERS_BASE_DIR}/.locks"
    for parser in "${parsers[@]}"; do
        venv_dir="${PARSERS_BASE_DIR}/${parser}/venv"
        target_dir="${PARSERS_BASE_DIR}/${parser}"

        if [[ -x "${venv_dir}/bin/python" ]]; then
            log_info "Parser '${parser}' already has venv: ${venv_dir} — skipping"
            continue
        fi

        log_info "Scheduling install for parser: ${parser}"
        (
            set -euo pipefail
            mkdir -p "${target_dir}"

            lockfile="${PARSERS_BASE_DIR}/.locks/${parser}.lock"
            status_file="${target_dir}/.install_status"

            # Acquire an exclusive lock per-parser to avoid races when multiple
            # containers start simultaneously using the same mounted /opt/parsers
            log_info "Attempting to acquire lock for ${parser} (lockfile=${lockfile})"
            # Use flock to serialize install operations; block up to timeout_seconds
            if command -v flock >/dev/null 2>&1; then
                exec 9>"${lockfile}"
                if ! flock -w $(( timeout_seconds )) 9; then
                    echo "LOCK_TIMEOUT" > "${status_file}"
                    log_warn "Timed out waiting for lock on ${lockfile}; skipping install"
                    exit 0
                fi
            else
                # If flock is not available, fall back to a simple atomic mkdir lock
                lockdir="${PARSERS_BASE_DIR}/.locks/${parser}.lockdir"
                start=$(date +%s)
                while ! mkdir "${lockdir}" 2>/dev/null; do
                    sleep 1
                    if [[ $(( $(date +%s) - start )) -ge ${timeout_seconds} ]]; then
                        echo "LOCK_TIMEOUT" > "${status_file}"
                        log_warn "Timed out waiting for mkdir-based lock ${lockdir}; skipping install"
                        exit 0
                    fi
                done
            fi

            # At this point we hold the lock (via fd 9 or lockdir). Report progress.
            echo "IN_PROGRESS: started $(date --iso-8601=seconds) pid=$$" > "${status_file}"

            if ! command -v python3 >/dev/null 2>&1; then
                echo "NO_PYTHON" > "${status_file}"
                log_error "python3 not available in image; cannot create virtualenv for parser ${parser}"
                # release lock
                if [[ -n "${lockdir:-}" && -d "${lockdir}" ]]; then rmdir "${lockdir}" || true; fi
                exit 1
            fi

            log_info "Creating venv for ${parser} at ${venv_dir}"
            python3 -m venv "${venv_dir}"
            "${venv_dir}/bin/pip" install --upgrade pip setuptools wheel || true

            case "${parser}" in
                marker)
                    pkg_spec="marker-pdf>=1.10.0"
                    ;;
                mineru)
                    pkg_spec="mineru[pipeline]>=2.1.10"
                    ;;
                docling)
                    pkg_spec="docling>=2.43.0"
                    ;;
                pymupdf4llm)
                    pkg_spec="pymupdf4llm>=0.0.27"
                    ;;
                *)
                    pkg_spec="${parser}"
                    ;;
            esac

            if [[ -n "${pkg_spec:-}" ]]; then
                echo "DOWNLOADING: $(date --iso-8601=seconds) pkg=${pkg_spec}" >> "${status_file}"
                log_info "Downloading wheels for ${pkg_spec} to ${WHEELS_DIR} (best-effort)"
                # Try to download wheels first to the local cache. If it fails, fall back to direct install.
                if ! "${venv_dir}/bin/pip" download --dest "${WHEELS_DIR}" --prefer-binary "${pkg_spec}" 2>/dev/null; then
                    log_warn "Wheel download failed for ${pkg_spec}; will attempt direct install from PyPI"
                fi

                echo "INSTALLING_FROM_WHEELS: $(date --iso-8601=seconds) pkg=${pkg_spec}" >> "${status_file}"
                log_info "Installing ${pkg_spec} into venv ${venv_dir} using wheels cache"
                if ! timeout "${timeout_seconds}" "${venv_dir}/bin/pip" install --no-cache-dir --upgrade --find-links "${WHEELS_DIR}" "${pkg_spec}" >/dev/null 2>&1; then
                    log_warn "Install from wheels cache failed for ${pkg_spec}; attempting network install"
                    if ! timeout "${timeout_seconds}" "${venv_dir}/bin/pip" install --no-cache-dir --upgrade "${pkg_spec}"; then
                        echo "FAILED: $(date --iso-8601=seconds) pkg=${pkg_spec}" > "${status_file}"
                        log_error "Runtime install failed for ${parser} (pkg: ${pkg_spec})"
                        # release lock
                        if [[ -n "${lockdir:-}" && -d "${lockdir}" ]]; then rmdir "${lockdir}" || true; fi
                        exit 1
                    fi
                fi
                echo "DONE: $(date --iso-8601=seconds)" > "${status_file}"
                log_info "Parser ${parser} installed into ${venv_dir}"
            else
                echo "NO_PKG_SPEC" > "${status_file}"
                log_warn "No package spec known for parser '${parser}'; skipping install"
            fi

            # release lock
            if [[ -n "${lockdir:-}" && -d "${lockdir}" ]]; then rmdir "${lockdir}" || true; fi
            if [[ -n "${lockfile:-}" && -e "${lockfile}" ]]; then : >/dev/null; fi
            # If we created fd 9 earlier, closing script will release flock automatically
        ) &

        pids+=("$!")

        # simple concurrency control using PID list
        while true; do
            live=0
            for pid in "${pids[@]:-}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    live=$((live+1))
                fi
            done
            if [[ $live -lt $concurrency ]]; then
                break
            fi
            sleep 1
        done
    done

    # Wait for all installers to finish
    for pid in "${pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" || log_warn "One of the parser installs exited with non-zero status"
        fi
    done

    log_info "Runtime parser installation finished. To persist installs across container restarts, mount ${PARSERS_BASE_DIR} as a volume."

    # Simple progress reporter: aggregate per-parser status files and write a
    # single summary file that local health checks or orchestrators can inspect.
    summary_file="/app/parser_install_status"
    echo "Parser install summary: $(date --iso-8601=seconds)" > "${summary_file}"
    for parser in "${parsers[@]}"; do
        status_file="${PARSERS_BASE_DIR}/${parser}/.install_status"
        if [[ -f "${status_file}" ]]; then
            echo "${parser}: $(tail -n 5 "${status_file}" | tr '\n' ' | ')" >> "${summary_file}"
        else
            echo "${parser}: NOT_PRESENT" >> "${summary_file}"
        fi
    done

    log_info "Wrote parser install summary to ${summary_file}"
}


# Select parser runtime environment without performing runtime uninstalls.
# Strategy:
# - If /opt/parsers/<parser>/venv exists, prefer that virtualenv by
#   prepending its bin to PATH and setting VIRTUAL_ENV so the process
#   runs fully isolated from the base image site-packages.
# - Else if /opt/parsers/<parser> exists (pip --target installs), prepend
#   it to PYTHONPATH for the current run.
# - Do NOT attempt to uninstall packages at runtime (unsafe). A runtime
#   pip install fallback is allowed only when explicitly enabled by
#   PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true (not recommended for CI/production).
select_parser_environment() {
    parser="${PDFKB_PDF_PARSER:-pymupdf4llm}"
    log_info "Selecting runtime environment for parser: ${parser}"

    PARSERS_BASE_DIR="/opt/parsers"
    venv_dir="${PARSERS_BASE_DIR}/${parser}/venv"
    target_dir="${PARSERS_BASE_DIR}/${parser}"

    # If a built virtualenv exists for this parser, prefer it
    if [[ -x "${venv_dir}/bin/python" ]]; then
        export VIRTUAL_ENV="${venv_dir}"
        export PATH="${VIRTUAL_ENV}/bin:${PATH}"
        log_info "Using parser virtualenv: ${VIRTUAL_ENV} (PATH updated)"
        return 0
    fi

    # If a pip --target directory exists, prepend it to PYTHONPATH
    if [[ -d "${target_dir}" ]]; then
        export PYTHONPATH="${target_dir}:${PYTHONPATH:-}"
        log_info "Using parser target dir: ${target_dir} (prepended to PYTHONPATH)"
        return 0
    fi

    log_warn "No baked parser environment found for '${parser}' in ${PARSERS_BASE_DIR}"
    if [[ "${PDFKB_ALLOW_RUNTIME_PARSER_INSTALL:-false}" == "true" ]]; then
        log_warn "PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true — attempting best-effort runtime install into ${target_dir}"
        mkdir -p "${target_dir}"
        # Use a short, conservative install attempt (no retries). This is
        # intentionally minimal; prefer build-time installs in Dockerfile.
        if command -v timeout >/dev/null 2>&1; then
            timeout_cmd=(timeout 120s)
        else
            timeout_cmd=()
        fi

        case "${parser}" in
            marker)
                pkg_spec="marker-pdf>=1.10.0"
                modulename="marker"
                ;;
            mineru)
                pkg_spec="mineru[pipeline]>=2.1.10"
                modulename="mineru"
                ;;
            docling)
                pkg_spec="docling"
                modulename="docling"
                ;;
            *)
                pkg_spec=""
                modulename="${parser}"
                ;;
        esac

        if [[ -n "${pkg_spec}" ]]; then
            log_info "Attempting runtime pip install ${pkg_spec} into ${target_dir} (best-effort)"
            if ("${timeout_cmd[@]}" pip install --no-cache-dir --upgrade --target "${target_dir}" "${pkg_spec}"); then
                export PYTHONPATH="${target_dir}:${PYTHONPATH:-}"
                log_info "Runtime install succeeded; prepended ${target_dir} to PYTHONPATH"
            else
                log_warn "Runtime pip install failed for ${pkg_spec}; parser may be unavailable"
            fi
        else
            log_warn "No install spec known for parser '${parser}'; skipping runtime install"
        fi
    else
        log_info "Runtime parser installs are disabled (PDFKB_ALLOW_RUNTIME_PARSER_INSTALL not true). To enable, set PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true — note: runtime installs are not recommended for production."
    fi

    return 0
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

    # Optionally install requested parsers at container start (if enabled).
    # This installs per-parser virtualenvs under /opt/parsers/<parser>/venv
    # and caches wheels under /opt/parsers/wheels to reduce repeated downloads.
    install_parsers_runtime

    # Select parser runtime environment. Prefer baked venvs or pip
    # --target directories. Runtime installs are allowed only if
    # PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true and are not recommended.
    select_parser_environment

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
