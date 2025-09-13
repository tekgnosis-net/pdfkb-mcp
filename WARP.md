# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**pdfkb-mcp** is a Model Context Protocol (MCP) server that provides intelligent document search and retrieval from PDF and Markdown collections. It features semantic search capabilities powered by local, OpenAI, or HuggingFace embeddings with ChromaDB vector storage, plus a modern web interface.

## Architecture Overview

### Core Components

- **MCP Server** (`src/pdfkb/main.py`): FastMCP-based server providing tools (`add_document`, `search_documents`, `list_documents`, `remove_document`)
- **Document Processing Pipeline**: Multi-parser system with intelligent caching and background processing
- **Vector Store** (`src/pdfkb/vector_store.py`): ChromaDB-based semantic search with hybrid search support
- **Web Interface** (`src/pdfkb/web/`): FastAPI-based web server with WebSocket support
- **Configuration System** (`src/pdfkb/config.py`): Environment-based configuration with comprehensive options

### Key Architecture Patterns

- **Plugin-based System**: Modular PDF parsers (`src/pdfkb/parsers/`) and chunkers (`src/pdfkb/chunker/`) with fallback mechanisms
- **Intelligent Caching** (`src/pdfkb/intelligent_cache.py`): Multi-stage caching with selective invalidation based on configuration changes
- **Background Processing**: Non-blocking document processing queue with semaphore-controlled parallelism
- **Dual Interface**: Both MCP protocol and web UI share the same underlying services

## Development Commands

Use **Hatch** for all development tasks:

```bash
# Run tests
hatch run test

# Run tests with coverage reporting
hatch run test-cov

# Generate HTML coverage report
hatch run cov-html

# Format code (Black + isort)
hatch run format

# Lint code (Black, isort, flake8)
hatch run lint

# Run the MCP server in development
hatch run python -m pdfkb.main

# Run with web interface enabled
PDFKB_WEB_ENABLE=true hatch run python -m pdfkb.main

# Run specific tests
hatch run test tests/test_pdf_processor.py
hatch run test -k "test_embeddings"
hatch run test -m "not slow"  # Skip slow tests
```

## Configuration and Environment

### Essential Environment Variables

```bash
# Embedding provider (default: "local" - no API key required)
PDFKB_EMBEDDING_PROVIDER="local"  # "local", "openai", "huggingface"

# For OpenAI embeddings
PDFKB_OPENAI_API_KEY="sk-proj-..."

# For HuggingFace embeddings
HF_TOKEN="hf_..."

# Directory paths
PDFKB_KNOWLEDGEBASE_PATH="/path/to/pdfs"
PDFKB_CACHE_DIR="./.cache"

# Web interface (disabled by default)
PDFKB_WEB_ENABLE="false"
PDFKB_WEB_PORT="8080"
PDFKB_WEB_HOST="localhost"

# Parser and chunker selection
PDFKB_PDF_PARSER="pymupdf4llm"  # pymupdf4llm, marker, mineru, docling, llm
PDFKB_PDF_CHUNKER="langchain"   # langchain, page, semantic, unstructured

# Hybrid search (enabled by default)
PDFKB_ENABLE_HYBRID_SEARCH="true"

# Reranking (optional)
PDFKB_ENABLE_RERANKER="false"
PDFKB_RERANKER_MODEL="Qwen/Qwen3-Reranker-0.6B"

# Document summarization (optional)
PDFKB_ENABLE_SUMMARIZER="false"
PDFKB_SUMMARIZER_PROVIDER="local"  # "local", "remote"
```

### Optional Dependency Groups

Install based on features needed:

```bash
# Core install (includes web interface, local embeddings, hybrid search)
pip install -e .

# With specific parsers
pip install -e ".[marker]"     # Marker parser
pip install -e ".[docling]"    # Docling parser
pip install -e ".[mineru]"     # MinerU parser
pip install -e ".[llm]"        # LLM parser

# With advanced chunking
pip install -e ".[semantic]"   # Semantic chunking
pip install -e ".[unstructured_chunker]"  # Unstructured chunking

# Development setup
pip install -e ".[dev]"
```

## Testing Strategy

### Test Organization

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Test performance characteristics
- **Slow tests**: Long-running tests (marked for optional execution)

### Running Tests

```bash
# Run all tests
hatch run test

# Run only fast tests
hatch run test -m "not slow"

# Run specific test types
hatch run test -m integration
hatch run test -m unit

# Run with coverage
hatch run test-cov

# Run specific test files
hatch run test tests/test_embeddings.py
```

## Common Development Tasks

### Adding a New PDF Parser

1. Create `src/pdfkb/parsers/parser_newname.py`
2. Implement the `PDFParser` interface from `src/pdfkb/parsers/parser.py`
3. Add parser selection logic in `src/pdfkb/parsers/__init__.py`
4. Add optional dependency group in `pyproject.toml`
5. Update configuration in `src/pdfkb/config.py`

### Adding New Embeddings Provider

1. Create `src/pdfkb/embeddings_newprovider.py`
2. Implement the `EmbeddingService` interface from `src/pdfkb/embeddings_base.py`
3. Update factory in `src/pdfkb/embeddings_factory.py`
4. Add configuration options in `src/pdfkb/config.py`

### Modifying Caching Behavior

- Edit `src/pdfkb/intelligent_cache.py`
- Understand the multi-stage invalidation rules:
  - Parser changes → Full reset (parsing + chunking + embeddings)
  - Chunker changes → Partial reset (chunking + embeddings)
  - Embedding changes → Minimal reset (embeddings only)

### Adding Web Endpoints

1. Extend `src/pdfkb/web/server.py`
2. Add WebSocket handlers in `src/pdfkb/web/websocket_handlers.py` if needed
3. Update middleware in `src/pdfkb/web/middleware.py` for CORS/security

## Code Quality and Standards

- **Formatting**: Black (120 character line length) + isort
- **Linting**: flake8 with Black-compatible rules
- **Type Checking**: mypy with strict configuration (currently disabled in lint script)
- **Testing**: pytest with async support, markers for test organization
- **Documentation**: Use type hints and docstrings for all public functions

## Performance Considerations

### Parallel Processing Control

The system uses semaphores to prevent overload:

```bash
# Control concurrent operations
PDFKB_MAX_PARALLEL_PARSING=1     # PDF parsing operations
PDFKB_MAX_PARALLEL_EMBEDDING=1   # Embedding operations
PDFKB_BACKGROUND_QUEUE_WORKERS=2 # Background workers
PDFKB_THREAD_POOL_SIZE=1         # CPU-intensive tasks
```

### Memory Management

- Embedding batch sizes are configurable and auto-adjust on OOM
- Intelligent caching prevents reprocessing unchanged documents
- Local embeddings support hardware acceleration (MPS, CUDA, CPU)

## Server Modes

### MCP-Only Mode (Default)
```bash
pdfkb-mcp  # stdio transport by default
pdfkb-mcp --transport http --server-port 8000  # HTTP transport (accessible at http://localhost:8000/mcp/)
pdfkb-mcp --transport sse --server-port 8000   # SSE transport (accessible at http://localhost:8000/sse/)
```

### Integrated Mode (MCP + Web)
```bash
PDFKB_WEB_ENABLE=true pdfkb-mcp  # Web interface only (no remote MCP)

# HTTP transport (for Cline, modern MCP clients)
PDFKB_WEB_ENABLE=true pdfkb-mcp --transport http --server-port 8084
# → Web: http://localhost:8084/, MCP: http://localhost:8085/mcp/, Docs: http://localhost:8084/docs

# SSE transport (for Roo, legacy MCP clients)
PDFKB_WEB_ENABLE=true pdfkb-mcp --transport sse --server-port 8084
# → Web: http://localhost:8084/, MCP: http://localhost:8085/sse/, Docs: http://localhost:8084/docs
```

## Key Files and Their Roles

- `src/pdfkb/main.py`: MCP server implementation and entry point
- `src/pdfkb/config.py`: Central configuration management with environment variable handling
- `src/pdfkb/intelligent_cache.py`: Multi-stage caching system with smart invalidation (line 139)
- `src/pdfkb/document_processor.py`: Document processing orchestrator
- `src/pdfkb/vector_store.py`: ChromaDB integration with hybrid search support
- `src/pdfkb/embeddings_factory.py`: Embedding provider factory with local/OpenAI/HuggingFace support
- `src/pdfkb/web_server.py`: Integrated server that runs both MCP and web interfaces
- `src/pdfkb/parsers/`: Modular PDF parser implementations
- `src/pdfkb/chunker/`: Text chunking strategies (LangChain, semantic, page-based, unstructured)
- `src/pdfkb/hybrid_search.py`: Hybrid search implementation combining vector and BM25 search

## Version Management

- Version is managed by `bump2version` - never manually change version numbers
- Only bump version when explicitly requested
- Version is defined in `src/pdfkb/__init__.py`

## Important Notes

- Do not run the web server during tests as it's blocking
- The web interface is **disabled by default** and must be explicitly enabled
- Local embeddings are the default (no API key required) using Qwen3-Embedding models
- Hybrid search (vector + BM25) is enabled by default for better search quality
- The system supports both PDF and Markdown document processing
- Multiple transport modes: stdio (default) and SSE for remote access
- Background processing queue prevents blocking operations
- Intelligent caching system minimizes reprocessing on configuration changes
