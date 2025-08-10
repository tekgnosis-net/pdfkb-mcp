# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is pdfkb-mcp, a Model Context Protocol (MCP) server that provides intelligent document search and retrieval from PDF collections. It features semantic search capabilities powered by OpenAI embeddings and ChromaDB vector storage, with both MCP protocol integration and a modern web interface.

## Architecture

### Core Components

- **MCP Server** (`src/pdfkb/main.py`): FastMCP-based server providing tools for document management
- **PDF Processing Pipeline**: Multiple parsers (PyMuPDF4LLM, Marker, MinerU, Docling, LLM) with intelligent caching
- **Vector Store** (`src/pdfkb/vector_store.py`): ChromaDB-based semantic search
- **Web Interface** (`src/pdfkb/web/`): FastAPI-based web server with WebSocket support
- **Configuration System** (`src/pdfkb/config.py`): Environment-based configuration with comprehensive options

### Key Architecture Patterns

- **Plugin-based Parsers**: Modular PDF parser system in `src/pdfkb/parsers/` with fallback mechanisms
- **Intelligent Caching** (`src/pdfkb/intelligent_cache.py`): Multi-stage caching that invalidates appropriately when configuration changes
- **Background Processing**: Non-blocking document processing queue
- **Dual Interface**: Both MCP protocol and web UI share the same underlying services

## Development Commands

Use Hatch for all development tasks:

```bash
# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Format code (Black + isort)
hatch run format

# Lint code (Black, isort, flake8)
hatch run lint

# Generate HTML coverage report
hatch run cov-html
```

**Important**: Always run `hatch run test`, `hatch run format`, and `hatch run lint` after significant changes.

## Testing Guidelines

- Do not run the web server during tests as it's blocking
- Use pytest markers: `unit`, `integration`, `slow`, `performance`, `asyncio`
- Test files follow patterns: `test_*.py` or `*_test.py`

## Configuration Management

- Main config class: `ServerConfig` in `src/pdfkb/config.py`
- Environment variables prefixed with `PDFKB_`
- Parser and chunker selection via `PDFKB_PDF_PARSER` and `PDFKB_PDF_CHUNKER`
- Web interface enabled via `PDFKB_ENABLE_WEB=true`

## Key Files and Their Roles

- `src/pdfkb/main.py`: MCP server implementation with tools (`add_document`, `search_documents`, `list_documents`, `remove_document`)
- `src/pdfkb/pdf_processor.py`: Document processing orchestrator
- `src/pdfkb/intelligent_cache.py:139`: Multi-stage caching system with smart invalidation
- `src/pdfkb/web/server.py`: Web interface API endpoints
- `src/pdfkb/parsers/`: Modular PDF parser implementations
- `src/pdfkb/chunker/`: Text chunking strategies (LangChain, Unstructured)

## Version Management

Version is managed by `bump2version` - never manually change version numbers. Only bump version when explicitly requested.

## Environment Setup

Essential environment variables:
- `PDFKB_OPENAI_API_KEY`: Required for embeddings
- `PDFKB_KNOWLEDGEBASE_PATH`: PDF directory path
- `PDFKB_ENABLE_WEB`: Enable web interface (`true`/`false`)

Optional parsers require additional installations:
- `pip install "pdfkb-mcp[marker]"` for Marker parser
- `pip install "pdfkb-mcp[docling]"` for Docling parser
- `pip install "pdfkb-mcp[mineru]"` for MinerU parser
- `pip install "pdfkb-mcp[llm]"` for LLM parser

## Common Tasks

- **Add new parser**: Create in `src/pdfkb/parsers/parser_newname.py`, implement `PDFParser` interface
- **Modify caching**: Edit `src/pdfkb/intelligent_cache.py`, understand invalidation rules
- **Add web endpoints**: Extend `src/pdfkb/web/server.py`
- **Change chunking**: Modify chunker classes in `src/pdfkb/chunker/`
