# dev.md

This file provides development guidelines for Kilo Code when working with code in this repository.

## Project Overview

This is pdfkb-mcp, a Model Context Protocol (MCP) server that enables intelligent document search and retrieval from PDF collections. It features semantic search capabilities powered by local, OpenAI, or HuggingFace embeddings and ChromaDB vector storage, with both MCP protocol integration and a modern web interface.

Key features include:
- Document Summarization: Automatic generation of document titles, short descriptions, and detailed summaries
- Reranking Support: Advanced result reranking for improved search relevance
- GGUF Quantized Models: Memory-optimized local embeddings and rerankers
- Hybrid Search: Combines semantic similarity with keyword matching (BM25)
- Minimum Chunk Filtering: Filters out short, low-information chunks
- Semantic Chunking: Content-aware chunking using embedding similarity
- Local Embeddings: Run embeddings locally with full privacy
- Web Interface: Modern web UI for document management alongside MCP protocol
- Markdown Document Support: Native support for .md files with frontmatter parsing

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
- Run only fast tests during development: `hatch run test -m "not slow"`
- Test against different Python versions when needed

## Configuration Management

- Main config class: `ServerConfig` in `src/pdfkb/config.py`
- Environment variables prefixed with `PDFKB_`
- Parser and chunker selection via `PDFKB_PDF_PARSER` and `PDFKB_PDF_CHUNKER`
- Web interface disabled by default (enable with `PDFKB_WEB_ENABLE=true`)

Essential environment variables:
- `PDFKB_OPENAI_API_KEY`: Required only for OpenAI embeddings (local embeddings are default)
- `PDFKB_OPENAI_API_BASE`: Custom base URL for OpenAI-compatible endpoints
- `HF_TOKEN`: Required for HuggingFace embeddings
- `PDFKB_KNOWLEDGEBASE_PATH`: PDF directory path
- `PDFKB_WEB_ENABLE`: Web interface control (default: `false`)

## Key Files and Their Roles

- `src/pdfkb/main.py`: MCP server implementation with tools (`add_document`, `search_documents`, `list_documents`, `remove_document`)
- `src/pdfkb/pdf_processor.py`: Document processing orchestrator
- `src/pdfkb/intelligent_cache.py:139`: Multi-stage caching system with smart invalidation
- `src/pdfkb/web/server.py`: Web interface API endpoints
- `src/pdfkb/parsers/`: Modular PDF parser implementations
- `src/pdfkb/chunker/`: Text chunking strategies (LangChain, Unstructured, Semantic, Page-based)

## Version Management

Version is managed by `bump2version` - never manually change version numbers. Only bump version when explicitly requested.

## Environment Setup

1. **Install Hatch** (if not already installed):
   ```bash
   pipx install hatch
   ```

2. **Enter development environment**:
   ```bash
   hatch shell
   ```

3. **Install project in editable mode**:
   ```bash
   pip install -e .[dev]
   ```

## Common Development Tasks

### Adding New Features

- **Add new parser**: Create in `src/pdfkb/parsers/parser_newname.py`, implement `PDFParser` interface
- **Modify caching**: Edit `src/pdfkb/intelligent_cache.py`, understand invalidation rules
- **Add web endpoints**: Extend `src/pdfkb/web/server.py`
- **Change chunking**: Modify chunker classes in `src/pdfkb/chunker/`

### Working with Optional Dependencies

The project includes several optional dependency groups:
- `unstructured`: Unstructured.io PDF processing
- `pymupdf4llm`: PyMuPDF for LLM workflows
- `langchain`: LangChain text splitters
- `mineru`: MinerU pipeline
- `marker`: Marker PDF processing
- `docling`: IBM Docling (basic)
- `docling-complete`: IBM Docling with OCR capabilities
- `llm`: Additional LLM utilities
- `unstructured_chunker`: Unstructured chunking utilities
- `all`: All optional dependencies combined

Install specific groups:
```bash
pip install -e ".[marker]"
pip install -e ".[docling]"
pip install -e ".[mineru]"
pip install -e ".[llm]"
```

### Development Workflow

1. **Setting Up**:
   ```bash
   git clone https://github.com/juanqui/pdfkb-mcp.git
   cd pdfkb-mcp
   hatch shell
   pip install -e .[dev]
   ```

2. **Making Changes**:
   ```bash
   git checkout -b feature/your-feature
   # Make your changes...
   hatch run format
   hatch run lint
   hatch run test
   ```

3. **Before Committing**:
   ```bash
   hatch run test-cov
   hatch run lint
   hatch build
   ```

## Code Quality Standards

- Follow Black formatting with 120 character line length
- Use isort for import organization
- Maintain strict type checking with mypy
- Write comprehensive tests for new features
- Use descriptive variable and function names
- Include docstrings for all public functions and classes

## Python Version Compatibility

The project supports Python 3.8 through 3.12:
```bash
hatch env create py38 --python=3.8
hatch env create py39 --python=3.9
hatch env create py311 --python=3.11
hatch env create py312 --python=3.12
hatch run --env-name py311 test
```

## Commit Message Conventions

Use conventional commit prefixes:
- `feat:` - New features
- `bugfix:` - Bug fixes
- `chore:` - Maintenance tasks
- `docs:` - Documentation updates
- `test:` - Test-related changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements

Example: `feat: add semantic chunking support`

## Best Practices

1. **Development Environment**:
   - Always work within the Hatch shell environment
   - Install only needed optional dependencies
   - Use .env files for local configuration

2. **Code Changes**:
   - Format code before committing (`hatch run format`)
   - Run linters to check code quality (`hatch run lint`)
   - Ensure all tests pass (`hatch run test`)

3. **Testing**:
   - Write unit tests for individual components
   - Include integration tests for component interactions
   - Mark slow tests appropriately to allow fast test runs

4. **Documentation**:
   - Update relevant documentation when adding features
   - Use Mermaid charts for diagrams when appropriate
   - Keep README.md and DEVELOPMENT.md in sync with code changes
