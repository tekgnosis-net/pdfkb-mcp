# Development Guide for pdfkb-mcp

This project uses [Hatch](https://hatch.pypa.io/latest/) for dependency management, environment handling, and build processes. This guide will help you get started with development.

## Prerequisites

- Python 3.8 or higher
- [Hatch](https://hatch.pypa.io/latest/install/) installed globally

### Installing Hatch

```bash
# Install via pipx (recommended)
pipx install hatch

# Or via pip
pip install hatch

# Or via conda
conda install -c conda-forge hatch
```

## Project Overview

pdfkb-mcp is a Model Context Protocol server for managing PDF documents with vector search capabilities. The project supports multiple PDF processing backends through optional dependency groups.

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/juanqui/pdfkb-mcp.git
   cd pdfkb-mcp
   ```

2. **Create and activate the default development environment**
   ```bash
   # Hatch will automatically create a virtual environment and install dependencies
   hatch shell
   ```

3. **Verify the installation**
   ```bash
   # Run tests to ensure everything is working
   hatch run test
   ```

## Environment Management

### Default Development Environment

The project includes a pre-configured default environment with all essential development dependencies and some common optional dependencies for testing:

```bash
# Enter the development shell
hatch shell

# Run commands in the environment without entering the shell
hatch run <command>

# Show available environments
hatch env show
```

### Working with Optional Dependencies

The project includes several optional dependency groups for different PDF processing backends:

- **`unstructured`**: Unstructured.io PDF processing
- **`pymupdf4llm`**: PyMuPDF for LLM workflows
- **`langchain`**: LangChain text splitters
- **`mineru`**: MinerU pipeline
- **`marker`**: Marker PDF processing
- **`docling`**: IBM Docling (basic)
- **`docling-complete`**: IBM Docling with OCR capabilities
- **`llm`**: Additional LLM utilities
- **`unstructured_chunker`**: Unstructured chunking utilities
- **`all`**: All optional dependencies combined (warning: very large installation)

#### Installing Optional Dependencies

```bash
# Install the project in editable mode with specific extras
pip install -e ".[unstructured]"

# Install multiple groups
pip install -e ".[unstructured,docling]"

# Install all optional dependencies (use with caution - very large)
pip install -e ".[all]"

# For development with common extras
pip install -e ".[dev,unstructured,docling]"
```

#### Creating Custom Environments

For working with specific dependency combinations, you can create custom environments:

```bash
# Create environment for specific Python version
hatch env create py311 --python=3.11

# Use a specific environment
hatch shell --env-name py311
```

## Development Tasks

All common development tasks are configured as Hatch scripts. Use these commands:

### Testing

```bash
# Run all tests
hatch run test

# Run tests with coverage reporting
hatch run test-cov

# Generate HTML coverage report (opens in browser)
hatch run cov-html

# Run specific test files or patterns
hatch run test tests/test_pdf_processor.py
hatch run test -k "test_embeddings"
hatch run test -v --tb=short

# Run tests with specific markers
hatch run test -m "not slow"  # Skip slow tests
hatch run test -m integration  # Run only integration tests
```

### Code Quality

```bash
# Format code with black and isort
hatch run format

# Run linters and format checks
hatch run lint

# Run individual tools manually if needed
hatch run black --check src tests
hatch run isort --check-only src tests
hatch run flake8 src tests
```

### Running the MCP Server

```bash
# Run the MCP server in development
hatch run python -m pdfkb.main

# Or use the installed console script (after pip install -e .)
hatch run pdfkb-mcp

# Run with environment variables
OPENAI_API_KEY=your-key hatch run python -m pdfkb.main
```

## Testing Strategy

### Test Organization

Tests are organized by functionality:
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Test performance characteristics
- **Slow tests**: Long-running tests (marked for optional execution)

### Running Different Test Types

```bash
# Run only fast tests (skip slow ones)
hatch run test -m "not slow"

# Run only integration tests
hatch run test -m integration

# Run only unit tests
hatch run test -m unit

# Run with verbose output
hatch run test -v

# Run with coverage and generate report
hatch run test-cov
```

### Test Configuration

The project uses pytest with the following key configurations:
- Async test support via `pytest-asyncio`
- Coverage reporting via `pytest-cov`
- Mocking support via `pytest-mock`
- Strict marker and config enforcement

## Development Workflow

### 1. Setting Up for Development

```bash
# Clone and enter the project
git clone <repository-url>
cd pdfkb-mcp

# Set up development environment
hatch shell

# Install with development dependencies
pip install -e .[dev]

# Install pre-commit hooks (recommended)
pre-commit install
```

### 2. Making Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature

# Make your changes...

# Format and lint your code
hatch run format
hatch run lint

# Run tests to ensure nothing is broken
hatch run test
```

### 3. Before Committing

```bash
# Run full test suite with coverage
hatch run test-cov

# Ensure code is properly formatted
hatch run lint

# Check that build works
hatch build
```

## Python Version Compatibility

The project supports Python 3.8 through 3.12. To test against different Python versions:

```bash
# Create environments for different Python versions
hatch env create py38 --python=3.8
hatch env create py39 --python=3.9
hatch env create py311 --python=3.11
hatch env create py312 --python=3.12

# Test against specific version
hatch run --env-name py311 test
```

## Configuration

### Environment Variables

Key environment variables for development:

- **`OPENAI_API_KEY`**: Required for embedding functionality
- **`CHROMA_HOST`**: ChromaDB host (default: localhost)
- **`CHROMA_PORT`**: ChromaDB port (default: 8000)
- **`LOG_LEVEL`**: Logging level (DEBUG, INFO, WARNING, ERROR)

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your-openai-api-key
LOG_LEVEL=DEBUG
```

### Tool Configuration

All tool configurations are in `pyproject.toml`:

- **Black**: 120 character line length, Python 3.8+ target
- **isort**: Black-compatible profile
- **flake8**: 120 character line length, ignores E203/W503
- **mypy**: Strict type checking with overrides for third-party packages
- **pytest**: Async support, markers for test organization
- **coverage**: Source tracking with HTML reports

## Building and Distribution

```bash
# Build wheel and source distribution
hatch build

# Clean build artifacts
hatch clean

# Publish to PyPI (when ready)
hatch publish
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you've installed the project in editable mode: `pip install -e .`

2. **Missing optional dependencies**: Install the required extras: `pip install -e .[unstructured]`

3. **Test failures**: Some tests require specific optional dependencies or environment variables

4. **Type checking failures**: mypy is configured strictly; some third-party dependencies may need type stubs

### Getting Help

```bash
# Show hatch help
hatch --help

# Show environment information
hatch env show

# Show project dependencies
hatch dep show requirements

# Debug environment issues
hatch shell --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the development workflow above
4. Ensure all tests pass and code is properly formatted
5. Submit a pull request with a clear description

## Additional Resources

- [Hatch Documentation](https://hatch.pypa.io/latest/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Project Issues](https://github.com/juanqui/pdfkb-mcp/issues)
