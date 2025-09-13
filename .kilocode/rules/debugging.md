# debugging.md

This file provides debugging guidelines for Kilo Code when troubleshooting issues in this repository.

## Common Issues and Troubleshooting

### Server Not Appearing in MCP Client

Ensure proper configuration with transport specified:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "transport": "stdio"
    }
  }
}
```

### System Overload When Processing Multiple PDFs

Reduce parallel operations to prevent system stress:
```bash
PDFKB_MAX_PARALLEL_PARSING=1       # Process one PDF at a time
PDFKB_MAX_PARALLEL_EMBEDDING=1     # Embed one document at a time
PDFKB_BACKGROUND_QUEUE_WORKERS=1   # Single background worker
```

### Memory Issues

Configure memory-efficient settings for low-powered systems:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_EMBEDDING_BATCH_SIZE": "25",
        "PDFKB_CHUNK_SIZE": "500"
      },
      "transport": "stdio"
    }
  }
}
```

### Processing Too Slow

Use faster parsers and optimize settings:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_PDF_PARSER": "pymupdf4llm"
      },
      "transport": "stdio"
    }
  }
}
```

### Poor Table Extraction

Switch to table-optimized parsers:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_PDF_PARSER": "docling",
        "PDFKB_DOCLING_TABLE_MODE": "ACCURATE"
      },
      "transport": "stdio"
    }
  }
}
```

## Resource Requirements

### Configuration-Based Resource Usage

| Configuration | RAM Usage | Processing Speed | Best For |
|---------------|-----------|------------------|----------|
| **Speed** | 2-4 GB | Fastest | Large collections |
| **Balanced** | 4-6 GB | Medium | Most users |
| **Quality** | 6-12 GB | Medium-Fast | Accuracy priority |
| **GPU** | 8-16 GB | Very Fast | High-volume processing |

## Parser-Specific Troubleshooting

### MinerU Parser Issues

```bash
# Install MinerU properly
pip install mineru[all]

# Verify installation
mineru --version
```

### Docling Parser Issues

Basic installation:
```bash
pip install docling
```

Complete installation with OCR capabilities:
```bash
pip install pdfkb-mcp[docling-complete]
```

### LLM Parser Issues

Requires OpenRouter API key:
```bash
PDFKB_OPENROUTER_API_KEY=your-key
```

## Performance Optimization Strategies

### 1. Speed Optimization
- Use `pymupdf4llm` parser (fastest, low memory footprint)
- Increase `PDFKB_MAX_PARALLEL_PARSING` if system can handle it
- Use smaller chunk sizes for faster processing

### 2. Memory Optimization
- Reduce `PDFKB_EMBEDDING_BATCH_SIZE` and `PDFKB_CHUNK_SIZE`
- Use pypdfium backend for Docling to reduce memory usage
- Set parallel processing limits to 1 for memory-constrained systems

### 3. Quality Optimization
- Use `mineru` with GPU for maximum processing speed
- Use `marker` parser for balanced quality and performance
- Increase chunk sizes for better context preservation

### 4. Table Processing Optimization
- Use `docling` with `PDFKB_DOCLING_TABLE_MODE=ACCURATE`
- Use `marker` with LLM mode for enhanced table merging

### 5. Batch Processing Optimization
- Use `marker` parser on H100 (~25 pages/s) for high-volume processing
- Use `mineru` with sglang acceleration for very fast parsing

## Development Environment Issues

### Import Errors
Ensure you've installed the project in editable mode:
```bash
pip install -e .
```

### Missing Optional Dependencies
Install the required extras:
```bash
pip install -e .[unstructured]
pip install -e .[docling]
pip install -e .[mineru]
pip install -e .[marker]
```

### Type Checking Failures
mypy is configured strictly; some third-party dependencies may need type stubs:
```bash
# Run mypy with configuration from pyproject.toml
hatch run mypy src
```

## Testing Troubleshooting

Some tests require specific optional dependencies or environment variables:
```bash
# Skip slow tests during development
hatch run test -m "not slow"

# Run specific test markers
hatch run test -m unit
hatch run test -m integration
hatch run test -m performance
```

## API Key Troubleshooting

### OpenAI API Key Issues
1. Verify key format starts with `sk-`
2. Check account has sufficient credits
3. Test connectivity:
   ```bash
   curl -H "Authorization: Bearer $PDFKB_OPENAI_API_KEY" https://api.openai.com/v1/models
   ```

### HuggingFace Token Issues
1. Get token from https://huggingface.co/settings/tokens
2. Set environment variable:
   ```bash
   HF_TOKEN=your-token-here
   ```

### Custom API Endpoint Issues
1. Verify base URL format (should end with `/v1/`)
2. Test endpoint connectivity:
   ```bash
   curl -H "Authorization: Bearer $PDFKB_OPENAI_API_KEY" $PDFKB_OPENAI_API_BASE/models
   ```

## Debugging Commands

### Hatch Debugging
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

### Server Debugging
```bash
# Run server with debug logging
PDFKB_LOG_LEVEL=DEBUG hatch run pdfkb-mcp

# Check for detailed error output
hatch run python -m pdfkb.main --log-level DEBUG
```

## Monitoring and Metrics

The server provides internal monitoring capabilities:
- File processing status via background queue
- Memory usage tracking
- Performance metrics for parsing and embedding
- Cache hit/miss statistics

For enhanced web metrics (requires `web` extra):
```bash
pip install -e ".[web]"
```

This provides psutil integration for detailed system monitoring.

## Error Handling Guidelines

### Graceful Failures
- The server logs warnings and falls back to default components when optional parsers/chunkers aren't installed
- Cache invalidation occurs automatically when configuration changes
- Background queue handles processing errors without blocking the main server

### Common Error Patterns
- **Configuration Validation**: Check environment variables and their values
- **Dependency Missing**: Install appropriate optional dependency groups
- **Model Loading**: Verify model cache directories and disk space
- **API Connectivity**: Test API keys and endpoints with curl commands
- **File Permissions**: Ensure proper read/write access to knowledgebase and cache directories

## Logs and Diagnostics

### Key Log Locations
- Server startup logs show selected components and configuration
- Processing logs include parsing, chunking, and embedding status
- Error logs contain detailed stack traces for debugging

### Log Levels
- **DEBUG**: Detailed internal operations
- **INFO**: Standard server operations (default)
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Critical failures that prevent operation

Configure logging:
```bash
PDFKB_LOG_LEVEL=DEBUG  # For detailed debugging
```

## First-Run Diagnostics

On first run, the server:
1. Initializes caches and vector store
2. Logs selected components:
   - Parser: PyMuPDF4LLM (default)
   - Chunker: LangChain (default)
   - Embedding Model: text-embedding-3-large (default)
3. Shows cache initialization status
4. Displays configuration validation results

## Environment Variables for Debugging

### Essential Debug Variables
- `PDFKB_LOG_LEVEL`: Set to DEBUG for detailed logging
- `PDFKB_CACHE_DIR`: Customize cache location for testing
- `PDFKB_MODEL_CACHE_DIR`: Control model download location

### Development Debug Variables
- `PDFKB_THREAD_POOL_SIZE`: Control concurrency for debugging
- `PDFKB_FILE_SCAN_INTERVAL`: Adjust file monitoring frequency
- `PDFKB_VECTOR_SEARCH_K`: Modify search result count for testing

## Getting Additional Help

### Documentation Resources
- See implementation details in [`src/pdfkb/main.py`](src/pdfkb/main.py) and [`src/pdfkb/config.py`](src/pdfkb/config.py)
- Check parser-specific implementations in [`src/pdfkb/parsers/`](src/pdfkb/parsers/)
- Review chunker implementations in [`src/pdfkb/chunker/`](src/pdfkb/chunker/)

### Support Channels
- Project repository: https://github.com/juanqui/pdfkb-mcp
- Issue tracker: https://github.com/juanqui/pdfkb-mcp/issues
- Model Context Protocol documentation: https://modelcontextprotocol.io/
