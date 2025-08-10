# PDF Knowledgebase MCP Server

A Model Context Protocol (MCP) server that enables intelligent document search and retrieval from PDF collections. Built for seamless integration with Claude Desktop, Continue, Cline, and other MCP clients, this server provides advanced search capabilities powered by OpenAI embeddings and ChromaDB vector storage.

**🆕 NEW Features:**
- **Hybrid Search**: Combines semantic similarity with keyword matching (BM25) for superior search quality
- **Web Interface**: Modern web UI for document management and search alongside the traditional MCP protocol

## Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🌐 Web Interface](#-web-interface)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [🔍 Hybrid Search](#-hybrid-search)
- [🎯 Parser Selection Guide](#-parser-selection-guide)
- [⚙️ Configuration](#️-configuration)
- [🖥️ MCP Client Setup](#️-mcp-client-setup)
- [📊 Performance & Troubleshooting](#-performance--troubleshooting)
- [🔧 Advanced Configuration](#-advanced-configuration)
- [📚 Appendix](#-appendix)

## 🚀 Quick Start

### Step 1: Install the Server

```bash
uvx pdfkb-mcp
```

### Step 2: Configure Your MCP Client

**Claude Desktop** (Most Common):

*Configuration file locations:*
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_KNOWLEDGEBASE_PATH": "/Users/yourname/Documents/PDFs"
      },
      "transport": "stdio",
      "autoRestart": true
    }
  }
}
```

**VS Code (Native MCP)** - Create `.vscode/mcp.json` in workspace:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_KNOWLEDGEBASE_PATH": "${workspaceFolder}/pdfs"
      },
      "transport": "stdio"
    }
  }
}
```

### Step 3: Verify Installation

1. **Restart your MCP client** completely
2. **Check for PDF KB tools**: Look for `add_document`, `search_documents`, `list_documents`, `remove_document`
3. **Test functionality**: Try adding a PDF and searching for content

## 🌐 Web Interface

The PDF Knowledgebase now includes a modern web interface for easy document management and search. You can run the server in two different modes:

### Server Modes

**1. MCP Only** (Traditional Mode):
```bash
pdfkb-mcp
```
- Runs only the MCP server for integration with Claude Desktop, VS Code, etc.
- Most resource-efficient option
- Web interface disabled by default

**2. Integrated** (Both MCP + Web):
```bash
PDFKB_ENABLE_WEB=true pdfkb-mcp
```
- Runs both MCP server AND web interface concurrently
- Shared document processing and storage
- Best of both worlds: API integration + web UI
- Web interface available at http://localhost:8080

### Web Interface Features

- **📄 Document Upload**: Drag & drop PDF files or upload via file picker
- **🔍 Semantic Search**: Powerful vector-based search with real-time results
- **📊 Document Management**: List, preview, and manage your PDF collection
- **📈 Real-time Status**: Live processing updates via WebSocket connections
- **🎯 Chunk Explorer**: View and navigate document chunks for detailed analysis
- **⚙️ System Metrics**: Monitor server performance and resource usage

### Quick Web Setup

1. **Install and run**:
   ```bash
   uvx pdfkb-mcp                    # Install if needed
   PDFKB_ENABLE_WEB=true pdfkb-mcp  # Start integrated server
   ```

2. **Open your browser**: http://localhost:8080

3. **Configure environment** (create `.env` file):
   ```bash
   PDFKB_OPENAI_API_KEY=sk-proj-abc123def456ghi789...
   PDFKB_KNOWLEDGEBASE_PATH=/path/to/your/pdfs
   PDFKB_WEB_PORT=8080
   PDFKB_WEB_HOST=localhost
   PDFKB_ENABLE_WEB=true
   ```

### Web Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PDFKB_ENABLE_WEB` | `false` | Enable/disable web interface |
| `PDFKB_WEB_PORT` | `8080` | Web server port |
| `PDFKB_WEB_HOST` | `localhost` | Web server host |
| `PDFKB_WEB_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS allowed origins |

### Command Line Options

The server supports command line arguments:

```bash
# Customize web server port (when web interface is enabled)
PDFKB_ENABLE_WEB=true pdfkb-mcp --port 9000

# Use custom configuration file
pdfkb-mcp --config myconfig.env

# Change log level
pdfkb-mcp --log-level DEBUG

# Enable web interface via command line
pdfkb-mcp --enable-web
```

### API Documentation

When running with web interface enabled, comprehensive API documentation is available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## 🏗️ Architecture Overview

### MCP Integration

```raw
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │    │   MCP Client     │    │   MCP Client    │
│ (Claude Desktop)│    │(VS Code/Continue)|    │   (Other)       │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    Model Context        │
                    │    Protocol (MCP)       │
                    │    Standard Layer       │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼───────────────────────┐
          │                      │                       │
┌─────────┴───────┐    ┌─────────┴────────┐    ┌─────────┴───────┐
│ PDF KB Server   │    │  Other MCP       │    │  Other MCP      │
│ (This Server)   │    │  Server          │    │  Server         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Available Tools & Resources

**Tools** (Actions your client can perform):
- [`add_document(path, metadata?)`](src/pdfkb/main.py:278) - Add PDF to knowledgebase
- [`search_documents(query, limit=5, metadata_filter?, search_type?)`](src/pdfkb/main.py:345) - Hybrid search across PDFs (semantic + keyword matching)
- [`list_documents(metadata_filter?)`](src/pdfkb/main.py:422) - List all documents with metadata
- [`remove_document(document_id)`](src/pdfkb/main.py:488) - Remove document from knowledgebase

**Resources** (Data your client can access):
- `pdf://{document_id}` - Full document content as JSON
- `pdf://{document_id}/page/{page_number}` - Specific page content
- `pdf://list` - List of all documents with metadata

## 🔍 Hybrid Search

The server now supports **Hybrid Search**, which combines the strengths of semantic similarity search (vector embeddings) with traditional keyword matching (BM25) for improved search quality.

### How It Works

1. **Dual Indexing**: Documents are indexed in both a vector database (ChromaDB) and a full-text search index (Whoosh)
2. **Parallel Search**: Queries execute both semantic and keyword searches simultaneously
3. **Reciprocal Rank Fusion (RRF)**: Results are intelligently merged using RRF algorithm for optimal ranking

### Benefits

- **Better Recall**: Finds documents that match exact keywords even if semantically different
- **Improved Precision**: Combines conceptual understanding with keyword relevance
- **Technical Terms**: Excellent for technical documentation, code references, and domain-specific terminology
- **Balanced Results**: Configurable weights let you adjust the balance between semantic and keyword matching

### Configuration

Enable hybrid search by setting:
```bash
PDFKB_ENABLE_HYBRID_SEARCH=true  # Enable hybrid search (default: true)
PDFKB_HYBRID_VECTOR_WEIGHT=0.6   # Weight for semantic search (default: 0.6)
PDFKB_HYBRID_TEXT_WEIGHT=0.4     # Weight for keyword search (default: 0.4)
PDFKB_RRF_K=60                   # RRF constant (default: 60)
```

### Installation

To use hybrid search, install with the optional dependency:
```bash
pip install "pdfkb-mcp[hybrid]"
```

Or if using uvx, it's included by default when hybrid search is enabled.

## 🎯 Parser Selection Guide

### Decision Tree

```
Document Type & Priority?
├── 🏃 Speed Priority → PyMuPDF4LLM (fastest processing, low memory)
├── 📚 Academic Papers → MinerU (fast with GPU, excellent formulas)
├── 📊 Business Reports → Docling (medium speed, best tables)
├── ⚖️ Balanced Quality → Marker (medium speed, good structure)
└── 🎯 Maximum Accuracy → LLM (slow, vision-based API calls)
```

### Performance Comparison

| Parser | Processing Speed | Memory | Text Quality | Table Quality | Best For |
|--------|------------------|--------|--------------|---------------|----------|
| **PyMuPDF4LLM** | **Fastest** | Low | Good | Basic | Speed priority |
| **MinerU** | Fast (with GPU) | High | Excellent | Excellent | Scientific papers |
| **Docling** | Medium | Medium | Excellent | **Excellent** | Business documents |
| **Marker** | Medium | Medium | Excellent | Good | **Balanced** |
| **LLM** | Slow | Low | Excellent | Excellent | Maximum accuracy |

*Benchmarks from research studies and technical reports*

## ⚙️ Configuration

### Tier 1: Basic Configurations (80% of users)

**Default (Recommended)**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_PDF_PARSER": "pymupdf4llm",
        "PDFKB_PDF_CHUNKER": "langchain",
        "PDFKB_EMBEDDING_MODEL": "text-embedding-3-large"
      },
      "transport": "stdio"
    }
  }
}
```

**Speed Optimized**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_PDF_PARSER": "pymupdf4llm",
        "PDFKB_CHUNK_SIZE": "800"
      },
      "transport": "stdio"
    }
  }
}
```

**Memory Efficient**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_PDF_PARSER": "pymupdf4llm",
        "PDFKB_EMBEDDING_BATCH_SIZE": "50"
      },
      "transport": "stdio"
    }
  }
}
```

### Tier 2: Use Case Specific (15% of users)

**Academic Papers**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_PDF_PARSER": "mineru",
        "PDFKB_CHUNK_SIZE": "1200"
      },
      "transport": "stdio"
    }
  }
}
```

**Business Documents**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_PDF_PARSER": "pymupdf4llm",
        "PDFKB_DOCLING_TABLE_MODE": "ACCURATE",
        "PDFKB_DOCLING_DO_TABLE_STRUCTURE": "true"
      },
      "transport": "stdio"
    }
  }
}
```

**Multi-language Documents**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_PDF_PARSER": "docling",
        "PDFKB_DOCLING_OCR_LANGUAGES": "en,fr,de,es",
        "PDFKB_DOCLING_DO_OCR": "true"
      },
      "transport": "stdio"
    }
  }
}
```

**Hybrid Search (NEW - Improved Search Quality)**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_ENABLE_HYBRID_SEARCH": "true",
        "PDFKB_HYBRID_VECTOR_WEIGHT": "0.6",
        "PDFKB_HYBRID_TEXT_WEIGHT": "0.4"
      },
      "transport": "stdio"
    }
  }
}
```

**Maximum Quality**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_OPENROUTER_API_KEY": "sk-or-v1-abc123def456ghi789...",
        "PDFKB_PDF_PARSER": "llm",
        "PDFKB_LLM_MODEL": "anthropic/claude-3.5-sonnet",
        "PDFKB_EMBEDDING_MODEL": "text-embedding-3-large"
      },
      "transport": "stdio"
    }
  }
}
```

### Essential Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PDFKB_OPENAI_API_KEY` | *required* | OpenAI API key for embeddings |
| `PDFKB_KNOWLEDGEBASE_PATH` | `./pdfs` | Directory containing PDF files |
| `PDFKB_CACHE_DIR` | `./.cache` | Cache directory for processing |
| `PDFKB_PDF_PARSER` | `pymupdf4llm` | Parser: `pymupdf4llm` (default), `marker`, `mineru`, `docling`, `llm` |
| `PDFKB_PDF_CHUNKER` | `langchain` | Chunking strategy: `langchain` (default), `unstructured` |
| `PDFKB_CHUNK_SIZE` | `1000` | Target chunk size for LangChain chunker |
| `PDFKB_ENABLE_WEB` | `false` | Enable/disable web interface |
| `PDFKB_WEB_PORT` | `8080` | Web server port |
| `PDFKB_WEB_HOST` | `localhost` | Web server host |
| `PDFKB_WEB_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS allowed origins (comma-separated) |
| `PDFKB_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model (use `text-embedding-3-small` for faster processing) |
| `PDFKB_ENABLE_HYBRID_SEARCH` | `true` | Enable hybrid search combining semantic and keyword matching |
| `PDFKB_HYBRID_VECTOR_WEIGHT` | `0.6` | Weight for semantic search (0-1, must sum to 1 with text weight) |
| `PDFKB_HYBRID_TEXT_WEIGHT` | `0.4` | Weight for keyword/BM25 search (0-1, must sum to 1 with vector weight) |
| `PDFKB_RRF_K` | `60` | Reciprocal Rank Fusion constant (higher = less emphasis on rank differences) |

## 🖥️ MCP Client Setup

### Claude Desktop

**Configuration File Location**:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Configuration**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_KNOWLEDGEBASE_PATH": "/Users/yourname/Documents/PDFs",
        "PDFKB_CACHE_DIR": "/Users/yourname/Documents/PDFs/.cache"
      },
      "transport": "stdio",
      "autoRestart": true,
                "PDFKB_EMBEDDING_MODEL": "text-embedding-3-small",
    }
  }
}
```

**Verification**:
1. Restart Claude Desktop completely
2. Look for PDF KB tools in the interface
3. Test with "Add a document" or "Search documents"

### VS Code with Native MCP Support

**Configuration** (`.vscode/mcp.json` in workspace):
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_KNOWLEDGEBASE_PATH": "${workspaceFolder}/pdfs"
      },
      "transport": "stdio"
    }
  }
}
```

**Verification**:
1. Reload VS Code window
2. Check VS Code's MCP server status in Command Palette
3. Use MCP tools in Copilot Chat

### VS Code with Continue Extension

**Configuration** (`.continue/config.json`):
```json
{
  "models": [...],
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDFKB_KNOWLEDGEBASE_PATH": "${workspaceFolder}/pdfs"
      },
      "transport": "stdio"
    }
  }
}
```

**Verification**:
1. Reload VS Code window
2. Check Continue panel for server connection
3. Use `@pdfkb` in Continue chat

### Generic MCP Client

**Standard Configuration Template**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "required",
        "PDFKB_KNOWLEDGEBASE_PATH": "required-absolute-path",
        "PDFKB_PDF_PARSER": "optional-default-pymupdf4llm"
      },
      "transport": "stdio",
      "autoRestart": true,
      "timeout": 30000
    }
  }
}
```

## 📊 Performance & Troubleshooting

### Common Issues

**Server not appearing in MCP client**:
```json
// ❌ Wrong: Missing transport
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"]
    }
  }
}

// ✅ Correct: Include transport and restart client
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

**Processing too slow**:
```json
// Switch to faster parser
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-key",
        "PDFKB_PDF_PARSER": "pymupdf4llm"
      },
      "transport": "stdio"
    }
  }
}
```

**Memory issues**:
```json
// Reduce memory usage
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-key",
        "PDFKB_EMBEDDING_BATCH_SIZE": "25",
        "PDFKB_CHUNK_SIZE": "500"
      },
      "transport": "stdio"
    }
  }
}
```

**Poor table extraction**:
```json
// Use table-optimized parser
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-key",
        "PDFKB_PDF_PARSER": "docling",
        "PDFKB_DOCLING_TABLE_MODE": "ACCURATE"
      },
      "transport": "stdio"
    }
  }
}
```

### Resource Requirements

| Configuration | RAM Usage | Processing Speed | Best For |
|---------------|-----------|------------------|----------|
| **Speed** | 2-4 GB | Fastest | Large collections |
| **Balanced** | 4-6 GB | Medium | Most users |
| **Quality** | 6-12 GB | Medium-Fast | Accuracy priority |
| **GPU** | 8-16 GB | Very Fast | High-volume processing |

## 🔧 Advanced Configuration

### Parser-Specific Options

**MinerU Configuration**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-key",
        "PDFKB_PDF_PARSER": "mineru",
        "PDFKB_MINERU_LANG": "en",
        "PDFKB_MINERU_METHOD": "auto",
        "PDFKB_MINERU_VRAM": "16"
      },
      "transport": "stdio"
    }
  }
}
```

**LLM Parser Configuration**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-key",
        "PDFKB_OPENROUTER_API_KEY": "sk-or-v1-abc123def456ghi789...",
        "PDFKB_PDF_PARSER": "llm",
        "PDFKB_LLM_MODEL": "google/gemini-2.5-flash-lite",
        "PDFKB_LLM_CONCURRENCY": "5",
        "PDFKB_LLM_DPI": "150"
      },
      "transport": "stdio"
    }
  }
}
```

### Performance Tuning

**High-Performance Setup**:
```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_OPENAI_API_KEY": "sk-key",
        "PDFKB_PDF_PARSER": "mineru",
        "PDFKB_KNOWLEDGEBASE_PATH": "/Volumes/FastSSD/Documents/PDFs",
        "PDFKB_CACHE_DIR": "/Volumes/FastSSD/Documents/PDFs/.cache",
        "PDFKB_EMBEDDING_BATCH_SIZE": "200",
        "PDFKB_VECTOR_SEARCH_K": "15",
        "PDFKB_FILE_SCAN_INTERVAL": "30"
      },
      "transport": "stdio"
    }
  }
}
```

### Intelligent Caching

The server uses multi-stage caching:
- **Parsing Cache**: Stores converted markdown ([`src/pdfkb/intelligent_cache.py:139`](src/pdfkb/intelligent_cache.py:139))
- **Chunking Cache**: Stores processed chunks
- **Vector Cache**: ChromaDB embeddings storage

**Cache Invalidation Rules**:
- Changing `PDFKB_PDF_PARSER` → Full reset (parsing + chunking + embeddings)
- Changing `PDFKB_PDF_CHUNKER` → Partial reset (chunking + embeddings)
- Changing `PDFKB_EMBEDDING_MODEL` → Minimal reset (embeddings only)

## 📚 Appendix

### Installation Options

**Primary (Recommended)**:
```bash
uvx pdfkb-mcp
**Web Interface Included**: All installation methods include the web interface. Use these commands:
- `pdfkb-mcp` - MCP server only (web disabled by default)
- `PDFKB_ENABLE_WEB=true pdfkb-mcp` - Integrated MCP + Web server
```

**With Specific Parser Dependencies**:
```bash
uvx pdfkb-mcp[marker]     # Marker parser
uvx pdfkb-mcp[mineru]     # MinerU parser
uvx pdfkb-mcp[docling]    # Docling parser
uvx pdfkb-mcp[llm]        # LLM parser
-uvx pdfkb-mcp[langchain]  # LangChain chunker
uvx pdfkb-mcp[web]        # Enhanced web features (psutil for metrics)
+uvx pdfkb-mcp[unstructured_chunker]  # Unstructured chunker
```

pip install "pdfkb-mcp[web]"               # Enhanced web features
Or via pip/pipx:
```bash
pip install "pdfkb-mcp[marker]"            # Marker parser
pip install "pdfkb-mcp[docling-complete]"  # Docling with OCR and full features
```

**Development Installation**:
```bash
git clone https://github.com/juanqui/pdfkb-mcp.git
cd pdfkb-mcp
pip install -e ".[dev]"
```

### Complete Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PDFKB_OPENAI_API_KEY` | *required* | OpenAI API key for embeddings |
| `PDFKB_OPENROUTER_API_KEY` | *optional* | Required for LLM parser |
| `PDFKB_KNOWLEDGEBASE_PATH` | `./pdfs` | PDF directory path |
| `PDFKB_CACHE_DIR` | `./.cache` | Cache directory |
| `PDFKB_PDF_PARSER` | `pymupdf4llm` | PDF parser selection |
| `PDFKB_PDF_CHUNKER` | `langchain` | Chunking strategy |
| `PDFKB_CHUNK_SIZE` | `1000` | LangChain chunk size |
| `PDFKB_CHUNK_OVERLAP` | `200` | LangChain chunk overlap |
| `PDFKB_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI model |
| `PDFKB_EMBEDDING_BATCH_SIZE` | `100` | Embedding batch size |
| `PDFKB_VECTOR_SEARCH_K` | `5` | Default search results |
| `PDFKB_FILE_SCAN_INTERVAL` | `60` | File monitoring interval |
| `PDFKB_LOG_LEVEL` | `INFO` | Logging level |
| `PDFKB_ENABLE_WEB` | `false` | Enable/disable web interface |
| `PDFKB_WEB_PORT` | `8080` | Web server port |
| `PDFKB_WEB_HOST` | `localhost` | Web server host |
| `PDFKB_WEB_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS allowed origins (comma-separated) |

### Parser Comparison Details

| Feature | PyMuPDF4LLM | Marker | MinerU | Docling | LLM |
|---------|-------------|--------|--------|---------|-----|
| **Speed** | Fastest | Medium | Fast (GPU) | Medium | Slowest |
| **Memory** | Lowest | Medium | High | Medium | Lowest |
| **Tables** | Basic | Good | Excellent | **Excellent** | Excellent |
| **Formulas** | Basic | Good | **Excellent** | Good | Excellent |
| **Images** | Basic | Good | Good | **Excellent** | **Excellent** |
| **Setup** | Simple | Simple | Moderate | Simple | Simple |
| **Cost** | Free | Free | Free | Free | API costs |

### Chunking Strategies

**LangChain** (`PDFKB_PDF_CHUNKER=langchain`):
- Header-aware splitting with [`MarkdownHeaderTextSplitter`](src/pdfkb/chunker/chunker_langchain.py)
- Configurable via `PDFKB_CHUNK_SIZE` and `PDFKB_CHUNK_OVERLAP`
- Best for customizable chunking
- Default and installed with base package

**Unstructured** (`PDFKB_PDF_CHUNKER=unstructured`):
- Intelligent semantic chunking with [`unstructured`](src/pdfkb/chunker/chunker_unstructured.py) library
- Zero configuration required
- Install extra: `pip install "pdfkb-mcp[unstructured_chunker]"` to enable
- Best for document structure awareness

### First-run notes

- On the first run, the server initializes caches and vector store and logs selected components:
  - Parser: PyMuPDF4LLM (default)
  - Chunker: LangChain (default)
  - Embedding Model: text-embedding-3-large (default)
- If you select a parser/chunker that isn’t installed, the server logs a warning with the exact install command and falls back to the default components instead of exiting.

### Troubleshooting Guide

**API Key Issues**:
1. Verify key format starts with `sk-`
2. Check account has sufficient credits
3. Test connectivity: `curl -H "Authorization: Bearer $PDFKB_OPENAI_API_KEY" https://api.openai.com/v1/models`

**Parser Installation Issues**:
1. MinerU: `pip install mineru[all]` and verify `mineru --version`
2. Docling: `pip install docling` for basic, `pip install pdfkb-mcp[docling-complete]` for all features
3. LLM: Requires `PDFKB_OPENROUTER_API_KEY` environment variable

**Performance Optimization**:
1. **Speed**: Use `pymupdf4llm` parser
2. **Memory**: Reduce `PDFKB_EMBEDDING_BATCH_SIZE` and `PDFKB_CHUNK_SIZE`
3. **Quality**: Use `mineru` (GPU) or `docling` (CPU)
4. **Tables**: Use `docling` with `PDFKB_DOCLING_TABLE_MODE=ACCURATE`

For additional support, see implementation details in [`src/pdfkb/main.py`](src/pdfkb/main.py) and [`src/pdfkb/config.py`](src/pdfkb/config.py).
