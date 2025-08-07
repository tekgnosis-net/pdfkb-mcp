# PDF Knowledgebase MCP Server

A Model Context Protocol (MCP) server that enables intelligent document search and retrieval from PDF collections. Built for seamless integration with Claude Desktop, Continue, Cline, and other MCP clients, this server provides semantic search capabilities powered by OpenAI embeddings and ChromaDB vector storage.

## Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture Overview](#️-architecture-overview)
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "KNOWLEDGEBASE_PATH": "/Users/yourname/Documents/PDFs"
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "KNOWLEDGEBASE_PATH": "${workspaceFolder}/pdfs"
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
- [`search_documents(query, limit=5, metadata_filter?)`](src/pdfkb/main.py:345) - Semantic search across PDFs
- [`list_documents(metadata_filter?)`](src/pdfkb/main.py:422) - List all documents with metadata
- [`remove_document(document_id)`](src/pdfkb/main.py:488) - Remove document from knowledgebase

**Resources** (Data your client can access):
- `pdf://{document_id}` - Full document content as JSON
- `pdf://{document_id}/page/{page_number}` - Specific page content
- `pdf://list` - List of all documents with metadata

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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDF_PARSER": "pymupdf4llm",
        "PDF_CHUNKER": "langchain",
        "EMBEDDING_MODEL": "text-embedding-3-large"
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDF_PARSER": "pymupdf4llm",
        "CHUNK_SIZE": "800"
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDF_PARSER": "pymupdf4llm",
        "EMBEDDING_BATCH_SIZE": "50"
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDF_PARSER": "mineru",
        "CHUNK_SIZE": "1200"
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDF_PARSER": "pymupdf4llm",
        "DoCLING_TABLE_MODE": "ACCURATE",
        "DOCLING_DO_TABLE_STRUCTURE": "true"
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "PDF_PARSER": "docling",
        "DOCLING_OCR_LANGUAGES": "en,fr,de,es",
        "DOCLING_DO_OCR": "true"
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "OPENROUTER_API_KEY": "sk-or-v1-abc123def456ghi789...",
        "PDF_PARSER": "llm",
        "LLM_MODEL": "anthropic/claude-3.5-sonnet",
        "EMBEDDING_MODEL": "text-embedding-3-large"
      },
      "transport": "stdio"
    }
  }
}
```

### Essential Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | OpenAI API key for embeddings |
| `KNOWLEDGEBASE_PATH` | `./pdfs` | Directory containing PDF files |
| `CACHE_DIR` | `./.cache` | Cache directory for processing |
-| `PDF_PARSER` | `marker` | Parser: `marker`, `pymupdf4llm`, `mineru`, `docling`, `llm` |
+| `PDF_PARSER` | `pymupdf4llm` | Parser: `pymupdf4llm` (default), `marker`, `mineru`, `docling`, `llm` |
-| `CHUNK_SIZE` | `1000` | Target chunk size for LangChain chunker |
-| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
+| `PDF_CHUNKER` | `langchain` | Chunking strategy: `langchain` (default), `unstructured` |
+| `CHUNK_SIZE` | `1000` | Target chunk size for LangChain chunker |
+| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model (use `text-embedding-3-small` for faster processing) |

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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "KNOWLEDGEBASE_PATH": "/Users/yourname/Documents/PDFs",
        "CACHE_DIR": "/Users/yourname/Documents/PDFs/.cache"
      },
      "transport": "stdio",
      "autoRestart": true,
                "EMBEDDING_MODEL": "text-embedding-3-small",
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "KNOWLEDGEBASE_PATH": "${workspaceFolder}/pdfs"
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
        "OPENAI_API_KEY": "sk-proj-abc123def456ghi789...",
        "KNOWLEDGEBASE_PATH": "${workspaceFolder}/pdfs"
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
        "OPENAI_API_KEY": "required",
        "KNOWLEDGEBASE_PATH": "required-absolute-path",
        "PDF_PARSER": "optional-default-marker"
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
        "OPENAI_API_KEY": "sk-key",
        "PDF_PARSER": "pymupdf4llm"
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
        "OPENAI_API_KEY": "sk-key",
        "EMBEDDING_BATCH_SIZE": "25",
        "CHUNK_SIZE": "500"
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
        "OPENAI_API_KEY": "sk-key",
        "PDF_PARSER": "docling",
        "DOCLING_TABLE_MODE": "ACCURATE"
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
        "OPENAI_API_KEY": "sk-key",
        "PDF_PARSER": "mineru",
        "MINERU_LANG": "en",
        "MINERU_METHOD": "auto",
        "MINERU_VRAM": "16"
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
        "OPENAI_API_KEY": "sk-key",
        "OPENROUTER_API_KEY": "sk-or-v1-abc123def456ghi789...",
        "PDF_PARSER": "llm",
        "LLM_MODEL": "google/gemini-2.5-flash-lite",
        "LLM_CONCURRENCY": "5",
        "LLM_DPI": "150"
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
        "OPENAI_API_KEY": "sk-key",
        "PDF_PARSER": "mineru",
        "KNOWLEDGEBASE_PATH": "/Volumes/FastSSD/Documents/PDFs",
        "CACHE_DIR": "/Volumes/FastSSD/Documents/PDFs/.cache",
        "EMBEDDING_BATCH_SIZE": "200",
        "VECTOR_SEARCH_K": "15",
        "FILE_SCAN_INTERVAL": "30"
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
- Changing `PDF_PARSER` → Full reset (parsing + chunking + embeddings)
- Changing `PDF_CHUNKER` → Partial reset (chunking + embeddings)
- Changing `EMBEDDING_MODEL` → Minimal reset (embeddings only)

## 📚 Appendix

### Installation Options

**Primary (Recommended)**:
```bash
uvx pdfkb-mcp
```

**With Specific Parser Dependencies**:
```bash
uvx pdfkb-mcp[marker]     # Marker parser
uvx pdfkb-mcp[mineru]     # MinerU parser
uvx pdfkb-mcp[docling]    # Docling parser
uvx pdfkb-mcp[llm]        # LLM parser
-uvx pdfkb-mcp[langchain]  # LangChain chunker
+uvx pdfkb-mcp[unstructured_chunker]  # Unstructured chunker
```

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
| `OPENAI_API_KEY` | *required* | OpenAI API key for embeddings |
| `OPENROUTER_API_KEY` | *optional* | Required for LLM parser |
| `KNOWLEDGEBASE_PATH` | `./pdfs` | PDF directory path |
| `CACHE_DIR` | `./.cache` | Cache directory |
| `PDF_PARSER` | `pymupdf4llm` | PDF parser selection |
| `PDF_CHUNKER` | `unstructured` | Chunking strategy |
| `CHUNK_SIZE` | `1000` | LangChain chunk size |
| `CHUNK_OVERLAP` | `200` | LangChain chunk overlap |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI model |
| `EMBEDDING_BATCH_SIZE` | `100` | Embedding batch size |
| `VECTOR_SEARCH_K` | `5` | Default search results |
| `FILE_SCAN_INTERVAL` | `60` | File monitoring interval |
| `LOG_LEVEL` | `INFO` | Logging level |

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

**LangChain** (`PDF_CHUNKER=langchain`):
- Header-aware splitting with [`MarkdownHeaderTextSplitter`](src/pdfkb/chunker/chunker_langchain.py)
- Configurable via `CHUNK_SIZE` and `CHUNK_OVERLAP`
- Best for customizable chunking
- Default and installed with base package

**Unstructured** (`PDF_CHUNKER=unstructured`):
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
3. Test connectivity: `curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models`

**Parser Installation Issues**:
1. MinerU: `pip install mineru[all]` and verify `mineru --version`
2. Docling: `pip install docling` for basic, `pip install pdfkb-mcp[docling-complete]` for all features
3. LLM: Requires `OPENROUTER_API_KEY` environment variable

**Performance Optimization**:
1. **Speed**: Use `pymupdf4llm` parser
2. **Memory**: Reduce `EMBEDDING_BATCH_SIZE` and `CHUNK_SIZE`
3. **Quality**: Use `mineru` (GPU) or `docling` (CPU)
4. **Tables**: Use `docling` with `DOCLING_TABLE_MODE=ACCURATE`

For additional support, see implementation details in [`src/pdfkb/main.py`](src/pdfkb/main.py) and [`src/pdfkb/config.py`](src/pdfkb/config.py).
