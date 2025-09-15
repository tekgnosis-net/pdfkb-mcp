# MCP Client Setup Guide

This guide covers setting up **pdfkb-mcp** with different MCP clients.

## Overview

**pdfkb-mcp** works with any MCP-compatible client through different transport modes:

- **stdio**: Direct process communication (most common)
- **http**: HTTP-based transport for remote access
- **sse**: Server-Sent Events for real-time communication

## Claude Desktop

### Configuration

**Location**:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\\Claude\\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### Option 1: Local Installation (stdio)

```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_KNOWLEDGEBASE_PATH": "/Users/yourname/Documents",
        "PDFKB_EMBEDDING_PROVIDER": "local"
      },
      "transport": "stdio",
      "autoRestart": true
    }
  }
}
```

### Option 2: Remote Server (HTTP)

For connecting to a remote pdfkb-mcp server:

```json
{
  "mcpServers": {
    "pdfkb": {
      "transport": "http",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

### Option 3: With API Keys

```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp[hybrid]"],
      "env": {
        "PDFKB_EMBEDDING_PROVIDER": "openai",
        "PDFKB_OPENAI_API_KEY": "sk-proj-your-key-here",
        "PDFKB_OPENAI_API_BASE": "https://api.deepinfra.com/v1",
        "PDFKB_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
        "PDFKB_KNOWLEDGEBASE_PATH": "/Users/yourname/Documents",
        "PDFKB_ENABLE_HYBRID_SEARCH": "true"
      },
      "transport": "stdio",
      "autoRestart": true
    }
  }
}
```

## VS Code with Continue Extension

### Configuration

**Location**: `.continue/config.json` in your project or home directory

```json
{
  "models": [
    {
      "title": "Claude 3.5 Sonnet",
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "apiKey": "your-anthropic-key"
    }
  ],
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_KNOWLEDGEBASE_PATH": "${workspaceFolder}/documents",
        "PDFKB_EMBEDDING_PROVIDER": "local"
      },
      "transport": "stdio"
    }
  }
}
```

### Using Workspace Variables

Continue supports workspace variables:

```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_KNOWLEDGEBASE_PATH": "${workspaceFolder}/pdfs",
        "PDFKB_CACHE_DIR": "${workspaceFolder}/.cache"
      },
      "transport": "stdio"
    }
  }
}
```

## VS Code with Native MCP Support

**Configuration**: `.vscode/mcp.json` in your workspace

### SSE Transport

```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp", "--transport", "sse"],
      "env": {
        "PDFKB_KNOWLEDGEBASE_PATH": "/path/to/documents"
      },
      "transport": "sse",
      "autoRestart": true
    }
  }
}
```

## Cline (HTTP Transport)

For Cline and similar clients that prefer HTTP transport:

```json
{
  "mcpServers": {
    "pdfkb": {
      "transport": "http",
      "url": "http://localhost:8000/mcp/",
      "headers": {
        "Authorization": "Bearer optional-api-key"
      }
    }
  }
}
```

## Generic MCP Client

### Template Configuration

```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp"],
      "env": {
        "PDFKB_KNOWLEDGEBASE_PATH": "/absolute/path/to/documents",
        "PDFKB_EMBEDDING_PROVIDER": "local",
        "PDFKB_LOG_LEVEL": "INFO"
      },
      "transport": "stdio",
      "autoRestart": true,
      "timeout": 30000
    }
  }
}
```

## Configuration Examples by Use Case

### Academic Research

```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp[semantic,marker]"],
      "env": {
        "PDFKB_KNOWLEDGEBASE_PATH": "/Users/researcher/Papers",
        "PDFKB_PDF_PARSER": "marker",
        "PDFKB_PDF_CHUNKER": "semantic",
        "PDFKB_EMBEDDING_PROVIDER": "local",
        "PDFKB_LOCAL_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-4B",
        "PDFKB_ENABLE_HYBRID_SEARCH": "true",
        "PDFKB_ENABLE_RERANKER": "true"
      },
      "transport": "stdio",
      "autoRestart": true
    }
  }
}
```

### Business Documents

```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp[hybrid]"],
      "env": {
        "PDFKB_KNOWLEDGEBASE_PATH": "/Users/analyst/BusinessDocs",
        "PDFKB_PDF_PARSER": "docling",
        "PDFKB_EMBEDDING_PROVIDER": "openai",
        "PDFKB_OPENAI_API_KEY": "sk-proj-your-key",
        "PDFKB_OPENAI_API_BASE": "https://api.deepinfra.com/v1",
        "PDFKB_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
        "PDFKB_ENABLE_HYBRID_SEARCH": "true"
      },
      "transport": "stdio",
      "autoRestart": true
    }
  }
}
```

### Privacy-Focused Setup

```json
{
  "mcpServers": {
    "pdfkb": {
      "command": "uvx",
      "args": ["pdfkb-mcp[hybrid]"],
      "env": {
        "PDFKB_KNOWLEDGEBASE_PATH": "/Users/privacy/SecureDocs",
        "PDFKB_EMBEDDING_PROVIDER": "local",
        "PDFKB_LOCAL_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "PDFKB_GGUF_QUANTIZATION": "Q6_K",
        "PDFKB_ENABLE_HYBRID_SEARCH": "true",
        "PDFKB_ENABLE_RERANKER": "true",
        "PDFKB_RERANKER_MODEL": "Mungert/Qwen3-Reranker-0.6B-GGUF"
      },
      "transport": "stdio",
      "autoRestart": true
    }
  }
}
```

## Remote Server Setup

### Start Remote Server

```bash
# HTTP transport
PDFKB_WEB_ENABLE=true pdfkb-mcp --transport http --server-port 8000

# SSE transport
PDFKB_WEB_ENABLE=true pdfkb-mcp --transport sse --server-port 8000
```

### Client Configuration for Remote Server

```json
{
  "mcpServers": {
    "pdfkb": {
      "transport": "http",
      "url": "http://your-server:8000/mcp/"
    }
  }
}
```

## Troubleshooting Client Setup

### Common Issues

**1. Server not appearing in client**
- ✅ Restart your MCP client completely
- ✅ Check configuration file syntax (valid JSON)
- ✅ Verify file paths are absolute paths
- ✅ Check logs in client for error messages

**2. Connection refused errors**
- ✅ Ensure pdfkb-mcp is properly installed: `uvx pdfkb-mcp --help`
- ✅ Check if the server starts: run command manually
- ✅ Verify network connectivity for remote servers

**3. Permission errors**
- ✅ Check document folder permissions
- ✅ Use absolute paths in configuration
- ✅ Ensure cache directory is writable

**4. API key errors**
- ✅ Verify API key format and validity
- ✅ Check API provider status
- ✅ Test with curl: `curl -H "Authorization: Bearer $API_KEY" https://api.provider.com/v1/models`

### Debugging

**Enable debug logging**:
```json
{
  "env": {
    "PDFKB_LOG_LEVEL": "DEBUG"
  }
}
```

**Test server manually**:
```bash
# Run server directly to see logs
PDFKB_LOG_LEVEL=DEBUG uvx pdfkb-mcp
```

**Check available tools**:
After connecting, you should see these MCP tools:
- `add_document` - Add PDFs to the knowledge base
- `search_documents` - Search across all documents
- `list_documents` - List all documents with metadata
- `remove_document` - Remove documents from the knowledge base
- `rescan_documents` - Rescan documents directory

## Next Steps

After setting up your MCP client:

1. **[🚀 Test Your Setup](quick-start.md#step-6-test-your-setup)**: Verify everything works
2. **[📄 Add Documents](web-interface.md#uploading-documents)**: Start building your knowledge base
3. **[🔍 Search Features](search-features.md)**: Learn about advanced search capabilities
4. **[🔧 Troubleshooting](troubleshooting.md)**: Common issues and solutions

## Platform-Specific Notes

### macOS
- Use `~/Library/Application Support/Claude/` for Claude Desktop config
- Apple Silicon automatically uses Metal acceleration for local embeddings

### Windows
- Use `%APPDATA%\\Claude\\` for Claude Desktop config
- Consider WSL2 for better compatibility

### Linux
- Use `~/.config/Claude/` for Claude Desktop config
- NVIDIA GPUs automatically detected for acceleration
