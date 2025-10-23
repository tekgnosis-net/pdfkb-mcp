# PDF Knowledgebase MCP Server

A powerful Model Context Protocol (MCP) server that transforms your PDF and Markdown document collection into an intelligent, searchable knowledge base. Built for seamless integration with Claude Desktop, VS Code, Continue, and other MCP-enabled AI assistants.

## Description

**pdfkb-mcp** processes your documents using advanced PDF parsing, creates semantic embeddings, and provides sophisticated search capabilities through the Model Context Protocol. Whether you're managing research papers, technical documentation, or business reports, pdfkb-mcp makes your document collection instantly searchable and accessible to your AI assistant.

### Motivation

I built pdfkb-mcp because I needed a way to efficiently index and search through hundreds of semiconductor datasheets and technical documents. Traditional file search wasn't sufficient—I needed semantic understanding, context preservation, and the ability to ask complex questions about technical specifications across multiple documents. This tool has transformed how I work with technical documentation, and I'm sharing it so others can benefit from intelligent document search in their workflows.

## ✨ Features

### 🤖 **Intelligent Document Processing**
- **Multiple PDF Parsers**: PyMuPDF4LLM (fast), Marker (balanced), Docling (tables), MinerU (academic), LLM (complex layouts)
- **Markdown Support**: Native processing of .md and .markdown files with metadata extraction
- **Smart Chunking**: LangChain, semantic, page-based, and unstructured chunking strategies
- **Background Processing**: Non-blocking document processing with intelligent caching

### 🔍 **Advanced Search & AI**
- **Hybrid Search**: Combines semantic similarity with keyword matching (BM25) for superior results
- **AI Reranking**: Qwen3-Reranker models improve search relevance by 15-30%
- **Local & Remote Embeddings**: Privacy-focused local models or high-performance API-based options
- **Document Summarization**: Auto-generates rich metadata with titles, descriptions, and summaries

### 🌐 **Multi-Client & Remote Access**
- **MCP Protocol Support**: Works with Claude Desktop, VS Code, Continue, Cline, and other MCP clients
- **Web Interface**: Modern web UI for document management, search, and analysis
- **HTTP/SSE Transport**: Remote access from multiple clients simultaneously
- **Docker Deployment**: Production-ready containerized deployment

### 🔒 **Privacy & Performance**
- **Local-First Option**: Run completely offline with local embeddings—no API costs, full privacy
- **Quantized Models**: GGUF models use 50-70% less memory with maintained quality
- **Best Practices**: Background processing, health checks, monitoring, and scalability

## 🌐 Web Interface Preview

Once your setup is complete, you'll have access to a modern web interface for document management and search:

![PDF Knowledgebase Web Interface](docs/images/web_documents_list.png)

*The web interface provides document upload, real-time processing status, semantic search, and comprehensive document management capabilities.*

**Key Features:**
- 🔍 **Real-time Search**: Instant semantic and hybrid search
- 📊 **Processing Status**: Live updates on document processing
- 📈 **Document Analytics**: View chunks, metadata, and summaries
- ⚙️ **System Monitoring**: Server performance and resource usage

## 🚀 Quick Start

Get up and running in minutes using Docker/Podman with DeepInfra as your AI provider.

### Prerequisites
- **Container Runtime**: Docker or Podman installed
- **DeepInfra API Key**: [Get your free key](https://deepinfra.com) (recommended for cost-effectiveness)
- **Documents**: A folder with PDF or Markdown files to index

### 1. Set Up Docker Compose

```bash
# Download configuration and create directories
curl -o docker-compose.yml https://raw.githubusercontent.com/tekgnosis-net/pdfkb-mcp/main/docker-compose.sample.yml
mkdir -p ./documents ./cache ./logs

# Edit docker-compose.yml and update:
# 1. Volume path: "/path/to/your/documents:/app/documents:rw"
# 2. API key: PDFKB_OPENAI_API_KEY: "your-deepinfra-api-key-here"
```

### 2. Start the Server

```bash
# Using Podman (recommended)
podman-compose up -d

# Or using Docker
docker compose up -d
```

**Access Points:**
- **Web Interface**: http://localhost:8000
- **MCP Endpoint**: http://localhost:8000/mcp/
- **Health Check**: http://localhost:8000/health

### 3. Configure Your MCP Client

**Claude Desktop** - Add to `claude_desktop_config.json`:
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

**VS Code with Continue** - Add to `.continue/config.json`:
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

### 4. Add Your Documents

- **Web Interface**: Open http://localhost:8000
- **File System**: Copy files to your documents directory — they're automatically detected

### 5. Start Searching

Ask your AI assistant to search your documents:
- *"What register do I need to configure to reset charging in the nPM1300?"*
- *"Is XYZ a clock capable pin according to the nRF54L15 datasheet?"*
- *"What is the conversion formula to interpret temperature as celcius according to the XYZ datashet?"*

The setup includes:
- ✅ **DeepInfra AI**: Cost-effective embeddings, reranking, and document summarization
- ✅ **Hybrid Search**: Semantic + keyword matching
- ✅ **Document Summarization**: Auto-generated metadata (i.e. title, description)
- ✅ **Web Interface**: Document management UI
- ✅ **Persistent Storage**: Documents and cache preserved

## 📚 User Guide

For complete documentation, configuration options, and advanced features:

**👉 [View the Complete User Guide](docs/userguide/index.md)**

The user guide includes:
- **[📦 Installation Options](docs/userguide/installation.md)** - uvx, pip, Docker setup
- **[⚙️ Configuration](docs/userguide/configuration.md)** - Environment variables and settings
- **[🔍 Search Features](docs/userguide/search-features.md)** - Hybrid search, reranking, semantic chunking
- **[🤖 Embeddings](docs/userguide/embeddings.md)** - Local, OpenAI, and HuggingFace options
- **[🔌 MCP Clients](docs/userguide/mcp-clients.md)** - Setup guides for all MCP clients
- **[🐳 Docker Deployment](docs/userguide/docker-deployment.md)** - Production deployment guide
- **[🔧 Troubleshooting](docs/userguide/troubleshooting.md)** - Common issues and performance tuning
- **[🎯 Advanced Features](docs/userguide/advanced.md)** - Document summarization and enterprise features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
