# Quick Start Guide

Get **pdfkb-mcp** running in minutes using Docker/Podman with DeepInfra as your AI provider.

## Prerequisites

- **Container Runtime**: Docker or Podman installed
- **DeepInfra API Key**: Get your free key at [deepinfra.com](https://deepinfra.com)
- **Documents**: A folder with PDF or Markdown files to index

## Step 1: Get Your DeepInfra API Key

1. Visit [deepinfra.com](https://deepinfra.com) and sign up
2. Navigate to your dashboard and create an API key
3. Copy your API key - you'll need it for configuration

💡 **Why DeepInfra?** We recommend DeepInfra because it's extremely cost-effective, supports the latest Qwen3 models, and provides excellent performance for both embeddings and reranking.

## Step 2: Set Up Docker Compose

1. **Download the configuration**:
   ```bash
   curl -o docker-compose.yml https://raw.githubusercontent.com/juanqui/pdfkb-mcp/main/docker-compose.sample.yml
   ```

2. **Create your documents directory**:
   ```bash
   mkdir -p ./documents ./cache ./logs
   ```

3. **Edit the configuration**:
   ```bash
   # Edit docker-compose.yml with your preferred editor
   nano docker-compose.yml
   ```

4. **Update these key settings**:
   ```yaml
   volumes:
     # Change this path to your document collection:
     - "/path/to/your/documents:/app/documents:rw"

   environment:
     # Add your DeepInfra API key:
     PDFKB_OPENAI_API_KEY: "your-deepinfra-api-key-here"
   ```

## Step 3: Start the Server

```bash
# Using Podman (recommended)
podman-compose up -d

# Or using Docker
docker compose up -d
```

The server will start and be accessible at:
- **Web Interface**: http://localhost:8000
- **MCP Endpoint**: http://localhost:8000/mcp/ (for HTTP transport)

## Step 4: Add Your Documents

### Option A: Web Interface
1. Open http://localhost:8000 in your browser
2. Use the upload interface to drag and drop PDF files
3. Watch the processing progress in real-time

### Option B: File System
1. Copy your PDF/Markdown files to your documents directory
2. The server automatically detects and processes new files
3. Check the logs: `podman-compose logs -f` (or `docker compose logs -f`)

## Step 5: Set Up Your MCP Client

### For Claude Desktop

Add this to your Claude Desktop configuration file:

**Location**:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\\Claude\\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Configuration**:
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

### For VS Code with Continue

Add to your `.continue/config.json`:

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

## Step 6: Test Your Setup

1. **Restart your MCP client** (Claude Desktop, VS Code, etc.)
2. **Look for PDF KB tools**: You should see `add_document`, `search_documents`, `list_documents`, `remove_document`, `rescan_documents`
3. **Try searching**: Ask your AI assistant to search your documents

Example queries:
- "Search for information about neural networks in my documents"
- "Find any mentions of API keys or authentication"
- "What documents do I have about machine learning?"

## Verify Everything is Working

### Check Container Status
```bash
# Using Podman
podman-compose ps
podman-compose logs -f

# Using Docker
docker compose ps
docker compose logs -f
```

### Test the Health Endpoint
```bash
curl http://localhost:8000/health
```

Should return: `{"status": "healthy"}`

### Test the Web Interface
Open http://localhost:8000 - you should see the document management interface.

## What's Configured in This Setup

The default configuration includes:

- **🚀 Fast Processing**: PyMuPDF4LLM parser for speed
- **🔍 Advanced Search**: Hybrid search (semantic + keyword matching)
- **📊 Optional Reranking**: Disabled by default; enable with `PDFKB_ENABLE_RERANKER=true`. Defaults to local `Qwen/Qwen3-Reranker-0.6B`. DeepInfra 8B available via `PDFKB_RERANKER_PROVIDER=deepinfra` and `PDFKB_DEEPINFRA_RERANKER_MODEL=Qwen/Qwen3-Reranker-8B`.
- **🧠 Optional Summarization**: Disabled by default; enable with `PDFKB_ENABLE_SUMMARIZER=true` for automatic titles and summaries.
- **🌐 Web + MCP**: Both web interface and MCP endpoints available
- **💾 Persistent Storage**: Documents and cache preserved across restarts

## Next Steps

- **[📄 Add More Documents](web-interface.md#uploading-documents)**: Learn about batch uploading and file monitoring
- **[⚙️ Customize Configuration](configuration.md)**: Tune performance and features
- **[🔧 Troubleshooting](troubleshooting.md)**: Common issues and solutions
- **[🎯 Advanced Features](advanced.md)**: Document summarization and enterprise features

## Quick Commands Reference

```bash
# Start services
podman-compose up -d          # Podman
docker compose up -d          # Docker

# View logs
podman-compose logs -f        # Podman
docker compose logs -f        # Docker

# Stop services
podman-compose down           # Podman
docker compose down           # Docker

# Update and restart
podman-compose pull && podman-compose up -d    # Podman
docker compose pull && docker compose up -d    # Docker

# Check status
curl http://localhost:8000/health
```

🎉 **Congratulations!** Your PDF knowledgebase is now ready to use with your favorite MCP-enabled AI assistant.
