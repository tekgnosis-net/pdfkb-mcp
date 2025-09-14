# Docker Container Usage

This document explains how to use the pre-built Docker containers from GitHub Container Registry (GHCR).

## 📦 Available Images

The `pdfkb-mcp` Docker images are automatically built and published to the GitHub Container Registry on every push to the `main` branch and for every release.

### Image Tags

- **Latest stable**: `ghcr.io/juanqui/pdfkb-mcp:latest`

## 🚀 Quick Start

### Using Podman (Recommended)

```bash
# Pull and run the latest image
podman run -p 8000:8000 ghcr.io/juanqui/pdfkb-mcp:latest

# Run with a local documents directory
podman run -p 8000:8000 \
  -v ./documents:/app/documents \
  ghcr.io/juanqui/pdfkb-mcp:latest

# Run with web interface enabled
podman run -p 8000:8000 \
  -e PDFKB_WEB_ENABLE=true \
  -v ./documents:/app/documents \
  ghcr.io/juanqui/pdfkb-mcp:latest
```

### Using Docker

```bash
# Pull and run the latest image
docker run -p 8000:8000 ghcr.io/juanqui/pdfkb-mcp:latest

# Run with a local documents directory
docker run -p 8000:8000 \
  -v ./documents:/app/documents \
  ghcr.io/juanqui/pdfkb-mcp:latest

# Run with web interface enabled
docker run -p 8000:8000 \
  -e PDFKB_WEB_ENABLE=true \
  -v ./documents:/app/documents \
  ghcr.io/juanqui/pdfkb-mcp:latest
```

## 🐳 Docker Compose Usage

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  pdfkb-mcp:
    image: ghcr.io/juanqui/pdfkb-mcp:latest
    ports:
      - "8000:8000"
    volumes:
      - ./documents:/app/documents
      - ./cache:/app/cache
      - ./logs:/app/logs
    environment:
      - PDFKB_WEB_ENABLE=true
      - PDFKB_EMBEDDING_PROVIDER=local
      - PDFKB_LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

Then run:

```bash
# Using podman-compose
podman-compose up -d

# Using docker-compose
docker-compose up -d
```

## 🔧 Configuration

The container can be configured using environment variables. Here are the most common ones:

### Core Settings

- `PDFKB_EMBEDDING_PROVIDER`: Embedding provider (`local`, `openai`, `huggingface`)
- `PDFKB_WEB_ENABLE`: Enable web interface (`true`/`false`)
- `PDFKB_KNOWLEDGEBASE_PATH`: Path to documents directory (default: `/app/documents`)
- `PDFKB_CACHE_DIR`: Cache directory (default: `/app/cache`)

### API Keys (if using external providers)

- `PDFKB_OPENAI_API_KEY`: OpenAI API key for OpenAI embeddings
- `HF_TOKEN`: HuggingFace token for HuggingFace embeddings

### Server Configuration

- `PDFKB_SERVER_HOST`: Server host (default: `0.0.0.0`)
- `PDFKB_SERVER_PORT`: Server port (default: `8000`)
- `PDFKB_LOG_LEVEL`: Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### Advanced Settings

- `PDFKB_PDF_PARSER`: PDF parser (`pymupdf4llm`, `marker`, `docling`, etc.)
- `PDFKB_DOCUMENT_CHUNKER`: Text chunker (`langchain`, `semantic`, `page`)
- `PDFKB_ENABLE_HYBRID_SEARCH`: Enable hybrid search (`true`/`false`)
- `PDFKB_MAX_PARALLEL_PARSING`: Max parallel parsing operations (default: `1`)
- `PDFKB_MAX_PARALLEL_EMBEDDING`: Max parallel embedding operations (default: `1`)

## 📂 Volume Mounts

The container defines several volume mount points for data persistence:

### Required Volumes

- `/app/documents`: Your PDF and Markdown documents
- `/app/cache`: Embeddings and processing cache (important for performance)

### Optional Volumes

- `/app/logs`: Application logs
- `/app/config`: Configuration files

### Example with all volumes

```bash
podman run -p 8000:8000 \
  -v $(pwd)/documents:/app/documents \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/logs:/app/logs \
  -e PDFKB_WEB_ENABLE=true \
  ghcr.io/juanqui/pdfkb-mcp:latest
```

## 🌐 Transport Modes

The container supports different transport modes:

### HTTP Transport (Default)

```bash
podman run -p 8000:8000 \
  ghcr.io/juanqui/pdfkb-mcp:latest \
  pdfkb-mcp --transport http
```

**Endpoints:**
- MCP: `http://localhost:8000/mcp/`
- Health: `http://localhost:8000/health`
- Web UI: `http://localhost:8000/` (if enabled)

### SSE Transport

```bash
podman run -p 8000:8000 \
  ghcr.io/juanqui/pdfkb-mcp:latest \
  pdfkb-mcp --transport sse
```

**Endpoints:**
- MCP: `http://localhost:8000/sse/`
- Health: `http://localhost:8000/health`

### STDIO Transport

```bash
podman run \
  ghcr.io/juanqui/pdfkb-mcp:latest \
  pdfkb-mcp --transport stdio
```

## 🏥 Health Checks

The container includes a built-in health check endpoint:

```bash
curl http://localhost:8000/health
```

This endpoint returns JSON with system status, including:
- Service health
- Vector store status
- Background queue status
- Cache statistics

## 🔍 Monitoring and Logs

### View Container Logs

```bash
# Using podman
podman logs -f pdfkb-mcp

# Using docker
docker logs -f pdfkb-mcp
```

### Access Container Shell

```bash
# Using podman
podman exec -it pdfkb-mcp /bin/bash

# Using docker
docker exec -it pdfkb-mcp /bin/bash
```

## 🔐 Security Notes

The container runs as a non-root user (`pdfkb:pdfkb` with UID/GID 1001) for better security.

### File Permissions

When mounting volumes, ensure your local directories have appropriate permissions:

```bash
# Create directories with correct permissions
mkdir -p documents cache logs
chown -R 1001:1001 documents cache logs

# Or make them writable by the container user
chmod 755 documents cache logs
```

## 🚀 Performance Optimization

### For Production Use

```bash
podman run -d \
  --name pdfkb-mcp \
  -p 8000:8000 \
  -v $(pwd)/documents:/app/documents \
  -v $(pwd)/cache:/app/cache \
  -e PDFKB_WEB_ENABLE=true \
  -e PDFKB_EMBEDDING_PROVIDER=local \
  -e PDFKB_ENABLE_HYBRID_SEARCH=true \
  -e PDFKB_MAX_PARALLEL_PARSING=2 \
  -e PDFKB_MAX_PARALLEL_EMBEDDING=2 \
  -e PDFKB_BACKGROUND_QUEUE_WORKERS=4 \
  --memory=4g \
  --cpus=2 \
  --restart=unless-stopped \
  ghcr.io/juanqui/pdfkb-mcp:latest
```

### Resource Limits

- **Memory**: 2-4GB recommended (depends on document size and embedding model)
- **CPU**: 2+ cores recommended for parallel processing
- **Storage**: Fast SSD recommended for cache directory

## 🔄 Updates

To update to the latest version:

```bash
# Stop the container
podman stop pdfkb-mcp

# Pull the latest image
podman pull ghcr.io/juanqui/pdfkb-mcp:latest

# Start with the updated image
podman run -d \
  --name pdfkb-mcp \
  -p 8000:8000 \
  -v $(pwd)/documents:/app/documents \
  -v $(pwd)/cache:/app/cache \
  ghcr.io/juanqui/pdfkb-mcp:latest
```

## 🆘 Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure mounted directories are accessible by UID/GID 1001
2. **Port Already in Use**: Change the host port: `-p 8001:8000`
3. **Out of Memory**: Increase container memory limit or reduce parallel operations
4. **Cache Issues**: Clear the cache directory or start with a fresh cache volume

### Getting Help

- Check container logs: `podman logs pdfkb-mcp`
- Verify health: `curl http://localhost:8000/health`
- Check container status: `podman ps -a`
- Access container shell: `podman exec -it pdfkb-mcp /bin/bash`

## 📚 Further Reading

- [Main README](README.md) - Project overview and features
- [WARP.md](WARP.md) - Development and contribution guidelines
- [GitHub Repository](https://github.com/juanqui/pdfkb-mcp) - Source code and issues
