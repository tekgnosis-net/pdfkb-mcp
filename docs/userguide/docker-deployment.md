# Docker/Podman Deployment Guide

Complete guide for deploying **pdfkb-mcp** using Docker or Podman containers.

## Quick Start with Docker Compose

### 1. Download Configuration

```bash
# Get the sample configuration
curl -o docker-compose.yml https://raw.githubusercontent.com/juanqui/pdfkb-mcp/main/docker-compose.sample.yml

# Create necessary directories
mkdir -p ./documents ./cache ./logs
```

### 2. Configure Your Setup

Edit `docker-compose.yml` and update these essential settings:

```yaml
volumes:
  # Update this path to your document collection
  - "/path/to/your/documents:/app/documents:rw"

environment:
  # Add your DeepInfra API key
  PDFKB_OPENAI_API_KEY: "your-deepinfra-api-key-here"
```

### 3. Start Services

```bash
# Using Podman (recommended)
podman-compose up -d

# Or using Docker
docker compose up -d
```

Your services will be available at:
- **Web Interface**: http://localhost:8000
- **MCP Endpoint**: http://localhost:8000/mcp/ (HTTP transport)

## Configuration Options

### Environment Variables

The docker-compose configuration supports all pdfkb-mcp environment variables:

#### Core Configuration
```yaml
environment:
  # Essential settings
  PDFKB_KNOWLEDGEBASE_PATH: "/app/documents"
  PDFKB_CACHE_DIR: "/app/documents/.cache"
  PDFKB_LOG_LEVEL: "INFO"
```

Note: MCP transport is configured by the client or via CLI flag (e.g., `--transport http`), not by an environment variable.

#### Embedding Configuration
```yaml
environment:
  # DeepInfra (recommended)
  PDFKB_EMBEDDING_PROVIDER: "openai"
  PDFKB_OPENAI_API_KEY: "your-deepinfra-api-key"
  PDFKB_OPENAI_API_BASE: "https://api.deepinfra.com/v1"
  PDFKB_EMBEDDING_MODEL: "Qwen/Qwen3-Embedding-8B"

  # Local embeddings (no API key required)
  # PDFKB_EMBEDDING_PROVIDER: "local"
  # PDFKB_LOCAL_EMBEDDING_MODEL: "Qwen/Qwen3-Embedding-0.6B"
```

#### Search Features
```yaml
environment:
  # Advanced search capabilities
  PDFKB_ENABLE_HYBRID_SEARCH: "true"
  PDFKB_ENABLE_RERANKER: "true"
  PDFKB_RERANKER_PROVIDER: "deepinfra"
  PDFKB_DEEPINFRA_API_KEY: "your-deepinfra-api-key"
  PDFKB_DEEPINFRA_RERANKER_MODEL: "Qwen/Qwen3-Reranker-8B"
```

#### Document Summarization
```yaml
environment:
  # AI-powered document summarization
  PDFKB_ENABLE_SUMMARIZER: "true"
  PDFKB_SUMMARIZER_PROVIDER: "remote"
  PDFKB_SUMMARIZER_API_KEY: "your-deepinfra-api-key"
  PDFKB_SUMMARIZER_API_BASE: "https://api.deepinfra.com/v1"
  PDFKB_SUMMARIZER_MODEL: "Qwen/Qwen3-Next-80B-A3B-Instruct"
```

### Volume Mounts

Configure persistent storage for your data:

```yaml
volumes:
  # Documents directory - your PDF/Markdown collection
  - "/path/to/your/documents:/app/documents:rw"

  # Cache directory - processed data and embeddings
  - "/path/to/your/documents/.cache:/app/documents/.cache:rw"

  # Logs directory - container logs (optional)
  - "pdfkb-logs:/app/logs"

  # Configuration directory - custom config files (optional)
  - "./config:/app/config:ro"
```

### Resource Limits

Adjust resource limits based on your system:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'        # Increase for better performance
      memory: 8G         # Increase for large document collections
    reservations:
      cpus: '0.5'
      memory: 1G
```

## Deployment Scenarios

### Privacy-Focused Deployment (Local Embeddings)

```yaml
services:
  pdfkb-mcp:
    image: ghcr.io/juanqui/pdfkb-mcp:latest
    environment:
      # No API keys required - everything runs locally
      PDFKB_EMBEDDING_PROVIDER: "local"
      PDFKB_LOCAL_EMBEDDING_MODEL: "Qwen/Qwen3-Embedding-0.6B-GGUF"
      PDFKB_GGUF_QUANTIZATION: "Q6_K"
      PDFKB_ENABLE_HYBRID_SEARCH: "true"
      PDFKB_ENABLE_RERANKER: "true"
      PDFKB_RERANKER_PROVIDER: "local"
      PDFKB_RERANKER_MODEL: "Mungert/Qwen3-Reranker-0.6B-GGUF"
    deploy:
      resources:
        limits:
          memory: 4G
```

### High-Performance Deployment

```yaml
services:
  pdfkb-mcp:
    image: ghcr.io/juanqui/pdfkb-mcp:latest
    environment:
      # Maximum performance with remote APIs
      PDFKB_EMBEDDING_PROVIDER: "openai"
      PDFKB_OPENAI_API_BASE: "https://api.deepinfra.com/v1"
      PDFKB_EMBEDDING_MODEL: "Qwen/Qwen3-Embedding-8B"
      PDFKB_PDF_PARSER: "docling"
      PDFKB_ENABLE_HYBRID_SEARCH: "true"
      PDFKB_ENABLE_RERANKER: "true"
      PDFKB_ENABLE_SUMMARIZER: "true"
      PDFKB_MAX_PARALLEL_PARSING: "4"
      PDFKB_BACKGROUND_QUEUE_WORKERS: "4"
    deploy:
      resources:
        limits:
          cpus: '6.0'
          memory: 12G
```

### Academic/Research Setup

```yaml
services:
  pdfkb-mcp:
    image: ghcr.io/juanqui/pdfkb-mcp:latest
    environment:
      # Optimized for academic papers
      PDFKB_PDF_PARSER: "marker"
      PDFKB_PDF_CHUNKER: "semantic"
      PDFKB_SEMANTIC_CHUNKER_THRESHOLD_TYPE: "percentile"
      PDFKB_SEMANTIC_CHUNKER_THRESHOLD_AMOUNT: "95.0"
      PDFKB_EMBEDDING_PROVIDER: "local"
      PDFKB_LOCAL_EMBEDDING_MODEL: "Qwen/Qwen3-Embedding-4B"
      PDFKB_ENABLE_HYBRID_SEARCH: "true"
      PDFKB_ENABLE_RERANKER: "true"
```

## Container Management

### Basic Operations

```bash
# Start services
podman-compose up -d          # Podman
docker compose up -d          # Docker

# Stop services
podman-compose down           # Podman
docker compose down           # Docker

# View logs
podman-compose logs -f        # Podman
docker compose logs -f        # Docker

# Check status
podman-compose ps             # Podman
docker compose ps             # Docker
```

### Updating

```bash
# Pull latest image and restart
podman-compose pull && podman-compose up -d    # Podman
docker compose pull && docker compose up -d    # Docker
```

### Health Monitoring

```bash
# Check container health
curl http://localhost:8000/health

# View resource usage
podman stats pdfkb-mcp       # Podman
docker stats pdfkb-mcp       # Docker

# Inspect container
podman inspect pdfkb-mcp     # Podman
docker inspect pdfkb-mcp     # Docker
```

## MCP Client Configuration

### Claude Desktop (HTTP Transport)

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

### VS Code with Continue

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

### Cline

```json
{
  "mcpServers": {
    "pdfkb": {
      "transport": "http",
      "url": "http://localhost:8000/mcp/",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
```

## Advanced Configuration

### Multi-Container Setup

For high-availability deployments:

```yaml
version: '3.8'

services:
  pdfkb-mcp:
    image: ghcr.io/juanqui/pdfkb-mcp:latest
    restart: unless-stopped
    environment:
      PDFKB_EMBEDDING_PROVIDER: "openai"
      PDFKB_OPENAI_API_BASE: "https://api.deepinfra.com/v1"
      PDFKB_OPENAI_API_KEY: "${DEEPINFRA_API_KEY}"
    volumes:
      - "documents:/app/documents:rw"
      - "cache:/app/cache"
    ports:
      - "8000:8000"
    networks:
      - pdfkb-network
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
      restart_policy:
        condition: on-failure

  loadbalancer:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - "./nginx.conf:/etc/nginx/nginx.conf:ro"
    depends_on:
      - pdfkb-mcp
    networks:
      - pdfkb-network

volumes:
  documents:
  cache:

networks:
  pdfkb-network:
    driver: bridge
```

### Custom Configuration Files

Mount custom configuration:

```yaml
volumes:
  - "./custom.env:/app/config/custom.env:ro"

environment:
  PDFKB_CONFIG_FILE: "/app/config/custom.env"
```

### Secrets Management

Use Docker secrets for sensitive data:

```yaml
secrets:
  deepinfra_key:
    file: ./secrets/deepinfra_key.txt

services:
  pdfkb-mcp:
    secrets:
      - deepinfra_key
    environment:
      PDFKB_OPENAI_API_KEY_FILE: "/run/secrets/deepinfra_key"
```

## Troubleshooting

### Common Issues

**1. Permission Errors**
```bash
# Fix volume permissions
sudo chown -R 1001:1001 ./documents ./cache ./logs

# Or use your current user
sudo chown -R $(id -u):$(id -g) ./documents ./cache ./logs
```

**2. Port Conflicts**
```bash
# Check port usage
lsof -i :8000
netstat -tulpn | grep :8000

# Use different port
PDFKB_WEB_PORT=8001 podman-compose up -d
```

**3. Memory Issues**
```yaml
# Reduce resource usage in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
environment:
  PDFKB_EMBEDDING_BATCH_SIZE: "25"
  PDFKB_MAX_PARALLEL_PARSING: "1"
```

**4. Container Won't Start**
```bash
# Check logs for errors
podman-compose logs pdfkb-mcp
docker compose logs pdfkb-mcp

# Run container interactively for debugging
podman run -it --rm ghcr.io/juanqui/pdfkb-mcp:latest /bin/bash
```

### Debug Mode

Enable debugging:

```yaml
environment:
  PDFKB_LOG_LEVEL: "DEBUG"
```

### Manual Container Run

For debugging, run container manually:

```bash
# Podman (recommended)
podman run -it --rm \
  -p 8000:8000 \
  -v "$(pwd)/documents:/app/documents:rw" \
  -e PDFKB_LOG_LEVEL=DEBUG \
  -e PDFKB_EMBEDDING_PROVIDER=local \
  ghcr.io/juanqui/pdfkb-mcp:latest

# Docker
docker run -it --rm \
  -p 8000:8000 \
  -v "$(pwd)/documents:/app/documents:rw" \
  -e PDFKB_LOG_LEVEL=DEBUG \
  -e PDFKB_EMBEDDING_PROVIDER=local \
  ghcr.io/juanqui/pdfkb-mcp:latest
```

## Building from Source

### Build Container Image

```bash
# Clone repository
git clone https://github.com/juanqui/pdfkb-mcp.git
cd pdfkb-mcp

# Build with Podman (preferred)
podman build -t pdfkb-mcp:local .

# Build with Docker
docker build -t pdfkb-mcp:local .

# Use local image
# Update docker-compose.yml:
# image: pdfkb-mcp:local
```

### Development Build

```bash
# Build development image
podman build -f Dockerfile.dev -t pdfkb-mcp:dev .

# Run development container
podman-compose -f docker-compose.dev.yml up -d
```

## Production Considerations

### Security

```yaml
# Security hardening in docker-compose.yml
security_opt:
  - no-new-privileges:true
user: "1001:1001"
read_only: true
tmpfs:
  - /tmp
  - /app/tmp
```

### Monitoring

```yaml
# Health checks
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  start_period: 60s
  retries: 3

# Logging
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Backup

```bash
# Backup documents and cache
tar -czf pdfkb-backup-$(date +%Y%m%d).tar.gz documents/ cache/

# Backup docker volumes
podman volume export pdfkb_documents > pdfkb-documents-backup.tar
podman volume export pdfkb_cache > pdfkb-cache-backup.tar
```

This guide covers comprehensive Docker/Podman deployment. For specific configuration details, see the [Configuration Guide](configuration.md).
