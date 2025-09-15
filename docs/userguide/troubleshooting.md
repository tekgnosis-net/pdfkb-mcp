# Troubleshooting Guide

Common issues, solutions, and performance tuning for **pdfkb-mcp**.

## Quick Diagnostics

### Check Server Status

```bash
# Test if server is installed and working
uvx pdfkb-mcp --help

# Test basic startup
PDFKB_EMBEDDING_PROVIDER=local PDFKB_LOG_LEVEL=DEBUG uvx pdfkb-mcp

# Test health endpoint (if running with web interface)
curl http://localhost:8000/health
```

### Check MCP Tools

After connecting to your MCP client, you should see these tools:
- ✅ `add_document` - Add PDFs to knowledge base
- ✅ `search_documents` - Search across documents
- ✅ `list_documents` - List all documents with metadata
- ✅ `remove_document` - Remove documents from knowledge base
- ✅ `rescan_documents` - Rescan documents directory

## Common Issues

### 🚨 Server Not Appearing in MCP Client

**Symptoms**: pdfkb-mcp doesn't show up in Claude Desktop, VS Code, etc.

**Solutions**:

1. **Check Configuration Syntax**:
   ```bash
   # Validate JSON configuration
   python -m json.tool ~/.config/Claude/claude_desktop_config.json
   ```

2. **Restart MCP Client Completely**:
   - Claude Desktop: Quit and restart the application
   - VS Code: Reload window (`Cmd/Ctrl + Shift + P` → "Developer: Reload Window")

3. **Verify Paths are Absolute**:
   ```json
   {
     "env": {
       "PDFKB_KNOWLEDGEBASE_PATH": "/Users/yourname/Documents",
       "PDFKB_CACHE_DIR": "/Users/yourname/Documents/.cache"
     }
   }
   ```

4. **Test Command Manually**:
   ```bash
   # Run the exact command from your config
   uvx pdfkb-mcp --help
   ```

### 🚨 Memory Issues / Out of Memory Errors

**Symptoms**: Process killed, "OOM" errors, system slowdown

**Solutions**:

1. **Reduce Memory Usage**:
   ```bash
   # Smaller embedding batches
   PDFKB_EMBEDDING_BATCH_SIZE=25

   # Smaller chunks
   PDFKB_CHUNK_SIZE=500

   # Reduce parallel operations
   PDFKB_MAX_PARALLEL_PARSING=1
   PDFKB_MAX_PARALLEL_EMBEDDING=1
   ```

2. **Use Memory-Efficient Models**:
   ```bash
   # GGUF quantized models use 50-70% less memory
   PDFKB_LOCAL_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B-GGUF"
   PDFKB_GGUF_QUANTIZATION="Q6_K"
   ```

3. **Optimize Parser Settings**:
   ```bash
   # Use fastest, most memory-efficient parser
   PDFKB_PDF_PARSER="pymupdf4llm"
   ```

### 🚨 Processing Too Slow

**Symptoms**: Documents take forever to process

**Solutions**:

1. **Optimize for Speed**:
   ```bash
   # Fastest parser
   PDFKB_PDF_PARSER="pymupdf4llm"

   # Larger chunks process faster
   PDFKB_CHUNK_SIZE=1200

   # Increase parallelism (if memory allows)
   PDFKB_MAX_PARALLEL_PARSING=2
   PDFKB_BACKGROUND_QUEUE_WORKERS=4
   ```

2. **Use Faster Embedding Models**:
   ```bash
   # Default local provider (fast, private)
   PDFKB_EMBEDDING_PROVIDER="local"
   PDFKB_LOCAL_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"

   # Or use an OpenAI-compatible provider (e.g., DeepInfra)
   PDFKB_EMBEDDING_PROVIDER="openai"
   PDFKB_OPENAI_API_BASE="https://api.deepinfra.com/v1"
   PDFKB_EMBEDDING_MODEL="text-embedding-3-small"
   ```

3. **Optimize Storage**:
   ```bash
   # Store cache on fast SSD
   PDFKB_CACHE_DIR="/path/to/fast/ssd/.cache"
   ```

### 🚨 API Key / Authentication Errors

**Symptoms**: "Invalid API key", "Authentication failed"

**Solutions**:

1. **Verify API Key Format**:
   ```bash
   # OpenAI keys start with 'sk-proj-' or 'sk-'
   echo $PDFKB_OPENAI_API_KEY | head -c 10

   # DeepInfra keys are typically shorter
   echo $PDFKB_OPENAI_API_KEY | wc -c
   ```

2. **Test API Key**:
   ```bash
   # Test OpenAI
   curl -H "Authorization: Bearer $PDFKB_OPENAI_API_KEY" \
        https://api.openai.com/v1/models

   # Test DeepInfra
   curl -H "Authorization: Bearer $PDFKB_OPENAI_API_KEY" \
        https://api.deepinfra.com/v1/models
   ```

3. **Check API Base URL**:
   ```bash
   # For DeepInfra, ensure correct base URL
   PDFKB_OPENAI_API_BASE="https://api.deepinfra.com/v1"
   ```

### 🚨 Poor Search Results

**Symptoms**: Irrelevant results, missing obvious matches

**Solutions**:

1. **Enable Hybrid Search**:
   ```bash
   PDFKB_ENABLE_HYBRID_SEARCH=true
   PDFKB_HYBRID_VECTOR_WEIGHT=0.6
   PDFKB_HYBRID_TEXT_WEIGHT=0.4
   ```

2. **Add Reranking**:
   ```bash
   PDFKB_ENABLE_RERANKER=true
   PDFKB_RERANKER_PROVIDER=deepinfra
   PDFKB_DEEPINFRA_API_KEY="your-key"
   ```

3. **Use Better Chunking**:
Note: Default chunker is `langchain`. Set `PDFKB_PDF_CHUNKER="semantic"` to use semantic chunking.
   ```bash
   # Semantic chunking for better context
   PDFKB_PDF_CHUNKER="semantic"
   PDFKB_SEMANTIC_CHUNKER_THRESHOLD_TYPE="percentile"
   PDFKB_SEMANTIC_CHUNKER_THRESHOLD_AMOUNT="95.0"
   ```

### 🚨 Docker/Container Issues

**Symptoms**: Container won't start, permission errors

**Solutions**:

1. **Fix Volume Permissions**:
   ```bash
   # Fix ownership
   sudo chown -R 1001:1001 ./documents ./cache ./logs

   # Or use current user
   sudo chown -R $(id -u):$(id -g) ./documents ./cache ./logs
   ```

2. **Check Port Conflicts**:
   ```bash
   # Check if port is in use
   lsof -i :8000
   netstat -tulpn | grep :8000

   # Use different port
   PDFKB_WEB_PORT=8001 podman-compose up -d
   ```

3. **Container Logs**:
   ```bash
   # View container logs
   podman logs pdfkb-mcp --tail 50
   docker logs pdfkb-mcp --tail 50

   # Follow logs in real-time
   podman-compose logs -f
   docker compose logs -f
   ```

## Performance Tuning

### Resource Optimization by System

| System Type | RAM | Configuration |
|-------------|-----|---------------|
| **Low-end** (2-4GB) | Limited | `pymupdf4llm`, local embeddings, batch=25 |
| **Mid-range** (4-8GB) | Moderate | `marker`, hybrid search, batch=50 |
| **High-end** (8GB+) | Abundant | `docling`, reranking, larger batches |

### Low-Resource Systems

```bash
# Minimal resource usage
PDFKB_PDF_PARSER="pymupdf4llm"
PDFKB_EMBEDDING_PROVIDER="local"
PDFKB_LOCAL_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B-GGUF"
PDFKB_GGUF_QUANTIZATION="Q4_K_M"
PDFKB_EMBEDDING_BATCH_SIZE=25
PDFKB_MAX_PARALLEL_PARSING=1
PDFKB_MAX_PARALLEL_EMBEDDING=1
PDFKB_BACKGROUND_QUEUE_WORKERS=1
PDFKB_CHUNK_SIZE=800
```

### High-Performance Systems

```bash
# Maximum performance and quality
PDFKB_PDF_PARSER="docling"
PDFKB_EMBEDDING_PROVIDER="openai"
PDFKB_OPENAI_API_BASE="https://api.deepinfra.com/v1"
PDFKB_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
PDFKB_ENABLE_HYBRID_SEARCH=true
PDFKB_ENABLE_RERANKER=true
PDFKB_RERANKER_PROVIDER="deepinfra"
PDFKB_EMBEDDING_BATCH_SIZE=100
PDFKB_MAX_PARALLEL_PARSING=4
PDFKB_MAX_PARALLEL_EMBEDDING=2
PDFKB_BACKGROUND_QUEUE_WORKERS=4
PDFKB_CHUNK_SIZE=1200
```

### GPU Optimization

**Apple Silicon (M1/M2/M3)**:
```bash
# Automatic MPS acceleration
PDFKB_EMBEDDING_DEVICE="mps"
PDFKB_LOCAL_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"
```

**NVIDIA GPU**:
```bash
# Automatic CUDA acceleration
PDFKB_EMBEDDING_DEVICE="cuda"
PDFKB_PDF_PARSER="mineru"  # GPU-accelerated parser
```

## Monitoring and Debugging

### Enable Debug Logging

```bash
# Debug all components
PDFKB_LOG_LEVEL=DEBUG uvx pdfkb-mcp

# Or in MCP client config
"env": {
  "PDFKB_LOG_LEVEL": "DEBUG"
}
```

### Monitor Resource Usage

```bash
# Container resource usage
podman stats pdfkb-mcp
docker stats pdfkb-mcp

# System resource usage
htop
# Look for pdfkb-mcp or python processes
```

### Check Cache Status

```bash
# View cache directory
ls -la .cache/
du -sh .cache/

# Clear cache if needed (will reprocess all documents)
rm -rf .cache/
```

### Test Components Individually

```bash
# Test embedding service
PDFKB_EMBEDDING_PROVIDER=local python -c "
from pdfkb.embeddings_factory import create_embedding_service
service = create_embedding_service()
result = service.embed(['test text'])
print('Embedding shape:', result[0].shape)
"
```

## Configuration Validation

### Check Environment Variables

```bash
# Print all PDFKB variables
env | grep PDFKB | sort

# Validate key settings
python -c "
import os
print('Documents:', os.environ.get('PDFKB_KNOWLEDGEBASE_PATH'))
print('Cache:', os.environ.get('PDFKB_CACHE_DIR'))
print('Provider:', os.environ.get('PDFKB_EMBEDDING_PROVIDER'))
print('Parser:', os.environ.get('PDFKB_PDF_PARSER'))
"
```

### Test Paths

```bash
# Check document directory exists and is readable
test -d "$PDFKB_KNOWLEDGEBASE_PATH" && echo "✅ Documents dir exists"
test -r "$PDFKB_KNOWLEDGEBASE_PATH" && echo "✅ Documents dir readable"

# Check cache directory is writable
test -w "$PDFKB_CACHE_DIR" && echo "✅ Cache dir writable"
```

## Getting Help

### Debug Information to Collect

When reporting issues, include:

1. **System Information**:
   ```bash
   uname -a
   python --version
   uvx --version
   ```

2. **Configuration**:
   ```bash
   env | grep PDFKB | sort
   ```

3. **Error Logs**:
   ```bash
   PDFKB_LOG_LEVEL=DEBUG uvx pdfkb-mcp 2>&1 | head -50
   ```

4. **Resource Usage**:
   ```bash
   free -h
   df -h
   ```

### Community Support

- **GitHub Issues**: [Report bugs](https://github.com/juanqui/pdfkb-mcp/issues)
- **Discussions**: [Ask questions](https://github.com/juanqui/pdfkb-mcp/discussions)
- **Documentation**: [User Guide](index.md)

## Prevention Tips

### Regular Maintenance

```bash
# Monitor cache size
du -sh .cache/

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Update to latest version
uvx pdfkb-mcp --reinstall
```

### Configuration Best Practices

1. **Always use absolute paths** in configuration
2. **Set appropriate resource limits** for your system
3. **Monitor memory usage** during processing
4. **Use version pinning** for production deployments
5. **Regular backups** of document cache

### Performance Monitoring

```bash
# Simple monitoring script
while true; do
  echo "$(date): Memory $(free -h | grep Mem | awk '{print $3}')"
  sleep 60
done
```

This covers most common issues you'll encounter. For specific problems not covered here, check the [Configuration Guide](configuration.md) or [Advanced Features](advanced.md) guides.
