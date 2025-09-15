# Installation Guide

This guide covers all installation methods for **pdfkb-mcp**.

## Recommended Installation Methods

### 1. Using uvx (Recommended)

**uvx** is the fastest and simplest way to install and run pdfkb-mcp:

```bash
# Basic installation
uvx pdfkb-mcp

# With specific features
uvx pdfkb-mcp[hybrid]         # Hybrid search
uvx pdfkb-mcp[semantic]       # Semantic chunking
uvx pdfkb-mcp[marker]         # Marker parser
uvx pdfkb-mcp[docling]        # Docling parser
```

**Advantages**:
- ✅ Automatic dependency isolation
- ✅ No virtual environment management
- ✅ Always gets the latest version
- ✅ Easy to use with MCP clients

### 2. Docker/Podman (Recommended for Production)

Perfect for production deployments and easy management:

```bash
# Quick start with Podman (recommended)
mkdir -p ./documents ./cache
podman run -d \
  --name pdfkb-mcp \
  -p 8000:8000 \
  -v "$(pwd)/documents:/app/documents:rw" \
  -v "$(pwd)/cache:/app/cache" \
  -e PDFKB_EMBEDDING_PROVIDER=local \
  ghcr.io/juanqui/pdfkb-mcp:latest

# Or with Docker
docker run -d \
  --name pdfkb-mcp \
  -p 8000:8000 \
  -v "$(pwd)/documents:/app/documents:rw" \
  -v "$(pwd)/cache:/app/cache" \
  -e PDFKB_EMBEDDING_PROVIDER=local \
  ghcr.io/juanqui/pdfkb-mcp:latest
```

**Advantages**:
- ✅ Consistent deployment across environments
- ✅ Easy scaling and management
- ✅ Isolated dependencies
- ✅ Production-ready with health checks

See the [Docker Deployment Guide](docker-deployment.md) for detailed setup.

## Alternative Installation Methods

### 3. Using pip/pipx

Install into a Python environment:

```bash
# Using pip
pip install pdfkb-mcp

# Using pipx (isolated installation)
pipx install pdfkb-mcp

# With optional features
pip install "pdfkb-mcp[hybrid,semantic,marker]"
```

### 4. Development Installation

For contributing or customizing:

```bash
git clone https://github.com/juanqui/pdfkb-mcp.git
cd pdfkb-mcp
pip install -e ".[dev]"
```

## Optional Feature Dependencies

Install additional features as needed:

| Feature | Install Command | Description |
|---------|----------------|-------------|
| **Hybrid Search** | `[hybrid]` | Semantic + keyword search combination |
| **Semantic Chunking** | `[semantic]` | Context-aware document chunking |
| **Marker Parser** | `[marker]` | High-quality PDF parsing with OCR |
| **MinerU Parser** | `[mineru]` | GPU-accelerated parsing for academic papers |
| **Docling Parser** | `[docling]` | Advanced table extraction and OCR |
| **LLM Parser** | `[llm]` | AI-powered parsing for complex layouts |
| **Unstructured Chunking** | `[unstructured_chunker]` | Advanced semantic chunking |
| **Web Enhanced** | `[web]` | Enhanced web interface features |

**Examples**:
```bash
# Multiple features
uvx pdfkb-mcp[hybrid,semantic,marker]

# All parsers
pip install "pdfkb-mcp[marker,mineru,docling,llm]"

# Full installation
pip install "pdfkb-mcp[hybrid,semantic,marker,docling,web]"
```

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or later
- **RAM**: 2GB available memory
- **Storage**: 1GB for dependencies + document storage
- **OS**: macOS, Linux, or Windows

### Recommended Requirements
- **RAM**: 4GB+ for better performance
- **Storage**: SSD for faster document processing
- **GPU**: NVIDIA GPU for accelerated parsers (MinerU, Marker)

### Platform-Specific Notes

**macOS**:
- Apple Silicon (M1/M2/M3) automatically uses Metal acceleration
- Intel Macs use CPU-based processing

**Linux**:
- NVIDIA GPUs automatically detected for CUDA acceleration
- Requires Docker/Podman for containerized deployment

**Windows**:
- WSL2 recommended for best compatibility
- Native Windows support available

## Verifying Your Installation

After installation, verify everything works:

```bash
# Test basic functionality
pdfkb-mcp --help

# Test with minimal configuration
PDFKB_EMBEDDING_PROVIDER=local pdfkb-mcp --log-level INFO
```

You should see startup logs indicating:
- ✅ Configuration loaded
- ✅ Parser selected (PyMuPDF4LLM by default)
- ✅ Embedding provider initialized (local by default)
- ✅ MCP server started

## Next Steps

After installation:

1. **[🚀 Quick Start](quick-start.md)**: Get up and running quickly
2. **[⚙️ Configuration](configuration.md)**: Configure for your needs
3. **[🔌 MCP Clients](mcp-clients.md)**: Set up your MCP client
4. **[🔧 Troubleshooting](troubleshooting.md)**: Common issues and solutions

## Common Installation Issues

### uvx Not Found
```bash
# Install uvx first
pip install uvx
# or with pipx
pipx install uvx
```

### Permission Errors
```bash
# Use user installation
pip install --user pdfkb-mcp

# Or use pipx for isolated installation
pipx install pdfkb-mcp
```

### Missing Dependencies
```bash
# Install with all optional dependencies
pip install "pdfkb-mcp[hybrid,semantic,marker,docling,web]"
```

### Docker Permission Issues
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Then log out and back in

# Or use Podman (rootless by default)
podman run ...
```

For more troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).
