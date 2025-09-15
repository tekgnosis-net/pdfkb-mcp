# Web Interface Guide

This guide is being expanded. For now:
- Defaults: parser = `pymupdf4llm`; embeddings provider = `local` (Qwen/Qwen3-Embedding-0.6B); chunker = `langchain`; hybrid search = enabled; reranker = disabled; summarizer = disabled.
- See the [Quick Start](quick-start.md), [Troubleshooting](troubleshooting.md), and [Docker Deployment](docker-deployment.md) guides for working configurations.

For immediate help, see:
- [Quick Start Guide](quick-start.md) for web interface setup
- [Docker Deployment Guide](docker-deployment.md) for container-based web interface

Current endpoints:
- Enable web: set `PDFKB_WEB_ENABLE=true`
- Web UI: `http://localhost:8000/`
- Health: `http://localhost:8000/health`
- MCP HTTP endpoint: `http://localhost:8000/mcp/`
- WebSocket: `ws://localhost:8000/ws`

This guide will include:
- Web interface setup and configuration
- Document upload and management
- Search interface usage
- Real-time processing monitoring
- API documentation access

**Coming soon...**
