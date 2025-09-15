# PDF Knowledgebase MCP Server - User Guide

Welcome to the comprehensive user guide for **pdfkb-mcp**, a Model Context Protocol (MCP) server that enables intelligent document search and retrieval from PDF and Markdown collections.

## Quick Navigation

### Getting Started
- [🚀 **Quick Start**](quick-start.md) - Get up and running in minutes with Docker/Podman
- [📦 **Installation**](installation.md) - Installation options and requirements

### Core Features
- [🔍 **Search Features**](search-features.md) - Hybrid search, reranking, and semantic chunking
- [🤖 **Embeddings**](embeddings.md) - Local, OpenAI, and HuggingFace embedding options
- [📄 **PDF Parsers**](parsers.md) - Parser selection and configuration guide

### Setup and Configuration
- [⚙️ **Configuration**](configuration.md) - Environment variables and settings
- [🌐 **Web Interface**](web-interface.md) - Web UI setup and usage
- [🔌 **MCP Clients**](mcp-clients.md) - Setup with Claude Desktop, VS Code, etc.

### Deployment
- [🐳 **Docker/Podman Deployment**](docker-deployment.md) - Container deployment with compose
- [🔧 **Troubleshooting**](troubleshooting.md) - Performance tuning and common issues
- [🎯 **Advanced Features**](advanced.md) - Document summarization and enterprise features

## What is pdfkb-mcp?

**pdfkb-mcp** is a powerful MCP server that transforms your document collection into an intelligent, searchable knowledge base. It processes PDF and Markdown files, creates semantic embeddings, and provides advanced search capabilities through the Model Context Protocol.

### Key Capabilities

- **Intelligent Document Processing**: Multiple PDF parsers optimized for different document types
- **Advanced Search**: Hybrid search combining semantic similarity with keyword matching
- **Privacy-Focused**: Local embeddings option - no API costs, full privacy
- **Multi-Client Support**: Works with Claude Desktop, VS Code, Continue, and other MCP clients
- **Web Interface**: Modern web UI for document management and search
- **Enterprise Ready**: Docker deployment, background processing, intelligent caching

## Getting Help

- **Quick Issues**: Check the [Troubleshooting Guide](troubleshooting.md)
- **Configuration**: See the [Configuration Reference](configuration.md)
- **GitHub Issues**: [Report bugs or request features](https://github.com/juanqui/pdfkb-mcp/issues)

## License

This project is licensed under the MIT License.
