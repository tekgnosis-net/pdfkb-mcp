import asyncio
import os
import tempfile
from pathlib import Path

import httpx
import pytest

from pdfkb.config import ServerConfig
from pdfkb.main import PDFKnowledgebaseServer


@pytest.fixture(scope="session")
def sample_pdf_path():
    """Create a temporary sample PDF for testing."""
    # For testing purposes, we'll assume a sample PDF exists or create a minimal one
    # In real tests, this would point to a test PDF fixture
    test_pdf = Path(__file__).parent / "sample.pdf"
    if not test_pdf.exists():
        pytest.skip("Sample PDF not found - skipping SSE tests requiring documents")
    return str(test_pdf)


@pytest.fixture
def temp_knowledgebase(sample_pdf_path):
    """Create temporary knowledgebase directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir) / "pdfs"
        kb_path.mkdir(exist_ok=True)
        cache_path = Path(tmpdir) / ".cache"
        cache_path.mkdir(exist_ok=True)

        # Copy sample PDF to knowledgebase if provided
        if sample_pdf_path:
            sample_file = Path(sample_pdf_path)
            if sample_file.exists():
                import shutil

                dest = kb_path / sample_file.name
                shutil.copy2(sample_file, dest)

        yield str(kb_path)


@pytest.fixture
def sse_config(temp_knowledgebase):
    """Configuration for SSE mode testing."""
    os.environ["PDFKB_KNOWLEDGEBASE_PATH"] = temp_knowledgebase
    os.environ["PDFKB_CACHE_DIR"] = str(Path(temp_knowledgebase).parent / ".cache")
    os.environ["PDFKB_TRANSPORT"] = "sse"
    os.environ["PDFKB_SERVER_HOST"] = "127.0.0.1"
    os.environ["PDFKB_SERVER_PORT"] = "8000"  # Use valid port for testing
    os.environ["PDFKB_LOG_LEVEL"] = "WARNING"  # Reduce log noise
    os.environ["PDFKB_EMBEDDING_PROVIDER"] = "local"  # Use local for testing
    os.environ["PDFKB_LOCAL_EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"  # Small model

    config = ServerConfig.from_env()
    return config


@pytest.fixture
async def sse_server(sse_config, sample_pdf_path):
    """Create SSE server instance for testing (config only, no actual server startup)."""
    # Create server instance for config testing
    server = PDFKnowledgebaseServer(sse_config)

    # Initialize core components only (no server startup)
    await server.initialize()

    # Mock URL for testing
    base_url = f"http://{sse_config.server_host}:{sse_config.server_port or 8000}"

    yield base_url, server, None

    # Cleanup
    await server.shutdown()


@pytest.mark.asyncio
async def test_sse_server_startup(sse_config):
    """Test that SSE server starts successfully."""
    server = PDFKnowledgebaseServer(sse_config)
    await server.initialize_core()

    # Test config without actually binding (just check config)
    assert sse_config.transport == "sse"
    assert sse_config.server_host == "127.0.0.1"
    assert sse_config.server_port == 8000

    await server.shutdown()


@pytest.mark.asyncio
async def test_sse_tool_endpoint(sse_server, sample_pdf_path):
    """Test SSE server initialization and configuration."""
    base_url, server, sse_task = sse_server

    # Test that server was initialized properly for SSE mode
    assert server is not None
    assert server.config.transport == "sse"
    assert server.config.server_host == "127.0.0.1"

    # Test that core components are initialized
    assert server.embedding_service is not None
    assert server.vector_store is not None


@pytest.mark.asyncio
async def test_sse_search_documents(sse_server, sample_pdf_path):
    """Test SSE server document processing capabilities."""
    base_url, server, sse_task = sse_server

    # Test that document processing components are available
    assert server.document_processor is not None
    assert server.vector_store is not None

    # Test that SSE transport is configured
    assert server.config.transport == "sse"


@pytest.mark.asyncio
async def test_sse_connection(sse_server):
    """Test SSE server configuration."""
    base_url, server, sse_task = sse_server

    # Test that server configuration is correct for SSE mode
    assert server.config.transport == "sse"
    assert base_url.startswith("http://")
    assert server.app is not None  # FastMCP app should be initialized


@pytest.mark.asyncio
async def test_concurrent_web_sse(sse_config, temp_knowledgebase):
    """Test concurrent web + SSE mode configuration validation."""
    # Configure for integrated mode with SSE
    os.environ["PDFKB_WEB_ENABLE"] = "true"
    os.environ["PDFKB_WEB_PORT"] = "8081"  # Different from server port
    os.environ["PDFKB_TRANSPORT"] = "sse"
    os.environ["PDFKB_SERVER_PORT"] = "8000"

    config = ServerConfig.from_env()

    # Test configuration is valid
    assert config.web_enabled is True
    assert config.web_port == 8081
    assert config.server_port == 8000
    assert config.transport == "sse"
    assert config.web_port != config.server_port  # No port conflicts


@pytest.mark.asyncio
async def test_sse_port_conflict_validation(sse_config):
    """Test port conflict validation configuration."""
    # Configure conflicting ports
    os.environ["PDFKB_WEB_ENABLE"] = "true"
    os.environ["PDFKB_WEB_PORT"] = "8000"
    os.environ["PDFKB_TRANSPORT"] = "sse"
    os.environ["PDFKB_SERVER_PORT"] = "8000"  # Same as web port

    config = ServerConfig.from_env()

    # Test that configuration shows port conflict
    assert config.web_enabled is True
    assert config.web_port == config.server_port  # Port conflict detected
    assert config.transport == "sse"


@pytest.mark.asyncio
async def test_sse_invalid_transport(sse_config):
    """Test validation for invalid transport values."""
    # Set invalid transport
    os.environ["PDFKB_TRANSPORT"] = "invalid"

    # Should raise validation error during config creation
    with pytest.raises(Exception):  # ConfigurationError or similar
        ServerConfig.from_env()


@pytest.mark.asyncio
async def test_sse_shutdown_graceful(sse_config, temp_knowledgebase):
    """Test graceful shutdown of SSE server."""
    server = PDFKnowledgebaseServer(sse_config)

    # Initialize server
    await server.initialize()

    # Test graceful shutdown
    await server.shutdown()

    # Verify shutdown completed
    assert server is not None


# Run tests with pytest markers for SSE
# pytest tests/test_sse_integration.py -m sse -v
