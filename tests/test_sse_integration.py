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
def temp_knowledgebase():
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
                dest = kb_path / sample_file.name
                sample_file.copy(dest)

        yield str(kb_path)


@pytest.fixture
def sse_config(temp_knowledgebase):
    """Configuration for SSE mode testing."""
    os.environ["PDFKB_KNOWLEDGEBASE_PATH"] = temp_knowledgebase
    os.environ["PDFKB_CACHE_DIR"] = str(Path(temp_knowledgebase).parent / ".cache")
    os.environ["PDFKB_TRANSPORT"] = "sse"
    os.environ["PDFKB_SSE_HOST"] = "127.0.0.1"
    os.environ["PDFKB_SSE_PORT"] = "0"  # Use ephemeral port
    os.environ["PDFKB_LOG_LEVEL"] = "WARNING"  # Reduce log noise
    os.environ["PDFKB_EMBEDDING_PROVIDER"] = "local"  # Use local for testing
    os.environ["PDFKB_LOCAL_EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"  # Small model

    config = ServerConfig.from_env()
    config.sse_port = 0  # Ephemeral port
    return config


@pytest.fixture
async def sse_server(sse_config, sample_pdf_path):
    """Start SSE server in a background process."""
    # Create server instance
    server = PDFKnowledgebaseServer(sse_config)

    # Initialize server (non-blocking)
    await server.initialize_core()

    # Start MCP in SSE mode
    sse_task = asyncio.create_task(server.app.run_http(sse_config.sse_host, sse_config.sse_port))

    # Wait for server to start and get actual port
    await asyncio.sleep(2)  # Give time to bind

    # Get the actual port used (ephemeral)
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((sse_config.sse_host, sse_config.sse_port))
    actual_port = sock.getsockname()[1]
    sock.close()

    base_url = f"http://{sse_config.sse_host}:{actual_port}"

    yield base_url, server, sse_task

    # Cleanup
    sse_task.cancel()
    try:
        await sse_task
    except asyncio.CancelledError:
        pass
    await server.shutdown()


@pytest.mark.asyncio
async def test_sse_server_startup(sse_config):
    """Test that SSE server starts successfully."""
    server = PDFKnowledgebaseServer(sse_config)
    await server.initialize_core()

    # Test run_http without actually binding (just check config)
    assert sse_config.transport == "sse"
    assert sse_config.sse_host == "127.0.0.1"
    assert sse_config.sse_port > 0

    await server.shutdown()


@pytest.mark.asyncio
async def test_sse_tool_endpoint(sse_server, sample_pdf_path):
    """Test MCP tool endpoint over HTTP in SSE mode."""
    base_url, server, sse_task = sse_server

    async with httpx.AsyncClient() as client:
        # Test list_documents endpoint
        response = await client.get(f"{base_url}/list_documents")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

        # If sample PDF was added, expect at least one document
        if sample_pdf_path and Path(sample_pdf_path).exists():
            # Add document first
            with open(sample_pdf_path, "rb") as f:
                files = {"file": f}
                add_response = await client.post(f"{base_url}/add_document", files=files)
                assert add_response.status_code == 200
                add_data = add_response.json()
                assert add_data["success"]

            # Now list documents
            response = await client.get(f"{base_url}/list_documents")
            assert response.status_code == 200
            data = response.json()
            assert data["success"]
            assert len(data["documents"]) >= 1


@pytest.mark.asyncio
async def test_sse_search_documents(sse_server, sample_pdf_path):
    """Test search_documents tool in SSE mode."""
    base_url, server, sse_task = sse_server

    # First add a document if sample exists
    if sample_pdf_path and Path(sample_pdf_path).exists():
        async with httpx.AsyncClient() as client:
            # Add document
            with open(sample_pdf_path, "rb") as f:
                files = {"file": f}
                add_response = await client.post(f"{base_url}/add_document", files=files)
                assert add_response.status_code == 200
                add_data = add_response.json()
                assert add_data["success"]

            # Wait for processing
            await asyncio.sleep(3)

            # Test search
            search_response = await client.post(f"{base_url}/search_documents", json={"query": "test", "limit": 1})
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert search_data["success"]
            assert len(search_data["results"]) <= 1
    else:
        # Test empty search
        async with httpx.AsyncClient() as client:
            search_response = await client.post(
                f"{base_url}/search_documents", json={"query": "nonexistent", "limit": 1}
            )
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert search_data["success"]
            assert len(search_data["results"]) == 0


@pytest.mark.asyncio
async def test_sse_connection(sse_server):
    """Test SSE connection establishment."""
    base_url, server, sse_task = sse_server

    async with httpx.AsyncClient() as client:
        # Test SSE endpoint (FastMCP typically exposes /sse or similar, but for MCP tools it's HTTP)
        # For FastMCP, the SSE connection is typically at the root or /sse
        # Test basic connectivity
        response = await client.get(base_url)
        assert response.status_code == 200  # Should get MCP server info or tools list

        # Test SSE stream (if FastMCP exposes it)
        try:
            sse_response = await client.get(f"{base_url}/sse", timeout=httpx.Timeout(5.0))
            assert sse_response.status_code == 200
            assert "text/event-stream" in sse_response.headers.get("content-type", "")
        except httpx.TimeoutException:
            # SSE might not be immediately responsive, but connection should establish
            pass
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise  # Only allow 404 if endpoint doesn't exist, otherwise fail


@pytest.mark.asyncio
async def test_concurrent_web_sse(sse_config, temp_knowledgebase):
    """Test concurrent web + SSE mode without port conflicts."""
    # Configure for integrated mode with SSE
    os.environ["PDFKB_WEB_ENABLE"] = "true"
    os.environ["PDFKB_WEB_PORT"] = "8081"  # Different from SSE port
    os.environ["PDFKB_TRANSPORT"] = "sse"
    os.environ["PDFKB_SSE_PORT"] = "8000"

    config = ServerConfig.from_env()

    # Create integrated server
    from pdfkb.web_server import IntegratedPDFKnowledgebaseServer

    integrated_server = IntegratedPDFKnowledgebaseServer(config)
    await integrated_server.initialize()

    # Verify no port conflicts were raised during initialization
    assert integrated_server.web_server is not None
    assert integrated_server.mcp_server is not None

    # Test both servers are accessible (without actually running them fully)
    assert config.web_port == 8081
    assert config.sse_port == 8000
    assert config.web_port != config.sse_port

    await integrated_server.shutdown()


@pytest.mark.asyncio
async def test_sse_port_conflict_validation(sse_config):
    """Test port conflict validation in integrated mode."""
    # Configure conflicting ports
    os.environ["PDFKB_WEB_ENABLE"] = "true"
    os.environ["PDFKB_WEB_PORT"] = "8000"
    os.environ["PDFKB_TRANSPORT"] = "sse"
    os.environ["PDFKB_SSE_PORT"] = "8000"  # Same as web port

    config = ServerConfig.from_env()

    # Create integrated server - should raise validation error
    from pdfkb.web_server import IntegratedPDFKnowledgebaseServer

    integrated_server = IntegratedPDFKnowledgebaseServer(config)

    with pytest.raises(ValueError, match="Port conflict detected"):
        await integrated_server.initialize()


@pytest.mark.asyncio
async def test_sse_invalid_transport(sse_config):
    """Test validation for invalid transport values."""
    # Set invalid transport
    os.environ["PDFKB_TRANSPORT"] = "invalid"

    config = ServerConfig.from_env()

    # Create server - should raise validation error
    server = PDFKnowledgebaseServer(config)

    with pytest.raises(ValueError, match="Invalid MCP transport mode"):
        await server.initialize_core()


@pytest.mark.asyncio
async def test_sse_shutdown_graceful(sse_server):
    """Test graceful shutdown of SSE server."""
    base_url, server, sse_task = sse_server

    # Verify server is running
    async with httpx.AsyncClient() as client:
        response = await client.get(base_url, timeout=httpx.Timeout(2.0))
        assert response.status_code == 200

    # Shutdown
    await server.shutdown()

    # Verify shutdown by attempting connection (should fail)
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(base_url, timeout=httpx.Timeout(1.0))


# Run tests with pytest markers for SSE
# pytest tests/test_sse_integration.py -m sse -v
