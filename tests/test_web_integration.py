"""Integration tests for the web server with existing MCP functionality."""

import asyncio
import logging
import tempfile
from pathlib import Path

import pytest

from src.pdfkb.config import ServerConfig
from src.pdfkb.web_server import IntegratedPDFKnowledgebaseServer

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestWebIntegration:
    """Test web server integration with MCP functionality."""

    @pytest.fixture
    async def temp_config(self):
        """Create a temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Use patch.dict with clear=True to isolate from .env files
            from unittest.mock import patch

            with (
                patch.dict(
                    "os.environ",
                    {
                        "OPENAI_API_KEY": "sk-test-key-for-testing-only-not-real",
                        "KNOWLEDGEBASE_PATH": str(temp_path / "pdfs"),
                        "CACHE_DIR": str(temp_path / "cache"),
                        "WEB_ENABLED": "true",
                        "WEB_PORT": "8081",  # Use different port for testing
                        "WEB_HOST": "127.0.0.1",
                    },
                    clear=True,
                ),
                patch("src.pdfkb.config.load_dotenv"),
            ):  # Prevent reading .env file
                yield ServerConfig.from_env()

    async def test_integrated_server_initialization(self, temp_config):
        """Test that the integrated server can be initialized."""
        server = IntegratedPDFKnowledgebaseServer(temp_config)

        # Test that server can be created
        assert server is not None
        assert server.config == temp_config
        assert server.is_web_enabled is True

        # Test initialization (without actually running)
        try:
            await server.initialize()

            # Verify MCP server was initialized
            assert server.get_mcp_server() is not None

            # Verify web server was initialized
            assert server.get_web_server() is not None
            assert server.get_web_app() is not None

            # Check web URL properties
            assert server.web_url == f"http://{temp_config.web_host}:{temp_config.web_port}"
            assert server.docs_url == f"http://{temp_config.web_host}:{temp_config.web_port}/docs"

            logger.info("✓ Integrated server initialization test passed")

        finally:
            await server.shutdown()

    async def test_web_server_components_import(self):
        """Test that all web server components can be imported correctly."""
        try:
            # Test web models import
            # Test middleware import
            from src.pdfkb.web.middleware import (
                ErrorHandlingMiddleware,
                RequestLoggingMiddleware,
                setup_exception_handlers,
                setup_middleware,
            )
            from src.pdfkb.web.models.web_models import (
                DocumentDetailResponse,
                DocumentListResponse,
                SearchRequest,
                SearchResponse,
                StatusResponse,
                WebsocketEventType,
                WebsocketMessage,
            )

            # Test web server import
            from src.pdfkb.web.server import PDFKnowledgebaseWebServer

            # Test web services import
            from src.pdfkb.web.services import WebDocumentService, WebSearchService, WebSocketManager, WebStatusService

            # Test websocket handlers import
            from src.pdfkb.web.websocket_handlers import WebSocketEventHandler

            # Test main integration import
            from src.pdfkb.web_server import IntegratedPDFKnowledgebaseServer

            # Use imports to avoid F401 errors
            assert ErrorHandlingMiddleware is not None
            assert RequestLoggingMiddleware is not None
            assert setup_exception_handlers is not None
            assert setup_middleware is not None
            assert DocumentDetailResponse is not None
            assert DocumentListResponse is not None
            assert SearchRequest is not None
            assert SearchResponse is not None
            assert StatusResponse is not None
            assert WebsocketEventType is not None
            assert WebsocketMessage is not None
            assert PDFKnowledgebaseWebServer is not None
            assert WebDocumentService is not None
            assert WebSearchService is not None
            assert WebSocketManager is not None
            assert WebStatusService is not None
            assert WebSocketEventHandler is not None
            assert IntegratedPDFKnowledgebaseServer is not None

            logger.info("✓ All web server components imported successfully")

        except ImportError as e:
            pytest.fail(f"Failed to import web server components: {e}")

    async def test_configuration_validation(self):
        """Test that web configuration validation works correctly."""
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test valid configuration with isolated environment
            with (
                patch.dict(
                    "os.environ",
                    {
                        "OPENAI_API_KEY": "sk-test-key-for-testing-only-not-real",
                        "KNOWLEDGEBASE_PATH": str(temp_path / "pdfs"),
                        "CACHE_DIR": str(temp_path / "cache"),
                        "WEB_ENABLED": "true",
                        "WEB_PORT": "8081",
                        "WEB_HOST": "127.0.0.1",
                    },
                    clear=True,
                ),
                patch("src.pdfkb.config.load_dotenv"),
            ):  # Prevent reading .env file
                temp_config = ServerConfig.from_env()
                assert temp_config.web_enabled is True
                assert temp_config.web_port == 8081
                assert temp_config.web_host == "127.0.0.1"
                assert isinstance(temp_config.web_cors_origins, list)

            # Test invalid port configuration with isolated environment
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
                with pytest.raises(Exception):  # Should raise ConfigurationError
                    ServerConfig(
                        openai_api_key="sk-test-key",
                        web_port=-1,  # Invalid port
                    )

    async def test_web_models_creation(self):
        """Test that web models can be created and validated."""
        from src.pdfkb.web.models.web_models import (
            DocumentSummary,
            SearchRequest,
            StatusResponse,
            WebsocketEventType,
            WebsocketMessage,
        )

        # Test SearchRequest
        search_req = SearchRequest(
            query="test query",
            limit=10,
            min_score=0.5,
        )
        assert search_req.query == "test query"
        assert search_req.limit == 10
        assert search_req.min_score == 0.5

        # Test DocumentSummary
        doc_summary = DocumentSummary(
            id="doc_123",
            title="Test Document",
            filename="test.pdf",
            path="/path/to/test.pdf",
            file_size=1024,
            page_count=5,
            chunk_count=20,
            has_embeddings=True,
            checksum="abc123",
        )
        assert doc_summary.id == "doc_123"
        assert doc_summary.title == "Test Document"

        # Test WebsocketMessage
        ws_message = WebsocketMessage(
            event_type=WebsocketEventType.DOCUMENT_ADDED,
            data={"document_id": "doc_123"},
            message="Document added successfully",
        )
        assert ws_message.event_type == WebsocketEventType.DOCUMENT_ADDED
        assert ws_message.data["document_id"] == "doc_123"

        # Test StatusResponse
        status_resp = StatusResponse(
            version="1.0.0",
            uptime=3600.0,
            documents_count=10,
            chunks_count=100,
            knowledgebase_path="/path/to/kb",
            cache_dir="/path/to/cache",
        )
        assert status_resp.version == "1.0.0"
        assert status_resp.uptime == 3600.0

        logger.info("✓ Web models creation test passed")

    async def test_web_services_initialization(self, temp_config):
        """Test that web services can be initialized with MCP components."""
        from src.pdfkb.main import PDFKnowledgebaseServer
        from src.pdfkb.web.services import WebDocumentService, WebSearchService, WebSocketManager, WebStatusService

        # Initialize MCP server first
        mcp_server = PDFKnowledgebaseServer(temp_config)
        try:
            await mcp_server.initialize()

            # Test WebSocketManager
            ws_manager = WebSocketManager()
            assert ws_manager is not None

            # Test WebDocumentService
            doc_service = WebDocumentService(
                document_processor=mcp_server.document_processor,
                vector_store=mcp_server.vector_store,
                document_cache=mcp_server._document_cache,
                save_cache_callback=mcp_server._save_document_cache,
            )
            assert doc_service is not None

            # Test WebSearchService
            search_service = WebSearchService(
                vector_store=mcp_server.vector_store,
                embedding_service=mcp_server.embedding_service,
                document_cache=mcp_server._document_cache,
            )
            assert search_service is not None

            # Test WebStatusService
            status_service = WebStatusService(
                config=temp_config,
                vector_store=mcp_server.vector_store,
                document_cache=mcp_server._document_cache,
                start_time=asyncio.get_event_loop().time(),
            )
            assert status_service is not None

            logger.info("✓ Web services initialization test passed")

        finally:
            await mcp_server.shutdown()

    @pytest.mark.asyncio
    async def test_fastapi_app_creation(self, temp_config):
        """Test that FastAPI application can be created."""
        from src.pdfkb.main import PDFKnowledgebaseServer
        from src.pdfkb.web.server import PDFKnowledgebaseWebServer

        # Initialize MCP server
        mcp_server = PDFKnowledgebaseServer(temp_config)

        try:
            await mcp_server.initialize()

            # Create web server
            web_server = PDFKnowledgebaseWebServer(
                config=temp_config,
                document_processor=mcp_server.document_processor,
                vector_store=mcp_server.vector_store,
                embedding_service=mcp_server.embedding_service,
                document_cache=mcp_server._document_cache,
                save_cache_callback=mcp_server._save_document_cache,
            )

            # Get FastAPI app
            app = web_server.get_app()
            assert app is not None

            # Check that app has the expected routes
            routes = [route.path for route in app.routes]

            # Verify key endpoints exist
            expected_routes = [
                "/health",
                "/api/status",
                "/api/config",
                "/api/documents",
                "/api/search",
                "/ws",
            ]

            for route in expected_routes:
                # Check if route exists (exact match or parameterized)
                route_exists = any(
                    route in existing_route or existing_route.startswith(route) for existing_route in routes
                )
                assert route_exists, f"Expected route {route} not found in {routes}"

            logger.info("✓ FastAPI application creation test passed")

        finally:
            await mcp_server.shutdown()


# Run the tests
if __name__ == "__main__":

    async def run_tests():
        """Run basic integration tests manually."""
        test_instance = TestWebIntegration()

        # Create temporary config
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            from unittest.mock import patch

            with (
                patch.dict(
                    "os.environ",
                    {
                        "OPENAI_API_KEY": "sk-test-key-for-testing-only-not-real",
                        "KNOWLEDGEBASE_PATH": str(temp_path / "pdfs"),
                        "CACHE_DIR": str(temp_path / "cache"),
                        "WEB_ENABLED": "true",
                        "WEB_PORT": "8081",
                        "WEB_HOST": "127.0.0.1",
                    },
                    clear=True,
                ),
                patch("src.pdfkb.config.load_dotenv"),
            ):  # Prevent reading .env file
                temp_config = ServerConfig.from_env()

                print("Running web integration tests...")

                try:
                    # Test imports
                    await test_instance.test_web_server_components_import()
                    print("✓ Import test passed")

                    # Test models
                    await test_instance.test_web_models_creation()
                    print("✓ Models test passed")

                    # Test configuration
                    await test_instance.test_configuration_validation(temp_config)
                    print("✓ Configuration test passed")

                    # Test server initialization
                    await test_instance.test_integrated_server_initialization(temp_config)
                    print("✓ Server initialization test passed")

                    # Test services initialization
                    await test_instance.test_web_services_initialization(temp_config)
                    print("✓ Services initialization test passed")

                    # Test FastAPI app creation
                    await test_instance.test_fastapi_app_creation(temp_config)
                    print("✓ FastAPI app creation test passed")

                    print("\n🎉 All web integration tests passed!")

                except Exception as e:
                    print(f"❌ Test failed: {e}")
                    raise

    # Run the tests
    asyncio.run(run_tests())
