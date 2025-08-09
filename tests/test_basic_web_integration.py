"""Basic integration test for web server components without external dependencies."""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_imports():
    """Test that all web server components can be imported."""
    print("Testing web server component imports...")

    try:
        # Test web models import
        from pdfkb.web.models.web_models import (  # noqa: F401
            SearchRequest,
            StatusResponse,
            WebsocketEventType,
            WebsocketMessage,
        )

        print("✓ Web models imported successfully")

        # Test web services import
        from pdfkb.web.services import (  # noqa: F401
            WebDocumentService,
            WebSearchService,
            WebSocketManager,
            WebStatusService,
        )

        print("✓ Web services imported successfully")

        # Test web server import
        from pdfkb.web.server import PDFKnowledgebaseWebServer  # noqa: F401

        print("✓ Web server imported successfully")

        # Test middleware import
        from pdfkb.web.middleware import setup_exception_handlers, setup_middleware  # noqa: F401

        print("✓ Middleware imported successfully")

        # Test websocket handlers import
        from pdfkb.web.websocket_handlers import WebSocketEventHandler  # noqa: F401

        print("✓ WebSocket handlers imported successfully")

        # Test main integration import
        from pdfkb.web_server import IntegratedPDFKnowledgebaseServer  # noqa: F401

        print("✓ Integrated server imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


async def test_models_creation():
    """Test that web models can be created and work correctly."""
    print("Testing web models creation...")

    try:
        from pdfkb.web.models.web_models import (
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
        print("✓ SearchRequest model works")

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
        print("✓ DocumentSummary model works")

        # Test WebsocketMessage
        ws_message = WebsocketMessage(
            event_type=WebsocketEventType.DOCUMENT_ADDED,
            data={"document_id": "doc_123"},
            message="Document added successfully",
        )
        assert ws_message.event_type == WebsocketEventType.DOCUMENT_ADDED
        print("✓ WebsocketMessage model works")

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
        print("✓ StatusResponse model works")

        return True

    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


async def test_config_integration():
    """Test that web configuration integrates with existing config."""
    print("Testing configuration integration...")

    try:
        from pdfkb.config import ServerConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Set environment variables
            os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing-only-not-real"
            os.environ["KNOWLEDGEBASE_PATH"] = str(temp_path / "pdfs")
            os.environ["CACHE_DIR"] = str(temp_path / "cache")
            os.environ["WEB_ENABLED"] = "true"
            os.environ["WEB_PORT"] = "8081"
            os.environ["WEB_HOST"] = "127.0.0.1"

            try:
                config = ServerConfig.from_env()

                # Test web configuration fields
                assert hasattr(config, "web_enabled")
                assert hasattr(config, "web_port")
                assert hasattr(config, "web_host")
                assert hasattr(config, "web_cors_origins")

                assert config.web_enabled is True
                assert config.web_port == 8081
                assert config.web_host == "127.0.0.1"
                assert isinstance(config.web_cors_origins, list)

                print("✓ Web configuration fields integrated correctly")
                return True

            finally:
                # Clean up environment variables
                for key in ["OPENAI_API_KEY", "KNOWLEDGEBASE_PATH", "CACHE_DIR", "WEB_ENABLED", "WEB_PORT", "WEB_HOST"]:
                    os.environ.pop(key, None)

    except Exception as e:
        print(f"❌ Configuration integration error: {e}")
        return False


async def test_server_creation():
    """Test that integrated server can be created."""
    print("Testing integrated server creation...")

    try:
        from pdfkb.config import ServerConfig
        from pdfkb.web_server import IntegratedPDFKnowledgebaseServer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Set environment variables
            os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing-only-not-real"
            os.environ["KNOWLEDGEBASE_PATH"] = str(temp_path / "pdfs")
            os.environ["CACHE_DIR"] = str(temp_path / "cache")
            os.environ["WEB_ENABLED"] = "true"
            os.environ["WEB_PORT"] = "8082"
            os.environ["WEB_HOST"] = "127.0.0.1"

            try:
                config = ServerConfig.from_env()
                server = IntegratedPDFKnowledgebaseServer(config)

                # Test basic properties
                assert server is not None
                assert server.config == config
                assert server.is_web_enabled is True

                # Test URL properties
                expected_web_url = f"http://{config.web_host}:{config.web_port}"
                expected_docs_url = f"http://{config.web_host}:{config.web_port}/docs"

                assert server.web_url == expected_web_url
                assert server.docs_url == expected_docs_url

                print("✓ Integrated server created successfully")
                print(f"  Web URL: {server.web_url}")
                print(f"  Docs URL: {server.docs_url}")

                return True

            finally:
                # Clean up environment variables
                for key in ["OPENAI_API_KEY", "KNOWLEDGEBASE_PATH", "CACHE_DIR", "WEB_ENABLED", "WEB_PORT", "WEB_HOST"]:
                    os.environ.pop(key, None)

    except Exception as e:
        print(f"❌ Server creation error: {e}")
        return False


async def run_all_tests():
    """Run all basic integration tests."""
    print("🧪 Starting basic web integration tests...\n")

    tests = [
        ("Import Tests", test_imports),
        ("Model Creation Tests", test_models_creation),
        ("Configuration Integration Tests", test_config_integration),
        ("Server Creation Tests", test_server_creation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"📋 Running {test_name}:")
        try:
            result = await test_func()
            if result:
                print(f"✅ {test_name} PASSED\n")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED\n")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All basic integration tests passed!")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -e .")
        print("2. Run full integration tests: pytest tests/test_web_integration.py")
        print("3. Start the web server: pdfkb-web")
        print("4. Access API docs at: http://localhost:8080/docs")
        return True
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
