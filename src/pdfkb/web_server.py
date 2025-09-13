"""Integration point for running both MCP and web servers concurrently."""

import asyncio
import logging
from typing import Optional

from .background_queue import BackgroundProcessingQueue
from .config import ServerConfig
from .main import PDFKnowledgebaseServer
from .web.middleware import setup_exception_handlers, setup_middleware
from .web.server import PDFKnowledgebaseWebServer

logger = logging.getLogger(__name__)


class IntegratedPDFKnowledgebaseServer:
    """Integrated server that runs both MCP and web interfaces concurrently."""

    def __init__(self, config: Optional[ServerConfig] = None):
        """Initialize the integrated server.

        Args:
            config: Server configuration. If None, loads from environment.
        """
        self.config = config or ServerConfig.from_env()

        # Background queue for async processing - initialize early so it can be passed to MCP server
        self.background_queue: Optional[BackgroundProcessingQueue] = None

        # Initialize the core MCP server (background queue will be set during web server initialization)
        self.mcp_server: Optional[PDFKnowledgebaseServer] = None

        # Web server will be initialized after MCP server is ready
        self.web_server: Optional[PDFKnowledgebaseWebServer] = None
        self.web_app = None
        self._web_server_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize both MCP and web servers."""
        try:
            logger.info("Initializing integrated PDF Knowledgebase server...")

            # Validate configuration before initialization
            self._validate_configuration()

            # Initialize background queue first if web is enabled
            if self.config.web_enabled:
                logger.info(
                    f"Initializing background processing queue with {self.config.background_queue_workers} workers..."
                )
                self.background_queue = BackgroundProcessingQueue(
                    concurrency=self.config.background_queue_workers,  # Use configured number of workers
                    max_retries=3,
                    thread_pool_size=self.config.thread_pool_size,  # Use configured thread pool size
                )
                logger.info(
                    f"Background processing queue initialized with {self.config.background_queue_workers} workers "
                    f"and {self.config.thread_pool_size} thread pool size"
                )

            # Initialize MCP server with background queue (this sets up all core components)
            self.mcp_server = PDFKnowledgebaseServer(self.config, background_queue=self.background_queue)

            # First initialize core components (without FileMonitor)
            await self.mcp_server.initialize_core()

            # Initialize web server if enabled
            if self.config.web_enabled:
                await self._initialize_web_server()
                # Now initialize FileMonitor with web document service reference
                await self.mcp_server.initialize_file_monitor(web_document_service=self.web_server.document_service)
            else:
                logger.info("Web interface disabled in configuration")
                # Initialize FileMonitor without web document service
                await self.mcp_server.initialize_file_monitor()

            logger.info("Integrated PDF Knowledgebase server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize integrated server: {e}")
            # Cleanup any partially initialized components
            await self._cleanup_on_error()
            raise

    def _validate_configuration(self) -> None:
        """Validate configuration for integrated server operation."""
        # Validate MCP transport configuration
        if self.config.transport not in ["stdio", "http", "sse"]:
            raise ValueError(f"Invalid MCP transport mode: {self.config.transport}")

        # For integrated mode, validate web port (MCP is mounted within same app)
        if self.config.web_enabled and self.config.transport in ["http", "sse"]:
            # Validate web port (unified server)
            if self.config.web_port <= 0 or self.config.web_port > 65535:
                raise ValueError(f"Invalid web port: {self.config.web_port}")
            if not self.config.web_host:
                raise ValueError("Web host cannot be empty")

        elif self.config.web_enabled:
            # Web-only mode (no HTTP MCP)
            if self.config.web_port <= 0 or self.config.web_port > 65535:
                raise ValueError(f"Invalid web server port: {self.config.web_port}")
            if not self.config.web_host:
                raise ValueError("Web server host cannot be empty")

            # Check if web port is already in use
            import socket

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((self.config.web_host, self.config.web_port))
                    logger.debug(f"Web server port {self.config.web_port} is available")
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    raise ValueError(f"Web server port {self.config.web_port} is already in use") from e
                elif e.errno == 49:  # Can't assign requested address
                    raise ValueError(f"Cannot bind to host {self.config.web_host}") from e
                else:
                    raise ValueError(f"Failed to bind to {self.config.web_host}:{self.config.web_port}: {e}") from e

    async def _cleanup_on_error(self) -> None:
        """Cleanup partially initialized components on error."""
        try:
            if hasattr(self, "mcp_server") and self.mcp_server:
                await self.mcp_server.shutdown()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

    async def _initialize_web_server(self) -> None:
        """Initialize the web server using shared MCP components."""
        try:
            logger.info("Initializing web server...")

            # Check for web dependencies
            self._check_web_dependencies()

            # Background queue was already initialized - create web server using MCP server components
            self.web_server = PDFKnowledgebaseWebServer(
                config=self.config,
                document_processor=self.mcp_server.document_processor,
                vector_store=self.mcp_server.vector_store,
                embedding_service=self.mcp_server.embedding_service,
                document_cache=self.mcp_server._document_cache,
                save_cache_callback=self.mcp_server._save_document_cache,
                background_queue=self.background_queue,
            )

            # Get the FastAPI app
            self.web_app = self.web_server.get_app()

            # Mount FastMCP into FastAPI for unified ASGI serving
            if self.config.transport in ["http", "sse"]:
                # Determine the mount path based on transport type
                mount_path = "/mcp" if self.config.transport == "http" else "/sse"
                logger.info(f"Mounting MCP server at {mount_path} with {self.config.transport.upper()} transport")

                # Mount the MCP ASGI app into FastAPI, ensuring internal routes know the mount prefix
                self.web_app.mount(mount_path, self.mcp_server.get_http_app(path=mount_path))

                logger.info(
                    f"MCP endpoints available at: http://{self.config.web_host}:{self.config.web_port}{mount_path}/"
                )

            # Setup middleware and exception handlers
            setup_middleware(self.web_app, self.config)
            setup_exception_handlers(self.web_app)

            logger.info(f"Unified server configured to run on {self.config.web_host}:{self.config.web_port}")

        except Exception as e:
            logger.error(f"Failed to initialize web server: {e}")
            raise

    def _check_web_dependencies(self) -> None:
        """Check that required web dependencies are available."""
        missing_deps = []

        try:
            import fastapi  # noqa: F401
        except ImportError:
            missing_deps.append("fastapi")

        try:
            import hypercorn  # noqa: F401
        except ImportError:
            missing_deps.append("hypercorn")

        if missing_deps:
            raise ImportError(
                f"Missing required web server dependencies: {', '.join(missing_deps)}. "
                "Install with: pip install 'pdfkb-mcp[web]'"
            )

    async def run_mcp_only(self) -> None:
        """Run only the MCP server (without web interface)."""
        try:
            logger.info("Starting MCP-only server...")
            await self.initialize()
            await self.mcp_server.run()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down MCP server...")
            await self.shutdown()
        except Exception as e:
            logger.error(f"MCP server error: {e}")
            raise

    async def run_web_only(self) -> None:
        """Run only the web server (without MCP interface)."""
        if not self.config.web_enabled:
            raise ValueError("Web server is disabled in configuration")

        try:
            logger.info("Starting web-only server...")
            await self.initialize()

            if not self.web_app:
                raise RuntimeError("Web server not initialized")

            import hypercorn.asyncio
            from hypercorn.config import Config as HypercornConfig

            # Create hypercorn config (replaces uvicorn for better websockets 14+ support)
            hypercorn_config = HypercornConfig()
            hypercorn_config.bind = [f"{self.config.web_host}:{self.config.web_port}"]
            hypercorn_config.loglevel = self.config.log_level.lower()
            hypercorn_config.access_log_format = "%(h)s %(r)s %(s)s %(b)s %(D)s"
            hypercorn_config.accesslog = "-"  # Log to stdout

            # Run the web server
            await hypercorn.asyncio.serve(self.web_app, hypercorn_config)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down web server...")
            await self.shutdown()
        except Exception as e:
            logger.error(f"Web server error: {e}")
            raise

    async def run_integrated(self) -> None:
        """Run integrated server with MCP and web on single port (if web enabled)."""
        try:
            logger.info("Starting integrated server (MCP + Web)...")
            await self.initialize()

            if self.config.transport == "stdio":
                # STDIO mode - run only MCP server
                logger.info("Running MCP server in stdio mode...")
                await self.mcp_server.run()

            elif self.config.transport in ["http", "sse"]:
                if self.config.web_enabled:
                    # Integrated mode: dual server with MCP and web on adjacent ports
                    logger.info(f"Running integrated server with {self.config.transport.upper()} MCP and web...")
                    await self._run_unified_server()
                else:
                    # HTTP/SSE MCP only
                    logger.info(f"Running MCP server in {self.config.transport.upper()} mode only...")
                    await self.mcp_server.run()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down integrated server...")
            self._shutdown_event.set()
        except Exception as e:
            logger.error(f"Integrated server error: {e}")
            raise
        finally:
            await self.shutdown()

    async def _run_unified_server(self) -> None:
        """Run unified server with FastMCP mounted in FastAPI via Hypercorn."""
        try:
            # Determine endpoint path based on transport
            endpoint_path = "mcp" if self.config.transport == "http" else "sse"

            logger.info(f"🌐 Starting unified server ({self.config.transport.upper()} transport)...")
            logger.info(f"🌍 Web interface: http://{self.config.web_host}:{self.config.web_port}")
            logger.info(f"📡 MCP endpoints: http://{self.config.web_host}:{self.config.web_port}/{endpoint_path}/")
            logger.info(f"📚 API docs: http://{self.config.web_host}:{self.config.web_port}/docs")

            if not self.web_app:
                raise RuntimeError("Web server not initialized")

            import hypercorn.asyncio
            from hypercorn.config import Config as HypercornConfig

            # Create hypercorn config (unified server for both web and MCP)
            hypercorn_config = HypercornConfig()
            hypercorn_config.bind = [f"{self.config.web_host}:{self.config.web_port}"]
            hypercorn_config.loglevel = self.config.log_level.lower()
            hypercorn_config.access_log_format = "%(h)s %(r)s %(s)s %(b)s %(D)s"
            hypercorn_config.accesslog = "-"  # Log to stdout

            # Run the unified server
            await hypercorn.asyncio.serve(self.web_app, hypercorn_config)

        except Exception as e:
            logger.error(f"❌ Unified server error: {e}")
            self._shutdown_event.set()
            raise
        finally:
            logger.info("🔴 Unified server completed")

    async def shutdown(self) -> None:
        """Shutdown both servers gracefully."""
        try:
            logger.info("Shutting down integrated server...")

            # Set shutdown event
            self._shutdown_event.set()

            # Shutdown background queue first to stop processing new jobs
            if self.background_queue:
                logger.info("Shutting down background processing queue...")
                # Use shorter timeout for more responsive shutdown
                await self.background_queue.shutdown(wait=True, timeout=3.0)
                logger.info("Background processing queue shutdown complete")

            # No separate web server task in unified mode

            # Shutdown MCP server
            if self.mcp_server:
                await self.mcp_server.shutdown()

            logger.info("Integrated server shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def get_mcp_server(self) -> PDFKnowledgebaseServer:
        """Get the MCP server instance.

        Returns:
            MCP server instance
        """
        return self.mcp_server

    def get_web_server(self) -> Optional[PDFKnowledgebaseWebServer]:
        """Get the web server instance.

        Returns:
            Web server instance or None if not initialized
        """
        return self.web_server

    def get_web_app(self):
        """Get the FastAPI application instance.

        Returns:
            FastAPI application instance or None if not initialized
        """
        return self.web_app

    @property
    def is_web_enabled(self) -> bool:
        """Check if web server is enabled.

        Returns:
            True if web server is enabled
        """
        return self.config.web_enabled

    @property
    def web_url(self) -> Optional[str]:
        """Get the web server URL.

        Returns:
            Web server URL or None if not enabled
        """
        if self.config.web_enabled:
            return f"http://{self.config.web_host}:{self.config.web_port}"
        return None

    @property
    def docs_url(self) -> Optional[str]:
        """Get the API documentation URL.

        Returns:
            API docs URL or None if not enabled
        """
        if self.config.web_enabled:
            return f"http://{self.config.web_host}:{self.config.web_port}/docs"
        return None


async def run_integrated_server(config: Optional[ServerConfig] = None) -> None:
    """Run the integrated server with both MCP and web interfaces.

    Args:
        config: Server configuration. If None, loads from environment.
    """
    server = IntegratedPDFKnowledgebaseServer(config)
    await server.run_integrated()


async def run_web_only_server(config: Optional[ServerConfig] = None) -> None:
    """Run only the web server.

    Args:
        config: Server configuration. If None, loads from environment.
    """
    server = IntegratedPDFKnowledgebaseServer(config)
    await server.run_web_only()


def main_integrated():
    """Entry point for integrated server (MCP + Web)."""
    import argparse
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PDF Knowledgebase Integrated Server (MCP + Web)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  OPENAI_API_KEY          OpenAI API key (required)
  KNOWLEDGEBASE_PATH      Path to PDF directory (default: ./pdfs)
  CACHE_DIR              Cache directory (default: <KNOWLEDGEBASE_PATH>/.cache)
  WEB_ENABLED            Enable web interface (default: true)
  WEB_PORT               Web server port (default: 8080)
  WEB_HOST               Web server host (default: localhost)
  WEB_CORS_ORIGINS       CORS origins (default: http://localhost:3000,http://127.0.0.1:3000)
  PDF_PARSER             PDF parser to use (default: pymupdf4llm)
  PDF_CHUNKER            Text chunker to use (default: langchain)
  LOG_LEVEL              Logging level (default: INFO)

Examples:
  pdfkb-web --enable-web             # Run both MCP and web servers
  pdfkb-web --enable-web --port 9000 # Use custom web port with web enabled
  pdfkb-web --config myconfig.env    # Use custom config file
  pdfkb-web                          # Run MCP server only (web disabled by default)
        """,
    )

    parser.add_argument("--config", type=str, help="Path to environment configuration file")

    parser.add_argument("--port", type=int, help="Override web server port")

    parser.add_argument("--host", type=str, help="Override web server host")

    parser.add_argument("--enable-web", action="store_true", help="Enable web interface")
    parser.add_argument("--disable-web", action="store_true", help="Disable web interface (run MCP only)")

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Override logging level")

    parser.add_argument("--version", action="version", version=f'pdfkb-web {__import__("pdfkb").__version__}')

    args = parser.parse_args()

    # Load configuration from custom file if specified
    if args.config:
        from dotenv import load_dotenv

        load_dotenv(args.config, override=True)
        logger.info(f"Loaded configuration from: {args.config}")

    # Load main configuration
    config = ServerConfig.from_env()

    # Apply command line overrides
    if args.port:
        config.web_port = args.port
    if args.host:
        config.web_host = args.host
    if args.enable_web:
        config.web_enabled = True
    if args.disable_web:
        config.web_enabled = False
    if args.log_level:
        config.log_level = args.log_level

    # Configure logging
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting integrated PDF Knowledgebase server...")
    logger.info(f"Version: {__import__('pdfkb').__version__}")
    logger.info("MCP server: enabled")
    logger.info(f"Web server: {'enabled' if config.web_enabled else 'disabled'}")
    logger.info(f"Configuration: {config.knowledgebase_path}")
    logger.info(f"Cache directory: {config.cache_dir}")

    if config.web_enabled:
        logger.info(f"Web interface will be available at: http://{config.web_host}:{config.web_port}")
        logger.info(f"API documentation will be available at: http://{config.web_host}:{config.web_port}/docs")

    try:
        asyncio.run(run_integrated_server(config))
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


def main_web_only():
    """Entry point for web-only server."""
    import argparse
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PDF Knowledgebase Web-Only Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  OPENAI_API_KEY          OpenAI API key (required)
  KNOWLEDGEBASE_PATH      Path to PDF directory (default: ./pdfs)
  CACHE_DIR              Cache directory (default: <KNOWLEDGEBASE_PATH>/.cache)
  WEB_PORT               Web server port (default: 8080)
  WEB_HOST               Web server host (default: localhost)
  WEB_CORS_ORIGINS       CORS origins (default: http://localhost:3000,http://127.0.0.1:3000)
  PDF_PARSER             PDF parser to use (default: pymupdf4llm)
  PDF_CHUNKER            Text chunker to use (default: langchain)
  LOG_LEVEL              Logging level (default: INFO)

Examples:
  pdfkb-web-only                     # Run web server only
  pdfkb-web-only --port 9000         # Use custom web port
  pdfkb-web-only --config myconfig.env  # Use custom config file
        """,
    )

    parser.add_argument("--config", type=str, help="Path to environment configuration file")

    parser.add_argument("--port", type=int, help="Override web server port")

    parser.add_argument("--host", type=str, help="Override web server host")

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Override logging level")

    parser.add_argument("--version", action="version", version=f'pdfkb-web-only {__import__("pdfkb").__version__}')

    args = parser.parse_args()

    # Load configuration from custom file if specified
    if args.config:
        from dotenv import load_dotenv

        load_dotenv(args.config, override=True)
        logger.info(f"Loaded configuration from: {args.config}")

    # Load main configuration
    config = ServerConfig.from_env()

    # Apply command line overrides
    if args.port:
        config.web_port = args.port
    if args.host:
        config.web_host = args.host
    if args.log_level:
        config.log_level = args.log_level

    # Force web to be enabled for web-only mode
    config.web_enabled = True

    # Configure logging
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting web-only PDF Knowledgebase server...")
    logger.info(f"Version: {__import__('pdfkb').__version__}")
    logger.info(f"Configuration: {config.knowledgebase_path}")
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info(f"Web interface will be available at: http://{config.web_host}:{config.web_port}")
    logger.info(f"API documentation will be available at: http://{config.web_host}:{config.web_port}/docs")

    try:
        asyncio.run(run_web_only_server(config))
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "web-only":
        main_web_only()
    else:
        main_integrated()
