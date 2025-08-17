"""FastAPI server implementation for PDF Knowledgebase web interface."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..background_queue import BackgroundProcessingQueue
from ..config import ServerConfig
from ..document_processor import DocumentProcessor
from ..embeddings import EmbeddingService
from ..models import Document
from ..vector_store import VectorStore
from .models.web_models import (
    AddDocumentByPathRequest,
    ChunkResponse,
    ConfigOverviewResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentPreviewResponse,
    DocumentUploadResponse,
    HealthCheckResponse,
    JobCancelResponse,
    JobStatusResponse,
    PaginationParams,
    SearchRequest,
    SearchResponse,
    SearchSuggestionsResponse,
    StatusResponse,
)
from .services.web_document_service import WebDocumentService
from .services.web_search_service import WebSearchService
from .services.web_status_service import WebStatusService
from .services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class PDFKnowledgebaseWebServer:
    """FastAPI web server for PDF Knowledgebase."""

    def __init__(
        self,
        config: ServerConfig,
        document_processor: DocumentProcessor,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        document_cache: Dict[str, Document],
        save_cache_callback: Optional[callable] = None,
        background_queue: Optional[BackgroundProcessingQueue] = None,
    ):
        """Initialize the web server.

        Args:
            config: Server configuration
            document_processor: Document processing service
            vector_store: Vector storage service
            embedding_service: Embedding generation service
            document_cache: Document metadata cache
            save_cache_callback: Optional callback to save document cache
            background_queue: Optional background processing queue
        """
        self.config = config
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.document_cache = document_cache
        self.save_cache_callback = save_cache_callback
        self.background_queue = background_queue
        self.start_time = time.time()

        # Initialize WebSocket manager first
        self.websocket_manager = WebSocketManager()

        # Initialize services with queue and websocket manager
        self.document_service = WebDocumentService(
            document_processor,
            vector_store,
            document_cache,
            save_cache_callback,
            background_queue,
            self.websocket_manager,
        )
        self.search_service = WebSearchService(vector_store, embedding_service, document_cache)
        self.status_service = WebStatusService(config, vector_store, document_cache, self.start_time)

        # Create FastAPI app
        self.app = self._create_app()
        self._setup_routes()

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application.

        Returns:
            Configured FastAPI application
        """
        app = FastAPI(
            title="PDF Knowledgebase MCP API",
            description="RESTful API for PDF document management and semantic search (MCP Server)",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Add startup/shutdown event handlers
        app.add_event_handler("startup", self._on_startup)
        app.add_event_handler("shutdown", self._on_shutdown)

        # Setup static file serving
        self._setup_static_files(app)

        return app

    def _setup_static_files(self, app: FastAPI) -> None:
        """Setup static file serving for the web UI.

        Args:
            app: FastAPI application instance
        """
        # Get the webui directory path
        # In development: src/pdfkb/web/server.py -> src/webui
        # In installed package: site-packages/pdfkb/web/server.py -> site-packages/webui
        package_dir = Path(__file__).parent.parent.parent
        webui_dir = package_dir / "webui"

        if webui_dir.exists():
            # Mount static files (CSS, JS, etc.)
            app.mount("/static", StaticFiles(directory=str(webui_dir)), name="static")

            # Serve index.html at root
            @app.get("/", include_in_schema=False)
            async def serve_frontend():
                """Serve the frontend index.html file."""
                index_file = webui_dir / "index.html"
                if index_file.exists():
                    return FileResponse(str(index_file))
                else:
                    raise HTTPException(status_code=404, detail="Frontend not found")

            # Serve other static files (CSS, JS, components)
            @app.get("/styles.css", include_in_schema=False)
            async def serve_styles():
                """Serve the CSS file."""
                css_file = webui_dir / "styles.css"
                if css_file.exists():
                    return FileResponse(str(css_file), media_type="text/css")
                else:
                    raise HTTPException(status_code=404, detail="CSS file not found")

            @app.get("/app.js", include_in_schema=False)
            async def serve_app_js():
                """Serve the main app JavaScript file."""
                js_file = webui_dir / "app.js"
                if js_file.exists():
                    return FileResponse(str(js_file), media_type="application/javascript")
                else:
                    raise HTTPException(status_code=404, detail="App JS file not found")

            @app.get("/performance.js", include_in_schema=False)
            async def serve_performance_js():
                """Serve the performance optimization JavaScript file."""
                js_file = webui_dir / "performance.js"
                if js_file.exists():
                    return FileResponse(str(js_file), media_type="application/javascript")
                else:
                    raise HTTPException(status_code=404, detail="Performance JS file not found")

            @app.get("/components/{filename}", include_in_schema=False)
            async def serve_components(filename: str):
                """Serve component JavaScript files."""
                if not filename.endswith(".js"):
                    raise HTTPException(status_code=400, detail="Only JS files allowed")

                component_file = webui_dir / "components" / filename
                if component_file.exists():
                    return FileResponse(str(component_file), media_type="application/javascript")
                else:
                    raise HTTPException(status_code=404, detail=f"Component {filename} not found")

            logger.info(f"Static file serving configured for webui directory: {webui_dir}")
        else:
            logger.warning(f"WebUI directory not found: {webui_dir}")

    def _setup_routes(self) -> None:
        """Set up all API routes."""

        # Health check endpoint
        @self.app.get("/health", response_model=HealthCheckResponse, tags=["System"])
        async def health_check() -> HealthCheckResponse:
            """Health check endpoint."""
            try:
                return await self.status_service.get_health_check()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Service unhealthy")

        # System status endpoint
        @self.app.get("/api/status", response_model=StatusResponse, tags=["System"])
        async def get_status() -> StatusResponse:
            """Get comprehensive system status and statistics."""
            try:
                return await self.status_service.get_status()
            except Exception as e:
                logger.error(f"Failed to get status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Configuration overview endpoint
        @self.app.get("/api/config", response_model=ConfigOverviewResponse, tags=["System"])
        async def get_config() -> ConfigOverviewResponse:
            """Get current configuration overview."""
            try:
                return await self.status_service.get_config_overview()
            except Exception as e:
                logger.error(f"Failed to get config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Document listing endpoint
        @self.app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
        async def list_documents(
            page: int = Query(1, ge=1, description="Page number"),
            page_size: int = Query(20, ge=1, le=100, description="Items per page"),
            metadata_filter: Optional[str] = Query(None, description="JSON metadata filter"),
        ) -> DocumentListResponse:
            """List documents with pagination and optional filtering."""
            try:
                pagination = PaginationParams(page=page, page_size=page_size)

                # Parse metadata filter if provided
                filter_dict = None
                if metadata_filter:
                    import json

                    try:
                        filter_dict = json.loads(metadata_filter)
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail="Invalid metadata_filter JSON")

                return await self.document_service.list_documents(pagination, filter_dict)
            except Exception as e:
                logger.error(f"Failed to list documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Document detail endpoint
        @self.app.get("/api/documents/{document_id}", response_model=DocumentDetailResponse, tags=["Documents"])
        async def get_document_detail(
            document_id: str,
            include_chunks: bool = Query(False, description="Include document chunks"),
        ) -> DocumentDetailResponse:
            """Get detailed document information."""
            try:
                return await self.document_service.get_document_detail(document_id, include_chunks)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Failed to get document detail: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Document chunks endpoint
        @self.app.get("/api/documents/{document_id}/chunks", response_model=List[ChunkResponse], tags=["Documents"])
        async def get_document_chunks(document_id: str) -> List[ChunkResponse]:
            """Get all chunks for a document."""
            try:
                return await self.document_service.get_document_chunks(document_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Failed to get document chunks: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Document preview endpoint
        @self.app.get(
            "/api/documents/{document_id}/preview", response_model=DocumentPreviewResponse, tags=["Documents"]
        )
        async def get_document_preview(document_id: str) -> DocumentPreviewResponse:
            """Get document preview/content."""
            try:
                return await self.document_service.get_document_preview(document_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Failed to get document preview: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Document upload endpoint
        @self.app.post("/api/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
        async def upload_document(
            file: UploadFile = File(..., description="PDF file to upload"),
            metadata: Optional[str] = Form(None, description="JSON metadata"),
        ) -> DocumentUploadResponse:
            """Upload and process a new document."""
            try:
                # Validate file type
                if not file.filename.lower().endswith(".pdf"):
                    raise HTTPException(status_code=400, detail="Only PDF files are supported")

                # Parse metadata if provided
                metadata_dict = None
                if metadata:
                    import json

                    try:
                        metadata_dict = json.loads(metadata)
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

                # Read file content
                file_content = await file.read()

                # Only broadcast from server endpoint if no background queue (synchronous processing)
                if not self.background_queue:
                    await self.websocket_manager.broadcast_processing_started(file.filename)

                # Process document
                result = await self.document_service.upload_document(file_content, file.filename, metadata_dict)

                # Only broadcast result from server endpoint if no background queue (synchronous processing)
                if not self.background_queue:
                    if result.success:
                        if result.document_id:
                            # Get document for broadcasting
                            doc = self.document_cache.get(result.document_id)
                            if doc:
                                # Send processing completed with document info (combines both events)
                                document_data = {
                                    "document_id": doc.id,
                                    "title": doc.title,
                                    "filename": doc.filename,
                                    "path": doc.path,
                                    "chunks_created": result.chunks_created,
                                }
                                await self.websocket_manager.broadcast_processing_completed(document_data)
                    else:
                        await self.websocket_manager.broadcast_processing_failed(file.filename, result.error)

                return result

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to upload document: {e}")
                # Only broadcast error from server endpoint if no background queue
                if not self.background_queue:
                    await self.websocket_manager.broadcast_processing_failed(
                        file.filename if file else "unknown", str(e)
                    )
                raise HTTPException(status_code=500, detail=str(e))

        # Add document by path endpoint
        @self.app.post("/api/documents/add-path", response_model=DocumentUploadResponse, tags=["Documents"])
        async def add_document_by_path(request: AddDocumentByPathRequest) -> DocumentUploadResponse:
            """Add a document by file path."""
            try:
                filename = Path(request.path).name

                # Only broadcast from server endpoint if no background queue (synchronous processing)
                if not self.background_queue:
                    await self.websocket_manager.broadcast_processing_started(filename)

                # Process document
                result = await self.document_service.add_document_by_path(request.path, request.metadata)

                # Only broadcast result from server endpoint if no background queue (synchronous processing)
                if not self.background_queue:
                    if result.success:
                        if result.document_id:
                            # Get document for broadcasting
                            doc = self.document_cache.get(result.document_id)
                            if doc:
                                # Send processing completed with document info (combines both events)
                                document_data = {
                                    "document_id": doc.id,
                                    "title": doc.title,
                                    "filename": doc.filename,
                                    "path": doc.path,
                                    "chunks_created": result.chunks_created,
                                }
                                await self.websocket_manager.broadcast_processing_completed(document_data)
                    else:
                        await self.websocket_manager.broadcast_processing_failed(filename, result.error)

                return result

            except Exception as e:
                logger.error(f"Failed to add document by path: {e}")
                filename = Path(request.path).name if request.path else "unknown"
                # Only broadcast error from server endpoint if no background queue
                if not self.background_queue:
                    await self.websocket_manager.broadcast_processing_failed(filename, str(e))
                raise HTTPException(status_code=500, detail=str(e))

        # Delete document endpoint
        @self.app.delete("/api/documents/{document_id}", tags=["Documents"])
        async def remove_document(document_id: str) -> Dict[str, Any]:
            """Remove a document from the knowledgebase."""
            try:
                result = await self.document_service.remove_document(document_id)

                # Broadcast document removal
                if result.get("success"):
                    await self.websocket_manager.broadcast_document_removed(
                        document_id, result.get("document_path", "")
                    )

                return result

            except Exception as e:
                logger.error(f"Failed to remove document: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Job status endpoint
        @self.app.get("/api/jobs/{job_id}/status", response_model=JobStatusResponse, tags=["Jobs"])
        async def get_job_status(job_id: str) -> JobStatusResponse:
            """Get the status of a background processing job."""
            try:
                return await self.document_service.get_job_status(job_id)
            except Exception as e:
                logger.error(f"Failed to get job status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Job cancellation endpoint
        @self.app.delete("/api/jobs/{job_id}", response_model=JobCancelResponse, tags=["Jobs"])
        async def cancel_job(job_id: str) -> JobCancelResponse:
            """Cancel a background processing job."""
            try:
                return await self.document_service.cancel_job(job_id)
            except Exception as e:
                logger.error(f"Failed to cancel job: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Job statistics endpoint
        @self.app.get("/api/jobs/statistics", tags=["Jobs"])
        async def get_job_statistics() -> Dict[str, Any]:
            """Get background job processing statistics."""
            try:
                if not self.background_queue:
                    return {"success": False, "error": "Background queue not available"}

                stats = await self.background_queue.get_statistics()
                return {
                    "success": True,
                    "statistics": {status.name.lower(): count for status, count in stats.items()},
                    "total_jobs": sum(stats.values()),
                }
            except Exception as e:
                logger.error(f"Failed to get job statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Search endpoint
        @self.app.post("/api/search", response_model=SearchResponse, tags=["Search"])
        async def search_documents(request: SearchRequest) -> SearchResponse:
            """Perform vector similarity search."""
            try:
                result = await self.search_service.search(request)

                # Broadcast search performed
                await self.websocket_manager.broadcast_search_performed(request.query, result.total_results)

                return result

            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Search suggestions endpoint
        @self.app.get("/api/search/suggestions", response_model=SearchSuggestionsResponse, tags=["Search"])
        async def get_search_suggestions(
            query: str = Query(..., min_length=1, description="Query fragment for suggestions")
        ) -> SearchSuggestionsResponse:
            """Get search query suggestions."""
            try:
                return await self.search_service.get_search_suggestions(query)
            except Exception as e:
                logger.error(f"Failed to get search suggestions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            client_id = None
            try:
                client_id = await self.websocket_manager.connect(websocket)

                while True:
                    # Listen for messages from client
                    try:
                        data = await websocket.receive_text()
                        import json

                        message_data = json.loads(data)
                        await self.websocket_manager.handle_client_message(client_id, message_data)
                    except Exception as e:
                        logger.debug(f"WebSocket message handling error: {e}")
                        break

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {client_id}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if client_id:
                    await self.websocket_manager.disconnect(client_id)

        # Additional system endpoints
        @self.app.get("/api/metrics", tags=["System"])
        async def get_system_metrics() -> Dict[str, Any]:
            """Get detailed system metrics for monitoring."""
            try:
                return await self.status_service.get_system_metrics()
            except Exception as e:
                logger.error(f"Failed to get system metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/websocket/info", tags=["System"])
        async def get_websocket_info() -> Dict[str, Any]:
            """Get WebSocket connection information."""
            try:
                return {
                    "connection_count": await self.websocket_manager.get_connection_count(),
                    "connections": await self.websocket_manager.get_connection_info(),
                }
            except Exception as e:
                logger.error(f"Failed to get WebSocket info: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _on_startup(self) -> None:
        """FastAPI startup event handler."""
        logger.info("PDF Knowledgebase MCP web server starting up...")

        # Start periodic cleanup task for WebSocket connections
        asyncio.create_task(self._periodic_websocket_cleanup())

        # Start periodic job status broadcasting if background queue is available
        if self.background_queue:
            asyncio.create_task(self._periodic_job_status_broadcast())
            logger.info("Background queue integration enabled")

    async def _on_shutdown(self) -> None:
        """FastAPI shutdown event handler."""
        logger.info("PDF Knowledgebase MCP web server shutting down...")

        # Broadcast shutdown to all connected clients
        await self.websocket_manager.broadcast(
            "system_status_changed", {"status": "shutting_down"}, "Server is shutting down"
        )

        # Shutdown background queue if available
        if self.background_queue:
            logger.info("Shutting down background processing queue...")
            await self.background_queue.shutdown(wait=False)

    async def _periodic_websocket_cleanup(self) -> None:
        """Periodically clean up inactive WebSocket connections."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.websocket_manager.cleanup_inactive_connections()
            except Exception as e:
                logger.error(f"Error in WebSocket cleanup task: {e}")

    async def _periodic_job_status_broadcast(self) -> None:
        """Periodically broadcast job status updates to connected clients."""
        if not self.background_queue:
            return

        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds

                # Get current job statistics
                stats = await self.background_queue.get_statistics()
                if stats:
                    await self.websocket_manager.broadcast(
                        "system_status_changed",
                        {
                            "job_statistics": {status.name.lower(): count for status, count in stats.items()},
                            "total_jobs": sum(stats.values()),
                        },
                        "Job statistics updated",
                    )

            except Exception as e:
                logger.error(f"Error in job status broadcast task: {e}")

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance.

        Returns:
            FastAPI application instance
        """
        return self.app
