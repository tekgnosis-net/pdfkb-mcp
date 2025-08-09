"""Web status service for system status and statistics."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from ...config import ServerConfig
from ...vector_store import VectorStore
from ..models.web_models import ConfigOverviewResponse, HealthCheckResponse, StatusResponse

logger = logging.getLogger(__name__)


class WebStatusService:
    """Service for system status and configuration information."""

    def __init__(
        self,
        config: ServerConfig,
        vector_store: VectorStore,
        document_cache: Dict[str, Any],
        start_time: float,
    ):
        """Initialize the web status service.

        Args:
            config: Server configuration
            vector_store: Vector storage service
            document_cache: Document metadata cache
            start_time: Server start timestamp
        """
        self.config = config
        self.vector_store = vector_store
        self.document_cache = document_cache
        self.start_time = start_time
        self._version = self._get_version()

    def _get_version(self) -> str:
        """Get application version from package info."""
        try:
            # Try to read version from package metadata
            import importlib.metadata

            return importlib.metadata.version("pdfkb-mcp")
        except Exception:
            # Fallback version
            return "0.1.0"

    async def get_status(self) -> StatusResponse:
        """Get comprehensive system status.

        Returns:
            StatusResponse with system status information
        """
        try:
            # Calculate uptime
            uptime = time.time() - self.start_time

            # Get vector store statistics
            documents_count = await self.vector_store.get_document_count()
            chunks_count = await self.vector_store.get_chunk_count()

            # Get storage statistics
            storage_stats = await self._get_storage_statistics()

            # Get processing statistics
            processing_stats = await self._get_processing_statistics()

            # Build configuration summary
            config_summary = {
                "embedding_model": self.config.embedding_model,
                "pdf_parser": self.config.pdf_parser,
                "pdf_chunker": self.config.pdf_chunker,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "vector_search_k": self.config.vector_search_k,
                "web_enabled": self.config.web_enabled,
                "web_port": self.config.web_port,
                "web_host": self.config.web_host,
            }

            # Build system statistics
            statistics = {
                "uptime_seconds": uptime,
                "uptime_hours": uptime / 3600,
                "documents_per_hour": documents_count / (uptime / 3600) if uptime > 0 else 0,
                "avg_chunks_per_document": chunks_count / documents_count if documents_count > 0 else 0,
                **storage_stats,
                **processing_stats,
            }

            return StatusResponse(
                status="healthy",
                version=self._version,
                uptime=uptime,
                documents_count=documents_count,
                chunks_count=chunks_count,
                knowledgebase_path=str(self.config.knowledgebase_path),
                cache_dir=str(self.config.cache_dir),
                configuration=config_summary,
                statistics=statistics,
            )

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return StatusResponse(
                status="error",
                version=self._version,
                uptime=time.time() - self.start_time,
                documents_count=0,
                chunks_count=0,
                knowledgebase_path=str(self.config.knowledgebase_path),
                cache_dir=str(self.config.cache_dir),
                configuration={"error": str(e)},
                statistics={"error": str(e)},
            )

    async def get_config_overview(self) -> ConfigOverviewResponse:
        """Get configuration overview.

        Returns:
            ConfigOverviewResponse with current configuration
        """
        try:
            return ConfigOverviewResponse(
                embedding_model=self.config.embedding_model,
                pdf_parser=self.config.pdf_parser,
                pdf_chunker=self.config.pdf_chunker,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                vector_search_k=self.config.vector_search_k,
                web_enabled=self.config.web_enabled,
                web_port=self.config.web_port,
                web_host=self.config.web_host,
                supported_extensions=self.config.supported_extensions,
            )

        except Exception as e:
            logger.error(f"Error getting configuration overview: {e}")
            raise

    async def get_health_check(self) -> HealthCheckResponse:
        """Get basic health check response.

        Returns:
            HealthCheckResponse with health status
        """
        try:
            # Perform basic health checks
            health_status = "ok"

            # Check if vector store is responsive
            try:
                await self.vector_store.get_chunk_count()
            except Exception as e:
                logger.error(f"Vector store health check failed: {e}")
                health_status = "degraded"

            # Check if required directories exist
            if not self.config.knowledgebase_path.exists():
                health_status = "degraded"

            if not self.config.cache_dir.exists():
                health_status = "degraded"

            return HealthCheckResponse(
                status=health_status,
                timestamp=datetime.now(timezone.utc),
                version=self._version,
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                version=self._version,
            )

    async def _get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage-related statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {}

            # Get knowledgebase directory size
            kb_path = self.config.knowledgebase_path
            if kb_path.exists():
                kb_size = sum(f.stat().st_size for f in kb_path.rglob("*") if f.is_file())
                stats["knowledgebase_size_bytes"] = kb_size
                stats["knowledgebase_size_mb"] = round(kb_size / (1024 * 1024), 2)

            # Get cache directory size
            cache_path = self.config.cache_dir
            if cache_path.exists():
                cache_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                stats["cache_size_bytes"] = cache_size
                stats["cache_size_mb"] = round(cache_size / (1024 * 1024), 2)

            # Get file counts
            if kb_path.exists():
                pdf_files = list(kb_path.rglob("*.pdf"))
                stats["pdf_file_count"] = len(pdf_files)
                stats["total_pdf_size_bytes"] = sum(f.stat().st_size for f in pdf_files)

            return stats

        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")
            return {"storage_error": str(e)}

    async def _get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing-related statistics.

        Returns:
            Dictionary with processing statistics
        """
        try:
            stats = {}

            # Count processing cache files
            processing_path = self.config.processing_path
            if processing_path.exists():
                parsing_cache_files = list(processing_path.rglob("parsing_result.json"))
                chunking_cache_files = list(processing_path.rglob("chunking_result.json"))

                stats["cached_parsing_results"] = len(parsing_cache_files)
                stats["cached_chunking_results"] = len(chunking_cache_files)

            # Get document type distribution
            doc_types = {}
            for document in self.document_cache.values():
                doc_type = document.metadata.get("document_type", "pdf")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            stats["document_types"] = doc_types

            # Get processing timestamps
            processing_times = []
            for document in self.document_cache.values():
                if "processing_timestamp" in document.metadata:
                    try:
                        timestamp = datetime.fromisoformat(document.metadata["processing_timestamp"])
                        processing_times.append(timestamp)
                    except Exception:
                        continue

            if processing_times:
                latest_processing = max(processing_times)
                stats["latest_processing"] = latest_processing.isoformat()
                stats["documents_processed_today"] = sum(
                    1 for t in processing_times if t.date() == datetime.now(timezone.utc).date()
                )

            return stats

        except Exception as e:
            logger.error(f"Error getting processing statistics: {e}")
            return {"processing_error": str(e)}

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics for monitoring.

        Returns:
            Dictionary with detailed system metrics
        """
        try:
            import sys

            import psutil

            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.config.knowledgebase_path.parent))

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "memory_percent": memory.percent,
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "disk_percent": round((disk.used / disk.total) * 100, 2),
                },
                "process": {
                    "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                    "memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                    "open_files": len(process.open_files()),
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                },
                "application": {
                    "version": self._version,
                    "uptime_seconds": time.time() - self.start_time,
                    "documents_count": len(self.document_cache),
                    "chunks_count": await self.vector_store.get_chunk_count(),
                },
            }

            return metrics

        except ImportError:
            logger.warning("psutil not available, returning limited metrics")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "application": {
                    "version": self._version,
                    "uptime_seconds": time.time() - self.start_time,
                    "documents_count": len(self.document_cache),
                },
                "note": "Install psutil for detailed system metrics",
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}
