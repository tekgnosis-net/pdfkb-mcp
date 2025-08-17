"""Advanced file system monitoring for document directory changes with metadata persistence."""

import asyncio
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .background_queue import BackgroundProcessingQueue, Job, JobType, Priority
from .config import ServerConfig
from .exceptions import FileMonitorError, FileSystemError

logger = logging.getLogger(__name__)


class FileEventDebouncer:
    """Debounces rapid file system events to avoid processing the same file multiple times."""

    def __init__(self, delay: float = 0.5):
        """Initialize the debouncer.

        Args:
            delay: Delay in seconds before processing events.
        """
        self.delay = delay
        self.pending_events: Dict[str, Tuple[str, float]] = {}
        self.lock = asyncio.Lock()

    async def add_event(self, file_path: str, event_type: str) -> bool:
        """Add an event to the debouncer.

        Args:
            file_path: Path to the file.
            event_type: Type of event (created, modified, deleted).

        Returns:
            True if event should be processed immediately, False if debounced.
        """
        async with self.lock:
            current_time = time.time()

            # If we have a pending event for this file, update it
            if file_path in self.pending_events:
                self.pending_events[file_path] = (event_type, current_time)
                return False

            # Add new event
            self.pending_events[file_path] = (event_type, current_time)
            return True

    async def get_ready_events(self) -> List[Tuple[str, str]]:
        """Get events that are ready to be processed.

        Returns:
            List of (file_path, event_type) tuples ready for processing.
        """
        async with self.lock:
            current_time = time.time()
            ready_events = []
            expired_files = []

            for file_path, (event_type, event_time) in self.pending_events.items():
                if current_time - event_time >= self.delay:
                    ready_events.append((file_path, event_type))
                    expired_files.append(file_path)

            # Remove processed events
            for file_path in expired_files:
                del self.pending_events[file_path]

            return ready_events


class FileMetadata:
    """Represents metadata for a tracked file."""

    def __init__(
        self,
        path: str,
        checksum: str,
        last_modified: float,
        file_size: int,
        processed_time: Optional[float] = None,
        document_id: Optional[str] = None,
    ):
        """Initialize file metadata.

        Args:
            path: File path.
            checksum: SHA-256 checksum.
            last_modified: Last modified timestamp.
            file_size: File size in bytes.
            processed_time: When the file was last processed.
            document_id: Associated document ID if processed.
        """
        self.path = path
        self.checksum = checksum
        self.last_modified = last_modified
        self.file_size = file_size
        self.processed_time = processed_time
        self.document_id = document_id
        self.created_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "checksum": self.checksum,
            "last_modified": self.last_modified,
            "file_size": self.file_size,
            "processed_time": self.processed_time,
            "document_id": self.document_id,
            "created_time": self.created_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileMetadata":
        """Create from dictionary."""
        metadata = cls(
            path=data["path"],
            checksum=data["checksum"],
            last_modified=data["last_modified"],
            file_size=data["file_size"],
            processed_time=data.get("processed_time"),
            document_id=data.get("document_id"),
        )
        metadata.created_time = data.get("created_time", time.time())
        return metadata


class FileMonitor:
    """Advanced file system monitoring with watchdog integration and metadata persistence."""

    def __init__(
        self,
        config: ServerConfig,
        document_processor=None,
        vector_store=None,
        document_cache_callback=None,
        background_queue: Optional[BackgroundProcessingQueue] = None,
        web_document_service=None,
    ):
        """Initialize the file monitor.

        Args:
            config: Server configuration.
            document_processor: Document processing service.
            vector_store: Vector storage service.
            document_cache_callback: Callback function to update main server's document cache.
            background_queue: Optional background processing queue for non-blocking file processing.
            web_document_service: Optional WebDocumentService for creating in-progress document placeholders.
        """
        self.config = config
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.document_cache_callback = document_cache_callback
        self.background_queue = background_queue
        self.web_document_service = web_document_service  # Store the web document service
        if self.background_queue:
            logger.info("🎯 FILE MONITOR: Background queue is AVAILABLE - will use background processing")
        else:
            logger.warning(
                "⚠️ FILE MONITOR: Background queue is NOT AVAILABLE - will use synchronous processing (blocks server!)"
            )

        # File system monitoring
        self.observer = None
        self.event_handler = None
        self.is_running = False

        # Metadata persistence
        self.file_index: Dict[str, FileMetadata] = {}
        self.file_index_path = self.config.metadata_path / "file_index.json"
        self.index_lock = asyncio.Lock()

        # Event processing
        self.debouncer = FileEventDebouncer(delay=1.0)
        self.processing_queue = asyncio.Queue()
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.event_processor_task: Optional[asyncio.Task] = None

        # Processing state
        self.processing_files: Set[str] = set()
        self.failed_files: Dict[str, str] = {}  # file_path -> error_message
        self.processing_stats = {"processed": 0, "failed": 0, "skipped": 0}

        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="FileMonitor")

        # Directory exclusion settings
        self.excluded_directories = {"uploads", ".cache"}

    async def start_monitoring(self) -> None:
        """Start file system monitoring."""
        try:
            if self.is_running:
                logger.warning("File monitor is already running")
                return

            logger.info("Starting file monitor...")

            # Load existing file index
            await self.load_file_index()

            # Perform startup synchronization
            await self.startup_synchronization()

            # Start watchdog observer
            await self._start_watchdog()

            # Start background processors
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            self.event_processor_task = asyncio.create_task(self._event_processor())

            self.is_running = True
            logger.info(f"File monitor started, watching: {self.config.knowledgebase_path}")

        except Exception as e:
            raise FileMonitorError(f"Failed to start file monitor: {e}", "start", e)

    async def stop_monitoring(self) -> None:
        """Stop file system monitoring."""
        try:
            if not self.is_running:
                return

            logger.info("Stopping file monitor...")
            self.is_running = False

            # Stop background tasks
            if self.batch_processor_task:
                self.batch_processor_task.cancel()
                try:
                    await self.batch_processor_task
                except asyncio.CancelledError:
                    pass

            if self.event_processor_task:
                self.event_processor_task.cancel()
                try:
                    await self.event_processor_task
                except asyncio.CancelledError:
                    pass

            # Stop watchdog observer
            await self._stop_watchdog()

            # Save file index
            logger.info("Saving file index during shutdown...")
            try:
                logger.info("About to call save_file_index during shutdown")
                await self.save_file_index()
                logger.info("File index saved during shutdown")
                # Verify file exists after saving
                if self.file_index_path.exists():
                    file_size = self.file_index_path.stat().st_size
                    logger.info(f"Verified file index exists after shutdown save (size: {file_size} bytes)")
                else:
                    logger.error(f"File index not found after shutdown save at {self.file_index_path}")
            except Exception as e:
                logger.error(f"Error saving file index during shutdown: {e}")

            # Shutdown executor
            self.executor.shutdown(wait=True)

            logger.info("File monitor stopped")

        except Exception as e:
            logger.error(f"Error stopping file monitor: {e}")

    async def startup_synchronization(self) -> None:
        """Synchronize file system state with cached metadata on startup."""
        try:
            logger.info("Performing startup synchronization...")

            current_files = await self.scan_directory()
            cached_files = set(self.file_index.keys())

            # Normalize paths for comparison - resolve to absolute paths
            current_file_strs = {str(f.resolve()) for f in current_files}
            cached_file_resolved = {str(Path(f).resolve()): f for f in cached_files}

            # Find new files (exist on disk but not in cache)
            potentially_new_files = []
            for file_path in current_files:
                resolved_path = str(file_path.resolve())
                if resolved_path not in cached_file_resolved:
                    potentially_new_files.append(file_path)

            # Filter out files that already exist as completed documents in the document cache
            # This prevents treating already-processed documents as "new" during startup
            new_files = []
            if self.web_document_service and hasattr(self.web_document_service, "document_cache"):
                for file_path in potentially_new_files:
                    # Check if a document with this path already exists in the document cache
                    document_exists = False
                    for doc_id, document in self.web_document_service.document_cache.items():
                        if hasattr(document, "path") and str(Path(document.path).resolve()) == str(file_path.resolve()):
                            document_exists = True
                            # Update our file index with the existing document info
                            try:
                                stat = file_path.stat()
                                checksum = await self.get_file_checksum(file_path)
                                metadata = FileMetadata(
                                    path=str(file_path.resolve()),
                                    checksum=checksum,
                                    last_modified=stat.st_mtime,
                                    file_size=stat.st_size,
                                    processed_time=time.time(),
                                    document_id=doc_id,
                                )
                                await self.update_file_metadata(file_path, metadata)
                                logger.debug(f"Updated file index for existing document {file_path}")
                            except Exception as e:
                                logger.warning(f"Failed to update file index for existing document {file_path}: {e}")
                            break

                    if not document_exists:
                        new_files.append(file_path)
            else:
                # No document cache available, treat all as potentially new
                new_files = potentially_new_files

            # Find deleted files (in cache but not on disk)
            deleted_files = []
            for cached_file in cached_files:
                resolved_cached_path = str(Path(cached_file).resolve())
                if resolved_cached_path not in current_file_strs:
                    deleted_files.append(Path(cached_file))

            # Find modified files
            modified_files = []
            for file_path in current_files:
                resolved_path = str(file_path.resolve())
                if resolved_path in cached_file_resolved:
                    original_cached_path = cached_file_resolved[resolved_path]
                    if await self._file_changed(file_path, self.file_index[original_cached_path]):
                        modified_files.append(file_path)

            logger.info(
                f"Startup sync: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted"
            )

            # Queue changes for background processing (non-blocking)
            files_queued = 0
            for file_path in new_files:
                if self.background_queue:
                    logger.info(f"🚀 STARTUP SYNC: Queuing NEW file {file_path} for BACKGROUND processing")
                    await self._queue_file_for_processing(file_path, JobType.FILE_WATCHER)
                    files_queued += 1
                else:
                    logger.warning(
                        f"⚠️ STARTUP SYNC: No background queue, queuing NEW file {file_path} for SYNCHRONOUS processing"
                    )
                    await self._queue_file_processing(file_path, "created")

            for file_path in modified_files:
                if self.background_queue:
                    logger.info(f"🚀 STARTUP SYNC: Queuing MODIFIED file {file_path} for BACKGROUND processing")
                    await self._queue_file_for_processing(file_path, JobType.FILE_WATCHER)
                    files_queued += 1
                else:
                    logger.warning(
                        f"⚠️ STARTUP SYNC: No background queue, queuing MODIFIED file {file_path} "
                        f"for SYNCHRONOUS processing"
                    )
                    await self._queue_file_processing(file_path, "modified")

            for file_path in deleted_files:
                await self.remove_file(file_path)

            if files_queued > 0:
                logger.info(f"Queued {files_queued} files for background processing - server will remain responsive")

            logger.info("Startup synchronization completed - background processing will continue independently")

        except Exception as e:
            logger.error(f"Startup synchronization failed: {e}")
            raise FileMonitorError(f"Startup synchronization failed: {e}", "startup_sync", e)

    def _is_excluded_directory(self, file_path: Path) -> bool:
        """Check if a file path is in an excluded directory.

        Args:
            file_path: Path to check.

        Returns:
            True if the file is in an excluded directory.
        """
        try:
            # Convert to relative path from knowledgebase directory for comparison
            try:
                relative_path = file_path.relative_to(self.config.knowledgebase_path)
            except ValueError:
                # Path is not within knowledgebase directory, allow it
                return False

            # Check if any part of the path matches excluded directories
            for part in relative_path.parts:
                if part in self.excluded_directories:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error checking directory exclusion for {file_path}: {e}")
            return False  # Default to not excluded on error

    async def scan_directory(self) -> List[Path]:
        """Scan the knowledgebase directory for supported files.

        Returns:
            List of file paths found.
        """
        try:
            if not self.config.knowledgebase_path.exists():
                logger.warning(f"Knowledgebase directory does not exist: {self.config.knowledgebase_path}")
                return []

            files = []
            for ext in self.config.supported_extensions:
                pattern = f"**/*{ext}"
                for file_path in self.config.knowledgebase_path.rglob(pattern):
                    if file_path.is_file() and not self._is_excluded_directory(file_path):
                        files.append(file_path)

            logger.debug(
                f"Scanned directory: found {len(files)} files (excluded directories: {self.excluded_directories})"
            )
            return files

        except Exception as e:
            logger.error(f"Directory scan failed: {e}")
            return []

    async def process_new_file(self, file_path: Path) -> None:
        """Process a newly detected file.

        Args:
            file_path: Path to the new file.
        """
        try:
            if str(file_path) in self.processing_files:
                logger.debug(f"File already being processed: {file_path}")
                return

            # If background queue is available, queue the file for processing
            if self.background_queue:
                logger.info(f"🚀 FILE MONITOR: Queuing file {file_path} for BACKGROUND processing")
                await self._queue_file_for_processing(file_path, JobType.FILE_WATCHER)
                return

            # Fallback: Process synchronously
            logger.warning(
                f"⚠️ FILE MONITOR: No background queue available, processing {file_path} "
                f"SYNCHRONOUSLY (this will block the server!)"
            )
            await self._process_file_synchronously(file_path)

        except Exception as e:
            self.failed_files[str(file_path)] = str(e)
            self.processing_stats["failed"] += 1
            logger.error(f"Error processing new file {file_path}: {e}")

    async def _queue_file_for_processing(self, file_path: Path, job_type: JobType) -> None:
        """Queue a file for background processing.

        Args:
            file_path: Path to the file to process.
            job_type: Type of job for prioritization.
        """
        try:
            # Calculate file metadata before queuing
            checksum = await self.get_file_checksum(file_path)
            stat = file_path.stat()

            # Create job metadata
            job_metadata = {
                "file_path": str(file_path),
                "checksum": checksum,
                "file_size": stat.st_size,
                "last_modified": stat.st_mtime,
                "queued_time": time.time(),
            }

            # Queue the job with appropriate priority
            priority = Priority.NORMAL if job_type == JobType.FILE_WATCHER else Priority.HIGH

            job_id = await self.background_queue.add_job(
                job_type=job_type,
                metadata=job_metadata,
                priority=priority,
                processor=self._process_document_job,
            )

            # Create in-progress document placeholder if web service is available
            # But only if document doesn't already exist (to avoid showing existing docs as "Processing")
            if self.web_document_service:
                try:
                    # Check if document already exists in document cache
                    normalized_path = str(file_path.resolve())
                    existing_metadata = self.file_index.get(normalized_path)

                    # Skip creating in-progress document if:
                    # 1. File metadata shows it's already processed, OR
                    # 2. A document with same path exists in document cache
                    should_create_placeholder = True

                    if existing_metadata and existing_metadata.document_id:
                        # Check if document exists in cache
                        if (
                            hasattr(self.web_document_service, "document_cache")
                            and existing_metadata.document_id in self.web_document_service.document_cache
                        ):
                            should_create_placeholder = False
                            logger.debug(f"Skipping in-progress document for {file_path} - already exists in cache")

                    if should_create_placeholder:
                        in_progress_doc = self.web_document_service._create_in_progress_document(
                            job_id=job_id, filename=file_path.name, file_size=stat.st_size, temp_path=str(file_path)
                        )
                        in_progress_doc.path = str(file_path)  # Use actual path
                        self.web_document_service.in_progress_documents[in_progress_doc.id] = in_progress_doc
                        logger.debug(f"Created in-progress document {in_progress_doc.id} for file {file_path}")

                except Exception as e:
                    logger.error(f"Failed to create in-progress document for {file_path}: {e}")

            logger.info(f"Queued file for processing: {file_path} (job_id: {job_id})")

        except Exception as e:
            logger.error(f"Failed to queue file for processing {file_path}: {e}")
            # Fallback to synchronous processing on queue failure
            await self._process_file_synchronously(file_path)

    async def _process_document_job(self, job: Job) -> None:
        """Background job processor for PDF files.

        Args:
            job: Job instance containing file processing metadata.
        """
        file_path_str = job.metadata["file_path"]
        file_path = Path(file_path_str)
        job_id = job.job_id

        try:
            self.processing_files.add(file_path_str)
            logger.info(f"Starting background processing for: {file_path} (job_id: {job_id})")

            # Process the file
            if not self.document_processor:
                # Just track the file without processing
                checksum = job.metadata["checksum"]
                metadata = FileMetadata(
                    path=file_path_str,
                    checksum=checksum,
                    last_modified=job.metadata["last_modified"],
                    file_size=job.metadata["file_size"],
                )
                await self.update_file_metadata(file_path, metadata)
                self.processing_stats["skipped"] += 1
                logger.info(f"Tracked file without processing: {file_path}")

                # Remove in-progress document on skipped processing if web service is available
                if self.web_document_service:
                    self._remove_in_progress_document(job_id)
                return

            result = await self.document_processor.process_document(file_path)

            if result.success and result.document:
                # Add to vector store
                if self.vector_store:
                    await self.vector_store.add_document(result.document)

                # Update metadata
                metadata = FileMetadata(
                    path=file_path_str,
                    checksum=job.metadata["checksum"],
                    last_modified=job.metadata["last_modified"],
                    file_size=job.metadata["file_size"],
                    processed_time=time.time(),
                    document_id=result.document.id,
                )
                await self.update_file_metadata(file_path, metadata)

                # Update main server's document cache via callback
                if self.document_cache_callback:
                    await self.document_cache_callback(result.document)

                # Remove in-progress document on success if web service is available
                if self.web_document_service:
                    self._remove_in_progress_document(job_id)

                self.processing_stats["processed"] += 1
                logger.info(f"Successfully processed file in background: {file_path}")
                logger.debug(f"Document ID for processed file: {result.document.id}")
            else:
                error_msg = result.error or "Unknown processing error"
                self.failed_files[file_path_str] = error_msg
                self.processing_stats["failed"] += 1

                # Update in-progress document status on failure if web service is available
                if self.web_document_service:
                    self._update_in_progress_document_failed(job_id, error_msg)
                logger.error(f"Failed to process file {file_path}: {error_msg}")
                raise Exception(error_msg)

        except Exception as e:
            self.failed_files[file_path_str] = str(e)
            self.processing_stats["failed"] += 1
            logger.error(f"Error in background processing for {file_path}: {e}")
            raise
        finally:
            self.processing_files.discard(file_path_str)

    async def _process_file_synchronously(self, file_path: Path) -> None:
        """Process a file synchronously (fallback when no background queue).

        Args:
            file_path: Path to the file to process.
        """
        try:
            self.processing_files.add(str(file_path))

            try:
                # Calculate file metadata
                checksum = await self.get_file_checksum(file_path)
                stat = file_path.stat()

                # Process the file
                if self.document_processor:
                    result = await self.document_processor.process_document(file_path)

                    if result.success and result.document:
                        # Add to vector store
                        if self.vector_store:
                            await self.vector_store.add_document(result.document)

                        # Update metadata
                        metadata = FileMetadata(
                            path=str(file_path),
                            checksum=checksum,
                            last_modified=stat.st_mtime,
                            file_size=stat.st_size,
                            processed_time=time.time(),
                            document_id=result.document.id,
                        )
                        await self.update_file_metadata(file_path, metadata)

                        # Update main server's document cache via callback
                        if self.document_cache_callback:
                            await self.document_cache_callback(result.document)

                        self.processing_stats["processed"] += 1
                        logger.info(f"Successfully processed new file: {file_path}")
                        logger.debug(f"Document ID for processed file: {result.document.id}")
                    else:
                        error_msg = result.error or "Unknown processing error"
                        self.failed_files[str(file_path)] = error_msg
                        self.processing_stats["failed"] += 1
                        logger.error(f"Failed to process file {file_path}: {error_msg}")
                else:
                    # No processor available, just track the file
                    metadata = FileMetadata(
                        path=str(file_path),
                        checksum=checksum,
                        last_modified=stat.st_mtime,
                        file_size=stat.st_size,
                    )
                    await self.update_file_metadata(file_path, metadata)
                    self.processing_stats["skipped"] += 1
                    logger.info(f"Tracked file without processing: {file_path}")

            finally:
                self.processing_files.discard(str(file_path))

        except Exception as e:
            self.failed_files[str(file_path)] = str(e)
            self.processing_stats["failed"] += 1
            logger.error(f"Error processing file synchronously {file_path}: {e}")

    async def remove_file(self, file_path: Path) -> None:
        """Remove a deleted file from tracking and vector store.

        Args:
            file_path: Path to the removed file.
        """
        try:
            # Use normalized path for lookup
            normalized_path = str(file_path.resolve())

            # Get metadata before removal
            metadata = self.file_index.get(normalized_path)

            document_id = None
            if metadata and metadata.document_id:
                document_id = metadata.document_id

                # Remove from vector store
                if self.vector_store:
                    removed_count = await self.vector_store.delete_document(document_id)
                    logger.info(f"Removed {removed_count} chunks from vector store for deleted file: {file_path}")

                # Remove from main document cache via callback if available
                if self.web_document_service and hasattr(self.web_document_service, "document_cache"):
                    if document_id in self.web_document_service.document_cache:
                        del self.web_document_service.document_cache[document_id]
                        logger.info(f"Removed document {document_id} from document cache")

                        # Save the updated cache if callback available
                        if self.web_document_service.save_cache_callback:
                            try:
                                await self.web_document_service.save_cache_callback()
                                logger.debug(f"Saved document cache after removing {document_id}")
                            except Exception as e:
                                logger.warning(f"Failed to save document cache after removal: {e}")

            # Remove from file index
            async with self.index_lock:
                if normalized_path in self.file_index:
                    del self.file_index[normalized_path]
                    await self.save_file_index()

            # Remove from failed files if present
            self.failed_files.pop(normalized_path, None)

            if document_id:
                logger.info(f"Removed file {file_path} (document {document_id}) from all tracking systems")
            else:
                logger.info(f"Removed file {file_path} from tracking (no associated document)")

        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")

    async def get_processed_files(self) -> List[dict]:
        """Get list of all processed files.

        Returns:
            List of file information dictionaries.
        """
        try:
            async with self.index_lock:
                files = []
                for metadata in self.file_index.values():
                    file_info = {
                        "path": metadata.path,
                        "checksum": metadata.checksum,
                        "last_modified": datetime.fromtimestamp(metadata.last_modified).isoformat(),
                        "file_size": metadata.file_size,
                        "processed": metadata.processed_time is not None,
                        "document_id": metadata.document_id,
                    }

                    if metadata.processed_time:
                        file_info["processed_time"] = datetime.fromtimestamp(metadata.processed_time).isoformat()

                    files.append(file_info)

                return files

        except Exception as e:
            logger.error(f"Error getting processed files: {e}")
            return []

    async def is_file_processed(self, file_path: Path) -> bool:
        """Check if a file has been processed.

        Args:
            file_path: Path to check.

        Returns:
            True if file has been processed.
        """
        normalized_path = str(file_path.resolve())
        metadata = self.file_index.get(normalized_path)
        return metadata is not None and metadata.processed_time is not None

    async def get_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file.

        Args:
            file_path: Path to the file.

        Returns:
            SHA-256 checksum as hex string.
        """
        try:

            def _calculate_checksum() -> str:
                hash_sha256 = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_sha256.update(chunk)
                return hash_sha256.hexdigest()

            # Run checksum calculation in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _calculate_checksum)

        except Exception as e:
            raise FileSystemError(f"Failed to calculate checksum: {e}", str(file_path), e)

    async def load_file_index(self) -> None:
        """Load file index from persistent storage."""
        try:
            logger.info(f"Looking for file index at: {self.file_index_path}")
            # Check if directory exists and is accessible
            directory = self.file_index_path.parent
            readable_status = os.access(directory, os.R_OK) if directory.exists() else "N/A"
            logger.info(f"Checking directory: {directory}, exists: {directory.exists()}, readable: {readable_status}")
            if self.file_index_path.exists():
                logger.info(f"Found file index, loading from: {self.file_index_path}")

                def _load_index() -> Dict[str, Any]:
                    with open(self.file_index_path, "r", encoding="utf-8") as f:
                        return json.load(f)

                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(self.executor, _load_index)

                async with self.index_lock:
                    self.file_index = {}
                    for file_path, metadata_dict in data.items():
                        try:
                            self.file_index[file_path] = FileMetadata.from_dict(metadata_dict)
                        except Exception as e:
                            logger.warning(f"Skipping corrupted metadata for {file_path}: {e}")

                logger.info(f"Loaded file index with {len(self.file_index)} entries")
            else:
                self.file_index = {}
                logger.info(f"No existing file index found at {self.file_index_path}, starting fresh")

        except Exception as e:
            logger.error(f"Failed to load file index: {e}")
            self.file_index = {}

    def _save_file_index_sync(self, data: Dict[str, Dict]) -> None:
        """Save file index to persistent storage (synchronous, no locking)."""
        logger.info(f"Saving file index with {len(data)} entries to {self.file_index_path}")

        try:
            # Ensure metadata directory exists
            self.file_index_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/exists: {self.file_index_path.parent}")

            # Check if directory is writable
            if not os.access(self.file_index_path.parent, os.W_OK):
                logger.error(f"Directory is not writable: {self.file_index_path.parent}")
                return

            # Write to temporary file first
            temp_path = self.file_index_path.with_suffix(".tmp")
            logger.info(f"Writing to temp file: {temp_path}")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully wrote temp file: {temp_path}")

            # Atomic rename
            logger.info(f"Renaming {temp_path} to {self.file_index_path}")
            temp_path.replace(self.file_index_path)
            logger.info(f"Successfully saved file index to {self.file_index_path}")

            # Verify file was saved
            if self.file_index_path.exists():
                file_size = self.file_index_path.stat().st_size
                logger.info(f"File index saved successfully (size: {file_size} bytes)")
            else:
                logger.error("File index not found after save operation")

        except Exception as e:
            logger.error(f"Error in save operation: {e}")
            raise

    async def save_file_index(self) -> None:
        """Save file index to persistent storage."""
        try:
            async with self.index_lock:
                data = {path: metadata.to_dict() for path, metadata in self.file_index.items()}

            self._save_file_index_sync(data)

        except Exception as e:
            logger.error(f"Failed to save file index: {e}")

    async def update_file_metadata(self, file_path: Path, metadata: FileMetadata) -> None:
        """Update metadata for a file.

        Args:
            file_path: File path.
            metadata: New metadata.
        """
        try:
            async with self.index_lock:
                # Store normalized path (resolved to absolute path)
                normalized_path = str(file_path.resolve())
                self.file_index[normalized_path] = metadata
                logger.info(
                    f"Updated metadata for {file_path} (normalized: {normalized_path}), "
                    f"total files: {len(self.file_index)}"
                )
                # Always save after updating metadata to ensure persistence
                logger.info(f"Saving file index after metadata update for {file_path}")
                data = {path: metadata.to_dict() for path, metadata in self.file_index.items()}
                self._save_file_index_sync(data)
                logger.info(f"Completed saving file index after metadata update for {file_path}")

        except Exception as e:
            logger.error(f"Failed to update file metadata for {file_path}: {e}")

    async def _start_watchdog(self) -> None:
        """Start the watchdog file system observer."""
        try:
            from watchdog.events import FileSystemEvent, FileSystemEventHandler
            from watchdog.observers import Observer

            class DocumentEventHandler(FileSystemEventHandler):
                def __init__(self, monitor: "FileMonitor"):
                    self.monitor = monitor
                    self.loop = asyncio.get_event_loop()

                def on_created(self, event: FileSystemEvent):
                    if not event.is_directory and self._is_supported_file(event.src_path):
                        if self.monitor.background_queue:
                            asyncio.run_coroutine_threadsafe(
                                self.monitor._queue_file_for_processing(Path(event.src_path), JobType.FILE_WATCHER),
                                self.loop,
                            )
                        else:
                            asyncio.run_coroutine_threadsafe(
                                self.monitor._queue_file_processing(Path(event.src_path), "created"),
                                self.loop,
                            )

                def on_modified(self, event: FileSystemEvent):
                    if not event.is_directory and self._is_supported_file(event.src_path):
                        if self.monitor.background_queue:
                            asyncio.run_coroutine_threadsafe(
                                self.monitor._queue_file_for_processing(Path(event.src_path), JobType.FILE_WATCHER),
                                self.loop,
                            )
                        else:
                            asyncio.run_coroutine_threadsafe(
                                self.monitor._queue_file_processing(Path(event.src_path), "modified"),
                                self.loop,
                            )

                def on_deleted(self, event: FileSystemEvent):
                    if not event.is_directory and self._is_supported_file(event.src_path):
                        asyncio.run_coroutine_threadsafe(self.monitor.remove_file(Path(event.src_path)), self.loop)

                def on_moved(self, event):
                    # Handle file moves as delete + create
                    if hasattr(event, "src_path") and self._is_supported_file(event.src_path):
                        asyncio.run_coroutine_threadsafe(self.monitor.remove_file(Path(event.src_path)), self.loop)

                    if hasattr(event, "dest_path") and self._is_supported_file(event.dest_path):
                        if self.monitor.background_queue:
                            asyncio.run_coroutine_threadsafe(
                                self.monitor._queue_file_for_processing(Path(event.dest_path), JobType.FILE_WATCHER),
                                self.loop,
                            )
                        else:
                            asyncio.run_coroutine_threadsafe(
                                self.monitor._queue_file_processing(Path(event.dest_path), "created"),
                                self.loop,
                            )

                def _is_supported_file(self, file_path: str) -> bool:
                    """Check if file has supported extension and is not in an excluded directory."""
                    path = Path(file_path)

                    # Check if file has supported extension
                    if path.suffix.lower() not in [ext.lower() for ext in self.monitor.config.supported_extensions]:
                        return False

                    # Check if file is in an excluded directory
                    if self.monitor._is_excluded_directory(path):
                        logger.debug(f"Excluding file in restricted directory: {path}")
                        return False

                    return True

            self.event_handler = DocumentEventHandler(self)
            self.observer = Observer()
            self.observer.schedule(self.event_handler, str(self.config.knowledgebase_path), recursive=True)

            # Start observer in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.observer.start)

            logger.info("Watchdog file system observer started")

        except ImportError:
            raise FileMonitorError(
                "Watchdog package not installed. Install with: pip install watchdog",
                "start_watchdog",
            )
        except Exception as e:
            raise FileMonitorError(f"Failed to start watchdog observer: {e}", "start_watchdog", e)

    async def _stop_watchdog(self) -> None:
        """Stop the watchdog file system observer."""
        try:
            if self.observer:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.observer.stop)
                await loop.run_in_executor(self.executor, self.observer.join)
                self.observer = None

            self.event_handler = None
            logger.info("Watchdog file system observer stopped")

        except Exception as e:
            logger.error(f"Error stopping watchdog observer: {e}")

    async def _queue_file_processing(self, file_path: Path, event_type: str) -> None:
        """Queue a file for processing with debouncing.

        Args:
            file_path: Path to the file.
            event_type: Type of event (created, modified, deleted).
        """
        try:
            # Add to debouncer
            should_process = await self.debouncer.add_event(str(file_path), event_type)

            if should_process:
                await self.processing_queue.put((file_path, event_type))

        except Exception as e:
            logger.error(f"Error queuing file processing for {file_path}: {e}")

    async def _batch_processor(self) -> None:
        """Background task that processes queued files in batches."""
        try:
            while self.is_running:
                try:
                    # Collect ready events from debouncer
                    ready_events = await self.debouncer.get_ready_events()

                    for file_path_str, event_type in ready_events:
                        await self.processing_queue.put((Path(file_path_str), event_type))

                    # Process events in small batches
                    batch = []
                    batch_size = 5

                    try:
                        # Try to get up to batch_size items with a timeout
                        for _ in range(batch_size):
                            try:
                                item = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                                batch.append(item)
                            except asyncio.TimeoutError:
                                break
                    except Exception:
                        pass

                    # Process the batch
                    if batch:
                        await self._process_batch(batch)

                    # Brief pause to avoid busy waiting
                    await asyncio.sleep(0.5)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in batch processor: {e}")
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass

    async def _event_processor(self) -> None:
        """Background task that handles debounced events."""
        try:
            while self.is_running:
                try:
                    # Check for ready events
                    ready_events = await self.debouncer.get_ready_events()

                    for file_path_str, event_type in ready_events:
                        file_path = Path(file_path_str)

                        if event_type in ["created", "modified"]:
                            if file_path.exists():
                                await self.process_new_file(file_path)
                        elif event_type == "deleted":
                            await self.remove_file(file_path)

                    await asyncio.sleep(0.5)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in event processor: {e}")
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass

    async def _process_batch(self, batch: List[Tuple[Path, str]]) -> None:
        """Process a batch of file events.

        Args:
            batch: List of (file_path, event_type) tuples.
        """
        try:
            for file_path, event_type in batch:
                try:
                    if event_type in ["created", "modified"]:
                        if file_path.exists():
                            await self.process_new_file(file_path)
                    elif event_type == "deleted":
                        await self.remove_file(file_path)

                except Exception as e:
                    logger.error(f"Error processing file {file_path} with event {event_type}: {e}")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

    async def _file_changed(self, file_path: Path, cached_metadata: FileMetadata) -> bool:
        """Check if a file has changed compared to cached metadata.

        Args:
            file_path: Path to the file.
            cached_metadata: Cached file metadata.

        Returns:
            True if file has changed.
        """
        try:
            if not file_path.exists():
                return True  # File was deleted

            stat = file_path.stat()

            # Check file size and modification time first (fast)
            if stat.st_size != cached_metadata.file_size or abs(stat.st_mtime - cached_metadata.last_modified) > 1.0:
                return True

            # If size and mtime are same, check checksum (slower but accurate)
            current_checksum = await self.get_file_checksum(file_path)
            return current_checksum != cached_metadata.checksum

        except Exception as e:
            logger.error(f"Error checking file changes for {file_path}: {e}")
            return True  # Assume changed on error

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary with processing statistics.
        """
        return {
            **self.processing_stats,
            "currently_processing": len(self.processing_files),
            "failed_files": len(self.failed_files),
            "tracked_files": len(self.file_index),
            "queue_size": self.processing_queue.qsize(),
            "is_running": self.is_running,
        }

    async def force_rescan(self) -> None:
        """Force a complete rescan of the directory."""
        try:
            logger.info("Starting forced rescan...")

            # Clear processing state
            self.processing_files.clear()
            self.failed_files.clear()

            # Perform startup synchronization (which handles the comparison)
            await self.startup_synchronization()

            logger.info("Forced rescan completed")

        except Exception as e:
            logger.error(f"Forced rescan failed: {e}")
            raise FileMonitorError(f"Forced rescan failed: {e}", "force_rescan", e)

    def _remove_in_progress_document(self, job_id: str) -> None:
        """Remove in-progress document by job ID."""
        if not self.web_document_service:
            return

        try:
            # Find in-progress document by job_id
            for doc_id, doc in list(self.web_document_service.in_progress_documents.items()):
                if doc.job_id == job_id:
                    del self.web_document_service.in_progress_documents[doc_id]
                    logger.info(f"📋 FILE MONITOR: Removed in-progress document {doc_id} after processing")
                    break
        except Exception as e:
            logger.error(f"Failed to remove in-progress document for job {job_id}: {e}")

    def _update_in_progress_document_failed(self, job_id: str, error_msg: str) -> None:
        """Update in-progress document status to failed."""
        if not self.web_document_service:
            return

        try:
            from .web.models.web_models import ProcessingStatus

            # Find and update in-progress document by job_id
            for doc_id, doc in self.web_document_service.in_progress_documents.items():
                if doc.job_id == job_id:
                    doc.processing_status = ProcessingStatus.FAILED
                    doc.processing_error = error_msg
                    logger.info(f"📋 FILE MONITOR: Updated in-progress document {doc_id} status to FAILED")
                    break
        except Exception as e:
            logger.error(f"Failed to update in-progress document status for job {job_id}: {e}")
