"""Web document service that wraps DocumentProcessor functionality."""

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...background_queue import BackgroundProcessingQueue, Job, JobStatus, JobType, Priority
from ...document_processor import DocumentProcessor
from ...models import Document
from ...vector_store import VectorStore
from ..models.web_models import (
    ChunkResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentPreviewResponse,
    DocumentSummary,
    DocumentUploadResponse,
    JobCancelResponse,
    JobStatusResponse,
    PaginationParams,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


class WebDocumentService:
    """Service for document management operations via web interface."""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_store: VectorStore,
        document_cache: Dict[str, Document],
        save_cache_callback: Optional[callable] = None,
        background_queue: Optional[BackgroundProcessingQueue] = None,
        websocket_manager: Optional[Any] = None,
    ):
        """Initialize the web document service.

        Args:
            document_processor: PDF processing service
            vector_store: Vector storage service
            document_cache: Document metadata cache
            save_cache_callback: Optional callback to save document cache
            background_queue: Optional background processing queue
            websocket_manager: Optional WebSocket manager for real-time updates
        """
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.document_cache = document_cache
        self.save_cache_callback = save_cache_callback
        self.background_queue = background_queue
        self.websocket_manager = websocket_manager

        # Store job metadata for status tracking
        self.job_metadata: Dict[str, Dict[str, Any]] = {}

        # Track in-progress documents (documents currently being processed)
        self.in_progress_documents: Dict[str, DocumentSummary] = {}

        # Clean existing document metadata in cache
        self._clean_existing_document_metadata()

    def _get_uploads_directory(self) -> Path:
        """Get the uploads directory path, creating it if necessary.

        Returns:
            Path to the uploads directory inside the knowledgebase.
        """
        uploads_dir = self.document_processor.config.knowledgebase_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        return uploads_dir

    def _clean_metadata(self, value: Any) -> Any:
        """Recursively convert metadata to JSON-serializable primitives.

        - Converts datetime to ISO 8601 UTC strings
        - Converts Path to str
        - Converts Enum to its value
        - Converts sets/tuples to lists
        - Converts modules/classes/callables to 'module.name' strings
        - Recursively processes dicts/lists
        - Falls back to str(...) for unknown types
        """
        # Localized imports to avoid changing module-level imports
        import enum as _enum
        import inspect as _inspect
        import types as _types
        from datetime import datetime as _dt
        from pathlib import Path as _Path

        def _clean(v):
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, _dt):
                # Ensure timezone-aware ISO 8601
                if getattr(v, "tzinfo", None) is None:
                    try:
                        from datetime import timezone as _timezone

                        v = v.replace(tzinfo=_timezone.utc)
                    except Exception:
                        pass
                else:
                    try:
                        from datetime import timezone as _timezone

                        v = v.astimezone(_timezone.utc)
                    except Exception:
                        pass
                return v.isoformat()
            if isinstance(v, _Path):
                return str(v)
            if isinstance(v, _enum.Enum):
                return v.value
            if isinstance(v, dict):
                # Clean dictionary more carefully
                cleaned_dict = {}
                for k, val in v.items():
                    try:
                        # Ensure keys are strings and properly cleaned
                        clean_key = str(_clean(k)) if k is not None else "null"
                        clean_val = _clean(val)
                        cleaned_dict[clean_key] = clean_val
                    except Exception as e:
                        logger.debug(f"Failed to clean dict item {k}: {e}")
                        # Fallback: convert problematic items to safe string representations
                        clean_key = str(k) if k is not None else "null"
                        cleaned_dict[clean_key] = str(val) if val is not None else None
                return cleaned_dict
            if isinstance(v, (list, tuple, set)):
                cleaned_list = []
                for item in v:
                    try:
                        cleaned_list.append(_clean(item))
                    except Exception as e:
                        logger.debug(f"Failed to clean list item {item}: {e}")
                        # Fallback: convert problematic items to string
                        cleaned_list.append(str(item) if item is not None else None)
                return cleaned_list
            if (
                isinstance(v, _types.ModuleType)
                or _inspect.isclass(v)
                or _inspect.isfunction(v)
                or _inspect.ismethod(v)
                or _inspect.isbuiltin(v)
            ):
                name = getattr(v, "__name__", str(v))
                module = getattr(v, "__module__", "")
                return f"{module}.{name}" if module else name
            # Handle any object with a __dict__ attribute (custom classes)
            if hasattr(v, "__dict__") and not isinstance(v, (_dt, _Path)):
                try:
                    # Convert object to dict and clean recursively
                    obj_dict = {}
                    for attr_name, attr_value in v.__dict__.items():
                        if not attr_name.startswith("_"):  # Skip private attributes
                            obj_dict[attr_name] = _clean(attr_value)
                    return obj_dict
                except Exception as e:
                    logger.debug(f"Failed to convert object {type(v)} to dict: {e}")
                    return (
                        f"{type(v).__module__}.{type(v).__name__}" if hasattr(type(v), "__module__") else str(type(v))
                    )
            try:
                # Try JSON serialization test
                import json

                json.dumps(v)
                return v  # If it passes JSON serialization, return as-is
            except (TypeError, ValueError):
                pass
            try:
                return str(v)
            except Exception:
                return repr(v)

        return _clean(value)

    def _clean_existing_document_metadata(self) -> None:
        """Clean metadata for all existing documents in the cache."""
        try:
            cleaned_count = 0
            for document_id, document in self.document_cache.items():
                if document.metadata:
                    original_metadata = document.metadata.copy()
                    cleaned_metadata = self._clean_metadata(document.metadata)

                    # Only update if there were changes
                    if cleaned_metadata != original_metadata:
                        document.metadata = cleaned_metadata
                        cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned metadata for {cleaned_count} existing documents")
                # Save the cleaned cache
                if self.save_cache_callback:
                    try:
                        # Try to run the save callback
                        import asyncio

                        if asyncio.iscoroutinefunction(self.save_cache_callback):
                            # If it's async, we can't run it from __init__, log a warning
                            logger.warning(
                                "Cannot save cleaned metadata synchronously - "
                                "cache will be saved on next document operation"
                            )
                        else:
                            self.save_cache_callback()
                    except Exception as e:
                        logger.warning(f"Failed to save cleaned document cache: {e}")

        except Exception as e:
            logger.error(f"Error cleaning existing document metadata: {e}")

    async def list_documents(
        self,
        pagination: PaginationParams,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> DocumentListResponse:
        """List documents with pagination and optional filtering.

        Args:
            pagination: Pagination parameters
            metadata_filter: Optional metadata filters

        Returns:
            DocumentListResponse with paginated document list
        """
        try:
            # Get all documents from cache
            all_documents = list(self.document_cache.values())

            # Apply metadata filter if provided
            if metadata_filter:
                filtered_docs = []
                for doc in all_documents:
                    matches = True
                    for key, value in metadata_filter.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            matches = False
                            break
                    if matches:
                        filtered_docs.append(doc)
                all_documents = filtered_docs

            # Convert completed documents to DocumentSummary objects, ensuring metadata is clean
            document_summaries = []
            for doc in all_documents:
                # Clean document metadata before converting to summary
                if doc.metadata:
                    doc.metadata = self._clean_metadata(doc.metadata)
                document_summaries.append(self._document_to_summary(doc))

            # Add in-progress documents to the list
            for in_progress_doc in self.in_progress_documents.values():
                # Apply metadata filter to in-progress documents too
                if metadata_filter:
                    # For in-progress docs, we don't have full metadata, so just skip filtering
                    pass
                document_summaries.append(in_progress_doc)

            # Sort by added_at descending (newest first)
            document_summaries.sort(key=lambda d: d.added_at or d.updated_at, reverse=True)

            # Calculate pagination
            total_count = len(document_summaries)
            total_pages = (total_count + pagination.page_size - 1) // pagination.page_size
            start_idx = pagination.offset
            end_idx = min(start_idx + pagination.page_size, total_count)

            # Get documents for current page
            page_documents = document_summaries[start_idx:end_idx]

            # Backfill chunk_count for completed documents that were populated without chunks
            for doc_summary in page_documents:
                # Only backfill for completed documents (they exist in document_cache)
                if (
                    doc_summary.processing_status == ProcessingStatus.COMPLETED
                    and doc_summary.id in self.document_cache
                ):
                    try:
                        doc = self.document_cache[doc_summary.id]
                        if getattr(doc, "chunk_count", 0) == 0:
                            chunks = await self.vector_store.get_document_chunks(doc.id)
                            doc.chunk_count = len(chunks)
                            # Update the summary too
                            doc_summary.chunk_count = len(chunks)
                    except Exception as e:
                        logger.debug(f"Could not backfill chunk_count for document {doc_summary.id}: {e}")

            return DocumentListResponse(
                documents=page_documents,
                total_count=total_count,
                page=pagination.page,
                page_size=pagination.page_size,
                total_pages=total_pages,
                has_next=pagination.page < total_pages,
                has_previous=pagination.page > 1,
            )

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise

    async def get_document_detail(self, document_id: str, include_chunks: bool = False) -> DocumentDetailResponse:
        """Get detailed information about a document.

        Args:
            document_id: Document ID
            include_chunks: Whether to include document chunks

        Returns:
            DocumentDetailResponse with document details
        """
        try:
            if document_id not in self.document_cache:
                raise ValueError(f"Document not found: {document_id}")

            document = self.document_cache[document_id]
            # Ensure metadata is clean before proceeding
            if document.metadata:
                document.metadata = self._clean_metadata(document.metadata)
            document_summary = self._document_to_summary(document)

            chunks = None
            if include_chunks:
                # Prefer in-memory chunks; fall back to vector store if not present
                chunk_source = document.chunks
                if not chunk_source:
                    try:
                        chunk_source = await self.vector_store.get_document_chunks(document_id)
                        if chunk_source:
                            # Backfill count for accurate summaries/status
                            document.chunk_count = len(chunk_source)
                    except Exception as e:
                        logger.debug(f"Failed to fetch chunks for {document_id} from vector store: {e}")
                        chunk_source = []
                if chunk_source:
                    chunks = [
                        ChunkResponse(
                            id=chunk.id,
                            text=getattr(chunk, "text", ""),
                            page_number=getattr(chunk, "page_number", None),
                            chunk_index=getattr(chunk, "chunk_index", 0),
                            metadata=self._clean_metadata(getattr(chunk, "metadata", {})),
                        )
                        for chunk in chunk_source
                    ]

            # Clean and log metadata for debugging
            original_metadata = document.metadata.copy() if document.metadata else {}
            cleaned_metadata = self._clean_metadata(document.metadata) if document.metadata else {}

            if document.metadata and cleaned_metadata != original_metadata:
                logger.info(
                    f"Document {document_id} metadata cleaned - "
                    f"element_types: {original_metadata.get('element_types')} -> "
                    f"{cleaned_metadata.get('element_types')}"
                )

            return DocumentDetailResponse(
                document=document_summary,
                chunks=chunks,
                metadata=cleaned_metadata,
            )

        except Exception as e:
            logger.error(f"Error getting document detail: {e}")
            raise

    async def get_document_chunks(self, document_id: str) -> List[ChunkResponse]:
        """Get all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            List of ChunkResponse objects
        """
        try:
            if document_id not in self.document_cache:
                raise ValueError(f"Document not found: {document_id}")

            document = self.document_cache[document_id]

            # Prefer in-memory chunks; fall back to vector store if not present
            chunk_source = document.chunks
            if not chunk_source:
                try:
                    chunk_source = await self.vector_store.get_document_chunks(document_id)
                except Exception as e:
                    logger.error(f"Failed to fetch chunks for document {document_id}: {e}")
                    raise

            return [
                ChunkResponse(
                    id=chunk.id,
                    text=getattr(chunk, "text", ""),
                    page_number=getattr(chunk, "page_number", None),
                    chunk_index=getattr(chunk, "chunk_index", 0),
                    metadata=self._clean_metadata(getattr(chunk, "metadata", {})),
                )
                for chunk in chunk_source
            ]

        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise

    async def get_document_preview(self, document_id: str) -> DocumentPreviewResponse:
        """Get document preview/content.

        Args:
            document_id: Document ID

        Returns:
            DocumentPreviewResponse with document content
        """
        try:
            if document_id not in self.document_cache:
                raise ValueError(f"Document not found: {document_id}")

            document = self.document_cache[document_id]

            # Combine all chunk texts to create full content
            content_parts = []
            chunk_source = document.chunks
            if not chunk_source:
                try:
                    chunk_source = await self.vector_store.get_document_chunks(document_id)
                except Exception as e:
                    logger.debug(f"Failed to fetch chunks for preview {document_id}: {e}")
                    chunk_source = []
            for chunk in sorted(chunk_source, key=lambda c: getattr(c, "chunk_index", 0)):
                content_parts.append(getattr(chunk, "text", ""))

            content = "\n\n".join(content_parts)

            return DocumentPreviewResponse(
                document_id=document_id,
                title=document.title,
                content=content,
                page_count=document.page_count,
                content_type="text/plain",
            )

        except Exception as e:
            logger.error(f"Error getting document preview: {e}")
            raise

    def _create_in_progress_document(
        self, job_id: str, filename: str, file_size: int = 0, temp_path: Optional[str] = None
    ) -> DocumentSummary:
        """Create a placeholder document for in-progress processing.

        Args:
            job_id: Background job ID
            filename: Original filename
            file_size: File size in bytes
            temp_path: Temporary file path (for size calculation)

        Returns:
            DocumentSummary for the in-progress document
        """
        # Try to get file size if not provided
        if file_size == 0 and temp_path:
            try:
                file_size = Path(temp_path).stat().st_size
            except Exception:
                file_size = 0

        # Generate a temporary document ID based on job_id
        document_id = f"processing_{job_id}"

        return DocumentSummary(
            id=document_id,
            title=None,
            filename=filename,
            path=f"processing/{filename}",  # Placeholder path
            file_size=file_size,
            page_count=0,  # Unknown until processed
            chunk_count=0,  # Unknown until processed
            added_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            has_embeddings=False,  # Will be True once processed
            checksum="",  # Unknown until processed
            processing_status=ProcessingStatus.PROCESSING,
            job_id=job_id,
            processing_error=None,
        )

    def _remove_in_progress_document(self, job_id: str) -> None:
        """Remove an in-progress document from tracking.

        Args:
            job_id: Job ID to remove
        """
        # Find and remove the in-progress document for this job
        to_remove = []
        for doc_id, doc_summary in self.in_progress_documents.items():
            if doc_summary.job_id == job_id:
                to_remove.append(doc_id)

        for doc_id in to_remove:
            del self.in_progress_documents[doc_id]
            logger.debug(f"Removed in-progress document {doc_id} for job {job_id}")

    def _update_in_progress_document_status(
        self, job_id: str, status: ProcessingStatus, error_msg: Optional[str] = None
    ) -> None:
        """Update the status of an in-progress document.

        Args:
            job_id: Job ID to update
            status: New processing status
            error_msg: Error message if status is FAILED
        """
        # Find and update the in-progress document for this job
        for doc_id, doc_summary in self.in_progress_documents.items():
            if doc_summary.job_id == job_id:
                doc_summary.processing_status = status
                if error_msg:
                    doc_summary.processing_error = error_msg
                doc_summary.updated_at = datetime.now(timezone.utc)
                logger.debug(f"Updated in-progress document {doc_id} status to {status}")
                break

    async def upload_document(
        self, file_content: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentUploadResponse:
        """Upload and process a document from file content.

        Args:
            file_content: Raw file content
            filename: Original filename
            metadata: Optional metadata

        Returns:
            DocumentUploadResponse with processing results
        """
        # If background queue is available, queue the job
        if self.background_queue:
            try:
                start_time = time.time()

                # Create temporary file in uploads directory with unique name
                uploads_dir = self._get_uploads_directory()
                # Add timestamp and UUID to make filename unique
                file_stem = Path(filename).stem
                file_suffix = Path(filename).suffix
                unique_id = str(uuid.uuid4())[:8]
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                unique_filename = f"{file_stem}_{timestamp}_{unique_id}{file_suffix}"
                temp_path = uploads_dir / unique_filename
                temp_path.write_bytes(file_content)

                # Create job metadata
                job_metadata = {
                    "operation": "upload_document",
                    "filename": filename,
                    "temp_path": str(temp_path),
                    "original_metadata": metadata or {},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                # Add job to queue
                job_id = await self.background_queue.add_job(
                    job_type=JobType.WEB_UPLOAD,
                    metadata=job_metadata,
                    priority=Priority.NORMAL,
                    processor=self._process_upload_job,
                )

                # Create in-progress document placeholder
                in_progress_doc = self._create_in_progress_document(
                    job_id=job_id, filename=filename, file_size=len(file_content), temp_path=str(temp_path)
                )
                self.in_progress_documents[in_progress_doc.id] = in_progress_doc

                # Store job info for status tracking
                self.job_metadata[job_id] = {
                    "filename": filename,
                    "operation": "upload_document",
                    "created_at": datetime.now(timezone.utc),
                    "temp_path": str(temp_path),
                    "in_progress_doc_id": in_progress_doc.id,
                }

                processing_time = time.time() - start_time

                logger.info(f"Created in-progress document {in_progress_doc.id} for upload job {job_id}")

                return DocumentUploadResponse(
                    success=True,
                    job_id=job_id,
                    filename=filename,
                    processing_time=processing_time,
                    message=f"Document upload queued for processing. Job ID: {job_id}",
                )

            except Exception as e:
                logger.error(f"Error queuing upload job: {e}")
                return DocumentUploadResponse(
                    success=False,
                    filename=filename,
                    processing_time=0.0,
                    error=str(e),
                )

        # Fallback to synchronous processing if no queue
        return await self._process_upload_synchronously(file_content, filename, metadata)

    async def add_document_by_path(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentUploadResponse:
        """Add a document by file path.

        Args:
            file_path: Path to the PDF file
            metadata: Optional metadata

        Returns:
            DocumentUploadResponse with processing results
        """
        # If background queue is available, queue the job
        if self.background_queue:
            try:
                start_time = time.time()
                path = Path(file_path)

                if not path.exists():
                    return DocumentUploadResponse(
                        success=False,
                        filename=path.name,
                        processing_time=0.0,
                        error=f"File not found: {file_path}",
                    )

                # Create job metadata
                job_metadata = {
                    "operation": "add_document_by_path",
                    "filename": path.name,
                    "file_path": file_path,
                    "original_metadata": metadata or {},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                # Add job to queue
                job_id = await self.background_queue.add_job(
                    job_type=JobType.WEB_UPLOAD,
                    metadata=job_metadata,
                    priority=Priority.NORMAL,
                    processor=self._process_path_job,
                )

                # Create in-progress document placeholder
                try:
                    file_size = path.stat().st_size
                except Exception:
                    file_size = 0

                in_progress_doc = self._create_in_progress_document(
                    job_id=job_id, filename=path.name, file_size=file_size, temp_path=file_path
                )
                in_progress_doc.path = file_path  # Use actual path instead of placeholder
                self.in_progress_documents[in_progress_doc.id] = in_progress_doc

                # Store job info for status tracking
                self.job_metadata[job_id] = {
                    "filename": path.name,
                    "operation": "add_document_by_path",
                    "created_at": datetime.now(timezone.utc),
                    "file_path": file_path,
                    "in_progress_doc_id": in_progress_doc.id,
                }

                processing_time = time.time() - start_time

                logger.info(f"Created in-progress document {in_progress_doc.id} for path job {job_id}")

                return DocumentUploadResponse(
                    success=True,
                    job_id=job_id,
                    filename=path.name,
                    processing_time=processing_time,
                    message=f"Document processing queued. Job ID: {job_id}",
                )

            except Exception as e:
                logger.error(f"Error queuing path job: {e}")
                return DocumentUploadResponse(
                    success=False,
                    filename=Path(file_path).name,
                    processing_time=0.0,
                    error=str(e),
                )

        # Fallback to synchronous processing if no queue
        return await self._process_path_synchronously(file_path, metadata)

    def _is_file_watcher_managed_document(self, document) -> bool:
        """Check if a document is managed by the file watcher (exists in knowledgebase directory).

        Args:
            document: Document to check

        Returns:
            True if the document is managed by file watcher and should not be removed via API
        """
        try:
            from pathlib import Path

            if not hasattr(document, "path") or not document.path:
                return False

            doc_path = Path(document.path).resolve()
            kb_path = self.document_processor.config.knowledgebase_path.resolve()
            uploads_path = (kb_path / "uploads").resolve()

            # Check if document is within knowledgebase directory
            try:
                doc_path.relative_to(kb_path)
            except ValueError:
                # Document path is not within knowledgebase directory
                return False

            # Check if document is NOT in uploads directory (uploads are user-managed)
            try:
                doc_path.relative_to(uploads_path)
                # Document is in uploads directory, so it's user-managed
                return False
            except ValueError:
                # Document is not in uploads, so it could be file-watcher-managed
                pass

            # Check if the original file still exists
            if doc_path.exists():
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking if document is file-watcher-managed: {e}")
            return False

    async def remove_document(self, document_id: str) -> Dict[str, Any]:
        """Remove a document from the knowledgebase.

        Args:
            document_id: Document ID to remove

        Returns:
            Removal result dictionary
        """
        try:
            if document_id not in self.document_cache:
                raise ValueError(f"Document not found: {document_id}")

            document = self.document_cache[document_id]

            # Check if document is managed by file watcher
            if self._is_file_watcher_managed_document(document):
                error_msg = (
                    f"Cannot remove document '{getattr(document, 'filename', None) or document_id}' as it exists "
                    f"in the knowledgebase directory ({document.path}). To remove this document, delete the "
                    f"file from the filesystem directly."
                )
                logger.warning(f"Attempted to remove file-watcher-managed document via web interface: {document_id}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": "file_watcher_managed",
                    "document_path": document.path,
                }

            start_time = time.time()

            # Remove from vector store
            await self.vector_store.delete_document(document_id)

            # For user-uploaded documents, also remove the physical file
            document_path = document.path
            try:
                file_path = Path(document_path)
                uploads_dir = self._get_uploads_directory()

                # Check if this is an uploaded file (in uploads directory)
                try:
                    file_path.relative_to(uploads_dir)
                    # It's in uploads directory, safe to delete
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"Deleted uploaded file: {file_path}")
                    else:
                        logger.warning(f"Uploaded file not found for deletion: {file_path}")
                except ValueError:
                    # File is not in uploads directory, don't delete it
                    logger.debug(f"Document file not in uploads directory, preserving: {file_path}")

            except Exception as e:
                logger.warning(f"Failed to delete uploaded file {document_path}: {e}")

            # Remove from document cache
            del self.document_cache[document_id]
            if self.save_cache_callback:
                await self.save_cache_callback()

            processing_time = time.time() - start_time

            return {
                "success": True,
                "document_id": document_id,
                "document_path": document.path,
                "processing_time": processing_time,
                "message": f"Document {document_id} removed successfully",
            }

        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        """Get the status of a background job.

        Args:
            job_id: Job identifier

        Returns:
            JobStatusResponse with job status information
        """
        if not self.background_queue:
            return JobStatusResponse(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                error="Background queue not available",
            )

        try:
            status = await self.background_queue.get_status(job_id)
            if status is None:
                return JobStatusResponse(
                    success=False,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error="Job not found",
                )

            # Get job metadata if available
            job_info = self.job_metadata.get(job_id, {})

            return JobStatusResponse(
                success=True,
                job_id=job_id,
                status=status,
                created_at=job_info.get("created_at"),
                metadata=self._clean_metadata(job_info),
            )

        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return JobStatusResponse(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                error=str(e),
            )

    async def cancel_job(self, job_id: str) -> JobCancelResponse:
        """Cancel a background job.

        Args:
            job_id: Job identifier

        Returns:
            JobCancelResponse with cancellation result
        """
        if not self.background_queue:
            return JobCancelResponse(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                message="Background queue not available",
            )

        try:
            success = await self.background_queue.cancel_job(job_id)
            if success:
                status = await self.background_queue.get_status(job_id)
                return JobCancelResponse(
                    success=True,
                    job_id=job_id,
                    status=status or JobStatus.CANCELED,
                    message="Job cancelled successfully",
                )
            else:
                status = await self.background_queue.get_status(job_id)
                if status is None:
                    message = "Job not found"
                elif status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                    message = f"Job already {status.name.lower()}"
                else:
                    message = "Job could not be cancelled"

                return JobCancelResponse(
                    success=False,
                    job_id=job_id,
                    status=status or JobStatus.FAILED,
                    message=message,
                )

        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return JobCancelResponse(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                message=f"Error cancelling job: {e}",
            )

    async def _process_upload_job(self, job: Job) -> None:
        """Process a document upload job in the background.

        Args:
            job: Job instance to process
        """
        try:
            metadata = job.metadata
            filename = metadata["filename"]
            temp_path = Path(metadata["temp_path"])
            original_metadata = metadata["original_metadata"]

            logger.info(f"Processing upload job {job.job_id} for file: {filename}")

            # Send WebSocket notification that processing started
            if self.websocket_manager:
                await self.websocket_manager.broadcast_processing_started(filename, job.job_id)

            # Process the document
            result = await self.document_processor.process_document(temp_path, original_metadata)

            if result.success and result.document:
                # Clean document metadata before storing
                if result.document.metadata:
                    result.document.metadata = self._clean_metadata(result.document.metadata)

                # Add to vector store
                await self.vector_store.add_document(result.document)

                # Update document cache
                self.document_cache[result.document.id] = result.document
                if self.save_cache_callback:
                    await self.save_cache_callback()

                # Update job metadata with result
                if job.job_id in self.job_metadata:
                    self.job_metadata[job.job_id].update(
                        {
                            "document_id": result.document.id,
                            "chunks_created": result.chunks_created,
                            "embeddings_generated": result.embeddings_generated,
                            "completed_at": datetime.now(timezone.utc),
                        }
                    )

                # Remove in-progress document since processing completed successfully
                self._remove_in_progress_document(job.job_id)

                # Send WebSocket notifications
                if self.websocket_manager:
                    # Processing completed (includes document added information)
                    await self.websocket_manager.broadcast_processing_completed(
                        {
                            "job_id": job.job_id,
                            "document_id": result.document.id,
                            "filename": filename,
                            "title": result.document.title,
                            "path": result.document.path,
                            "chunks_created": result.chunks_created,
                        }
                    )

                logger.info(f"Upload job {job.job_id} completed successfully")

            else:
                # Job failed
                error_msg = result.error or "Unknown processing error"
                if job.job_id in self.job_metadata:
                    self.job_metadata[job.job_id].update(
                        {
                            "error": error_msg,
                            "failed_at": datetime.now(timezone.utc),
                        }
                    )

                if self.websocket_manager:
                    await self.websocket_manager.broadcast_processing_failed(filename, error_msg)

                logger.error(f"Upload job {job.job_id} failed: {error_msg}")
                raise RuntimeError(error_msg)

        finally:
            # Note: We DO NOT clean up uploaded files - they should remain in the uploads directory
            # as they are user-managed documents that can be removed via the API
            temp_path = Path(job.metadata.get("temp_path", ""))
            logger.debug(f"Upload job completed, file preserved at: {temp_path}")

    async def _process_path_job(self, job: Job) -> None:
        """Process a document path job in the background.

        Args:
            job: Job instance to process
        """
        try:
            metadata = job.metadata
            filename = metadata["filename"]
            file_path = metadata["file_path"]
            original_metadata = metadata["original_metadata"]

            logger.info(f"Processing path job {job.job_id} for file: {file_path}")

            # Send WebSocket notification that processing started
            if self.websocket_manager:
                await self.websocket_manager.broadcast_processing_started(filename, job.job_id)

            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Process the document
            result = await self.document_processor.process_document(path, original_metadata)

            if result.success and result.document:
                # Clean document metadata before storing
                if result.document.metadata:
                    result.document.metadata = self._clean_metadata(result.document.metadata)

                # Add to vector store
                await self.vector_store.add_document(result.document)

                # Update document cache
                self.document_cache[result.document.id] = result.document
                if self.save_cache_callback:
                    await self.save_cache_callback()

                # Update job metadata with result
                if job.job_id in self.job_metadata:
                    self.job_metadata[job.job_id].update(
                        {
                            "document_id": result.document.id,
                            "chunks_created": result.chunks_created,
                            "embeddings_generated": result.embeddings_generated,
                            "completed_at": datetime.now(timezone.utc),
                        }
                    )

                # Send WebSocket notifications
                if self.websocket_manager:
                    # Processing completed (includes document added information)
                    await self.websocket_manager.broadcast_processing_completed(
                        {
                            "job_id": job.job_id,
                            "document_id": result.document.id,
                            "filename": filename,
                            "title": result.document.title,
                            "path": result.document.path,
                            "chunks_created": result.chunks_created,
                        }
                    )

                logger.info(f"Path job {job.job_id} completed successfully")

            else:
                # Job failed
                error_msg = result.error or "Unknown processing error"
                if job.job_id in self.job_metadata:
                    self.job_metadata[job.job_id].update(
                        {
                            "error": error_msg,
                            "failed_at": datetime.now(timezone.utc),
                        }
                    )

                if self.websocket_manager:
                    await self.websocket_manager.broadcast_processing_failed(filename, error_msg)

                logger.error(f"Path job {job.job_id} failed: {error_msg}")
                raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = str(e)
            if job.job_id in self.job_metadata:
                self.job_metadata[job.job_id].update(
                    {
                        "error": error_msg,
                        "failed_at": datetime.now(timezone.utc),
                    }
                )

            if self.websocket_manager:
                filename = job.metadata.get("filename", "unknown")
                await self.websocket_manager.broadcast_processing_failed(filename, error_msg)

            logger.error(f"Path job {job.job_id} failed: {error_msg}")
            raise

    async def _process_upload_synchronously(
        self, file_content: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentUploadResponse:
        """Fallback method to process upload synchronously when no queue is available."""
        try:
            start_time = time.time()

            # Create temporary file in uploads directory with unique name
            uploads_dir = self._get_uploads_directory()
            # Add timestamp and UUID to make filename unique
            file_stem = Path(filename).stem
            file_suffix = Path(filename).suffix
            unique_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{file_stem}_{timestamp}_{unique_id}{file_suffix}"
            temp_path = uploads_dir / unique_filename
            temp_path.write_bytes(file_content)

            try:
                # Process the document
                result = await self.document_processor.process_document(temp_path, metadata)
                processing_time = time.time() - start_time

                if not result.success:
                    return DocumentUploadResponse(
                        success=False,
                        filename=filename,
                        processing_time=processing_time,
                        error=result.error,
                    )

                # Add to vector store if successful
                if result.document:
                    # Clean document metadata before storing
                    if result.document.metadata:
                        result.document.metadata = self._clean_metadata(result.document.metadata)

                    await self.vector_store.add_document(result.document)

                    # Update document cache
                    self.document_cache[result.document.id] = result.document
                    if self.save_cache_callback:
                        await self.save_cache_callback()

                return DocumentUploadResponse(
                    success=True,
                    document_id=result.document.id if result.document else None,
                    filename=filename,
                    processing_time=processing_time,
                    chunks_created=result.chunks_created,
                    embeddings_generated=result.embeddings_generated,
                )

            finally:
                # Note: We DO NOT clean up uploaded files - they should remain in the uploads directory
                # as they are user-managed documents that can be removed via the API
                logger.debug(f"Upload completed synchronously, file preserved at: {temp_path}")

        except Exception as e:
            logger.error(f"Error uploading document synchronously: {e}")
            return DocumentUploadResponse(
                success=False,
                filename=filename,
                processing_time=time.time() - start_time if "start_time" in locals() else 0.0,
                error=str(e),
            )

    async def _process_path_synchronously(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentUploadResponse:
        """Fallback method to process path synchronously when no queue is available."""
        try:
            start_time = time.time()
            path = Path(file_path)

            if not path.exists():
                return DocumentUploadResponse(
                    success=False,
                    filename=path.name,
                    processing_time=0.0,
                    error=f"File not found: {file_path}",
                )

            # Process the document
            result = await self.document_processor.process_document(path, metadata)
            processing_time = time.time() - start_time

            if not result.success:
                return DocumentUploadResponse(
                    success=False,
                    filename=path.name,
                    processing_time=processing_time,
                    error=result.error,
                )

            # Add to vector store if successful
            if result.document:
                # Clean document metadata before storing
                if result.document.metadata:
                    result.document.metadata = self._clean_metadata(result.document.metadata)

                await self.vector_store.add_document(result.document)

                # Update document cache
                self.document_cache[result.document.id] = result.document
                if self.save_cache_callback:
                    await self.save_cache_callback()

            return DocumentUploadResponse(
                success=True,
                document_id=result.document.id if result.document else None,
                filename=path.name,
                processing_time=processing_time,
                chunks_created=result.chunks_created,
                embeddings_generated=result.embeddings_generated,
            )

        except Exception as e:
            logger.error(f"Error adding document by path synchronously: {e}")
            return DocumentUploadResponse(
                success=False,
                filename=Path(file_path).name,
                processing_time=time.time() - start_time if "start_time" in locals() else 0.0,
                error=str(e),
            )

    def _document_to_summary(self, document: Document) -> DocumentSummary:
        """Convert Document to DocumentSummary.

        Args:
            document: Document object

        Returns:
            DocumentSummary object
        """
        # Ensure document metadata is clean before creating summary
        if document.metadata:
            try:
                original_metadata = document.metadata
                cleaned_metadata = self._clean_metadata(document.metadata)
                if cleaned_metadata != original_metadata:
                    logger.debug(f"Cleaned metadata for document {document.id}")
                    document.metadata = cleaned_metadata
            except Exception as e:
                logger.error(f"Error cleaning metadata for document {document.id}: {e}")

        # If chunk_count > 0, the document must have embeddings (vector store only stores embedded chunks)
        inferred_has_embeddings = bool(getattr(document, "chunk_count", 0) > 0) or bool(
            getattr(document, "chunks", None) and all(getattr(c, "embedding", None) for c in document.chunks)
        )

        return DocumentSummary(
            id=document.id,
            title=document.title,
            filename=document.filename,
            path=document.path,
            file_size=document.file_size,
            page_count=document.page_count,
            chunk_count=document.chunk_count,
            added_at=document.added_at,
            updated_at=document.updated_at,
            has_embeddings=inferred_has_embeddings,
            checksum=document.checksum,
            processing_status=ProcessingStatus.COMPLETED,  # Existing documents are always completed
            job_id=None,  # No job ID for completed documents
            processing_error=None,  # No error for completed documents
        )
