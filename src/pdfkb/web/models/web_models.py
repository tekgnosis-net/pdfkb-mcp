"""Web-specific request/response models for FastAPI endpoints."""

import enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...background_queue import JobStatus


def utc_now() -> datetime:
    """Get current UTC datetime in timezone-aware format.

    Replaces deprecated datetime.utcnow() with the recommended approach.
    """
    return datetime.now(timezone.utc)


class ProcessingStatus(str, enum.Enum):
    """Document processing status."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        """Calculate offset from page and page_size."""
        return (self.page - 1) * self.page_size


class ErrorResponse(BaseModel):
    """Standard error response model."""

    success: bool = False
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for client handling")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class DocumentSummary(BaseModel):
    """Summary information for a document in list views."""

    id: str = Field(..., description="Unique document identifier")
    title: Optional[str] = Field(None, description="Document title")
    filename: str = Field(..., description="Original filename")
    path: str = Field(..., description="File path")
    file_size: int = Field(..., description="File size in bytes")
    page_count: int = Field(..., description="Number of pages")
    chunk_count: int = Field(..., description="Number of text chunks")
    added_at: Optional[datetime] = Field(None, description="When document was added")
    updated_at: Optional[datetime] = Field(None, description="When document was last updated")
    has_embeddings: bool = Field(..., description="Whether document has embeddings")
    checksum: str = Field(..., description="File checksum for change detection")
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.COMPLETED, description="Document processing status"
    )
    job_id: Optional[str] = Field(None, description="Background job ID if currently being processed")
    processing_error: Optional[str] = Field(None, description="Error message if processing failed")


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""

    success: bool = True
    documents: List[DocumentSummary] = Field(..., description="List of document summaries")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class ChunkResponse(BaseModel):
    """Response model for a document chunk."""

    id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    page_number: Optional[int] = Field(None, description="Source page number")
    chunk_index: int = Field(..., description="Index within document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class DocumentDetailResponse(BaseModel):
    """Response model for detailed document information."""

    success: bool = True
    document: DocumentSummary = Field(..., description="Document summary")
    chunks: Optional[List[ChunkResponse]] = Field(None, description="Document chunks (if requested)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class DocumentPreviewResponse(BaseModel):
    """Response model for document preview/content."""

    success: bool = True
    document_id: str = Field(..., description="Document identifier")
    title: Optional[str] = Field(None, description="Document title")
    content: str = Field(..., description="Document text content")
    page_count: int = Field(..., description="Number of pages")
    content_type: str = Field(default="text/plain", description="Content type")


class FileUploadRequest(BaseModel):
    """Request model for file upload metadata."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(default="application/pdf", description="File content type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    success: bool = Field(..., description="Whether upload was successful")
    job_id: Optional[str] = Field(None, description="Background job ID for tracking progress")
    document_id: Optional[str] = Field(
        None, description="Generated document ID (if processing completed synchronously)"
    )
    filename: str = Field(..., description="Uploaded filename")
    processing_time: float = Field(..., description="Processing time in seconds")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    embeddings_generated: int = Field(default=0, description="Number of embeddings generated")
    error: Optional[str] = Field(None, description="Error message if upload failed")
    message: Optional[str] = Field(None, description="Additional status message")


class SearchRequest(BaseModel):
    """Request model for search operations."""

    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_chunks: bool = Field(default=True, description="Whether to include full chunk text")


class SearchResultItem(BaseModel):
    """Individual search result item."""

    document_id: str = Field(..., description="Source document ID")
    document_title: Optional[str] = Field(None, description="Document title")
    document_path: str = Field(..., description="Document path")
    chunk_id: str = Field(..., description="Matching chunk ID")
    chunk_text: str = Field(..., description="Matching chunk text")
    page_number: Optional[int] = Field(None, description="Source page number")
    chunk_index: int = Field(..., description="Chunk position in document")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Combined metadata")


class SearchResponse(BaseModel):
    """Response model for search operations."""

    success: bool = True
    results: List[SearchResultItem] = Field(..., description="Search results")
    total_results: int = Field(..., description="Number of results returned")
    query: str = Field(..., description="Original search query")
    search_time: float = Field(..., description="Search execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")


class StatusResponse(BaseModel):
    """Response model for system status."""

    success: bool = True
    status: str = Field(default="healthy", description="System status")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")
    documents_count: int = Field(..., description="Total number of documents")
    chunks_count: int = Field(..., description="Total number of chunks")
    knowledgebase_path: str = Field(..., description="Path to knowledgebase directory")
    cache_dir: str = Field(..., description="Path to cache directory")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Current configuration")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="System statistics")


class ConfigOverviewResponse(BaseModel):
    """Response model for configuration overview."""

    success: bool = True
    embedding_model: str = Field(..., description="Current embedding model")
    pdf_parser: str = Field(..., description="Current PDF parser")
    pdf_chunker: str = Field(..., description="Current text chunker")
    chunk_size: int = Field(..., description="Text chunk size")
    chunk_overlap: int = Field(..., description="Text chunk overlap")
    vector_search_k: int = Field(..., description="Default search result count")
    reranker_enabled: bool = Field(..., description="Whether reranker is enabled")
    reranker_model: Optional[str] = Field(None, description="Current reranker model")
    web_enabled: bool = Field(..., description="Whether web interface is enabled")
    web_port: int = Field(..., description="Web server port")
    web_host: str = Field(..., description="Web server host")
    supported_extensions: List[str] = Field(..., description="Supported file extensions")


class WebsocketEventType(str, enum.Enum):
    """WebSocket event types."""

    DOCUMENT_ADDED = "document_added"
    DOCUMENT_REMOVED = "document_removed"
    DOCUMENT_UPDATED = "document_updated"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    JOB_STATUS_CHANGED = "job_status_changed"
    JOB_PROGRESS_UPDATED = "job_progress_updated"
    JOB_CANCELLED = "job_cancelled"
    SEARCH_PERFORMED = "search_performed"
    SYSTEM_STATUS_CHANGED = "system_status_changed"
    ERROR_OCCURRED = "error_occurred"


class WebsocketMessage(BaseModel):
    """WebSocket message format."""

    model_config = ConfigDict(use_enum_values=True)

    event_type: WebsocketEventType = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=utc_now, description="Event timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    message: Optional[str] = Field(None, description="Human-readable message")
    client_id: Optional[str] = Field(None, description="Target client ID (for unicast)")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(default="ok", description="Health status")
    timestamp: datetime = Field(default_factory=utc_now, description="Check timestamp")
    version: Optional[str] = Field(None, description="Application version")


class AddDocumentByPathRequest(BaseModel):
    """Request model for adding document by file path."""

    path: str = Field(..., description="File path to add")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""

    success: bool = True
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Job progress (0.0 to 1.0)")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result data (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    created_at: Optional[datetime] = Field(None, description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Job metadata")


class JobCancelResponse(BaseModel):
    """Response model for job cancellation."""

    success: bool = Field(..., description="Whether cancellation was successful")
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Job status after cancellation attempt")
    message: str = Field(..., description="Cancellation result message")


class JobListResponse(BaseModel):
    """Response model for listing jobs."""

    success: bool = True
    jobs: List[Dict[str, Any]] = Field(..., description="List of job summaries")
    total_count: int = Field(..., description="Total number of jobs")
    active_count: int = Field(..., description="Number of active jobs")
    completed_count: int = Field(..., description="Number of completed jobs")
    failed_count: int = Field(..., description="Number of failed jobs")


class SearchSuggestionsResponse(BaseModel):
    """Response model for search query suggestions."""

    success: bool = True
    suggestions: List[str] = Field(..., description="Query suggestions")
    query: str = Field(..., description="Original query fragment")
