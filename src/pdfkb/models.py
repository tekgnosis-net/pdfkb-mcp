"""Data models and schemas for the PDF Knowledgebase server."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class Chunk:
    """Represents a text chunk from a document."""

    id: str = ""
    document_id: str = ""
    text: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_number: Optional[int] = None
    chunk_index: int = 0

    def __post_init__(self):
        """Set default metadata and generate deterministic ID."""
        # Generate deterministic ID based on content
        if not self.id:
            self.id = self._generate_content_id()

        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()

    def _generate_content_id(self) -> str:
        """Generate a deterministic ID based on chunk content.

        Returns:
            Deterministic hash-based ID for the chunk.
        """
        # Create a unique string from the chunk's key characteristics
        content_string = f"{self.document_id}|{self.chunk_index}|{self.text}|{self.page_number}"

        # Generate SHA-256 hash
        chunk_hash = hashlib.sha256(content_string.encode("utf-8")).hexdigest()

        # Return first 16 characters for readability (still extremely unlikely to collide)
        return f"chunk_{chunk_hash[:16]}"

    @property
    def has_embedding(self) -> bool:
        """Check if chunk has an embedding."""
        return self.embedding is not None and len(self.embedding) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            document_id=data.get("document_id", ""),
            text=data.get("text", ""),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            page_number=data.get("page_number"),
            chunk_index=data.get("chunk_index", 0),
        )


@dataclass
class Document:
    """Represents a processed PDF document."""

    id: str = ""
    path: str = ""
    title: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    page_count: int = 0
    chunk_count: int = 0
    file_size: int = 0
    added_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Set default values and generate deterministic ID."""
        if self.added_at is None:
            self.added_at = datetime.now(timezone.utc)

        if self.updated_at is None:
            self.updated_at = self.added_at

        # Update chunk count
        self.chunk_count = len(self.chunks)

        # Generate deterministic ID based on file path and checksum
        if not self.id:
            self.id = self._generate_document_id()

        # Set default metadata
        if "source" not in self.metadata:
            self.metadata["source"] = self.path

        if "added_at" not in self.metadata:
            self.metadata["added_at"] = self.added_at.isoformat()

    def _generate_document_id(self) -> str:
        """Generate a deterministic ID based on document characteristics.

        Returns:
            Deterministic hash-based ID for the document.
        """
        # Use file path and checksum for uniqueness
        content_string = f"{self.path}|{self.checksum}"

        # Generate SHA-256 hash
        doc_hash = hashlib.sha256(content_string.encode("utf-8")).hexdigest()

        # Return first 16 characters for readability
        return f"doc_{doc_hash[:16]}"

    @property
    def filename(self) -> str:
        """Get the filename from the path."""
        return Path(self.path).name

    @property
    def has_chunks(self) -> bool:
        """Check if document has chunks."""
        return len(self.chunks) > 0

    @property
    def has_embeddings(self) -> bool:
        """Check if all chunks have embeddings."""
        return all(chunk.has_embedding for chunk in self.chunks)

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the document."""
        chunk.document_id = self.id
        self.chunks.append(chunk)
        self.chunk_count = len(self.chunks)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self, include_chunks: bool = True) -> Dict[str, Any]:
        """Convert document to dictionary.

        Args:
            include_chunks: Whether to include chunks in the output.
        """
        result = {
            "id": self.id,
            "path": self.path,
            "title": self.title,
            "metadata": self.metadata,
            "checksum": self.checksum,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "file_size": self.file_size,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        if include_chunks:
            result["chunks"] = [chunk.to_dict() for chunk in self.chunks]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        # Parse datetime fields
        added_at = None
        if data.get("added_at"):
            added_at = datetime.fromisoformat(data["added_at"].replace("Z", "+00:00"))

        updated_at = None
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))

        # Parse chunks
        chunks = []
        if "chunks" in data:
            chunks = [Chunk.from_dict(chunk_data) for chunk_data in data["chunks"]]

        return cls(
            id=data.get("id", str(uuid4())),
            path=data.get("path", ""),
            title=data.get("title"),
            chunks=chunks,
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
            page_count=data.get("page_count", 0),
            chunk_count=data.get("chunk_count", 0),
            file_size=data.get("file_size", 0),
            added_at=added_at,
            updated_at=updated_at,
        )


@dataclass
class SearchResult:
    """Represents a search result."""

    chunk: Chunk
    score: float
    document: Document
    search_type: str = "hybrid"  # Which search contributed this result
    vector_score: Optional[float] = None  # Original vector similarity
    text_score: Optional[float] = None  # Original BM25 score

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "document_id": self.document.id,
            "document_title": self.document.title or self.document.filename,
            "document_path": self.document.path,
            "chunk_id": self.chunk.id,
            "chunk_text": self.chunk.text,
            "page_number": self.chunk.page_number,
            "chunk_index": self.chunk.chunk_index,
            "score": self.score,
            "metadata": {
                **self.document.metadata,
                **self.chunk.metadata,
            },
        }


@dataclass
class SearchQuery:
    """Represents a search query."""

    query: str
    limit: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None
    min_score: float = 0.0
    search_type: str = "hybrid"  # "hybrid", "vector", "text"

    def __post_init__(self):
        """Validate query parameters."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")

        if self.limit <= 0:
            raise ValueError("Limit must be positive")

        if self.min_score < 0 or self.min_score > 1:
            raise ValueError("min_score must be between 0 and 1")


@dataclass
class ProcessingResult:
    """Represents the result of processing a document."""

    success: bool
    document: Optional[Document] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    chunks_created: int = 0
    embeddings_generated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert processing result to dictionary."""
        result = {
            "success": self.success,
            "processing_time": self.processing_time,
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
        }

        if self.document:
            result["document"] = self.document.to_dict(include_chunks=False)

        if self.error:
            result["error"] = self.error

        return result
