"""Custom exceptions for the PDF Knowledgebase server."""

from typing import Optional


class PDFKnowledgebaseError(Exception):
    """Base exception for all PDF Knowledgebase server errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        """Initialize the exception.

        Args:
            message: Error message.
            cause: Optional underlying exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        """String representation of the error."""
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ConfigurationError(PDFKnowledgebaseError):
    """Configuration-related errors."""

    pass


class PDFProcessingError(PDFKnowledgebaseError):
    """Errors during PDF processing."""

    def __init__(self, message: str, file_path: Optional[str] = None, cause: Optional[Exception] = None):
        """Initialize PDF processing error.

        Args:
            message: Error message.
            file_path: Optional path to the PDF file that caused the error.
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.file_path = file_path

    def __str__(self) -> str:
        """String representation including file path."""
        base_msg = super().__str__()
        if self.file_path:
            return f"{base_msg} (file: {self.file_path})"
        return base_msg


class EmbeddingError(PDFKnowledgebaseError):
    """Errors during embedding generation."""

    def __init__(self, message: str, model: Optional[str] = None, cause: Optional[Exception] = None):
        """Initialize embedding error.

        Args:
            message: Error message.
            model: Optional embedding model name.
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.model = model

    def __str__(self) -> str:
        """String representation including model name."""
        base_msg = super().__str__()
        if self.model:
            return f"{base_msg} (model: {self.model})"
        return base_msg


class VectorStoreError(PDFKnowledgebaseError):
    """Errors with vector database operations."""

    def __init__(self, message: str, operation: Optional[str] = None, cause: Optional[Exception] = None):
        """Initialize vector store error.

        Args:
            message: Error message.
            operation: Optional operation that failed (e.g., 'search', 'add', 'delete').
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.operation = operation

    def __str__(self) -> str:
        """String representation including operation."""
        base_msg = super().__str__()
        if self.operation:
            return f"{base_msg} (operation: {self.operation})"
        return base_msg


class FileSystemError(PDFKnowledgebaseError):
    """File system related errors."""

    def __init__(self, message: str, file_path: Optional[str] = None, cause: Optional[Exception] = None):
        """Initialize file system error.

        Args:
            message: Error message.
            file_path: Optional file path that caused the error.
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.file_path = file_path

    def __str__(self) -> str:
        """String representation including file path."""
        base_msg = super().__str__()
        if self.file_path:
            return f"{base_msg} (path: {self.file_path})"
        return base_msg


class DocumentNotFoundError(PDFKnowledgebaseError):
    """Error when a requested document is not found."""

    def __init__(self, document_id: str):
        """Initialize document not found error.

        Args:
            document_id: ID of the document that was not found.
        """
        super().__init__(f"Document not found: {document_id}")
        self.document_id = document_id


class ValidationError(PDFKnowledgebaseError):
    """Input validation errors."""

    def __init__(self, message: str, field: Optional[str] = None):
        """Initialize validation error.

        Args:
            message: Error message.
            field: Optional field name that failed validation.
        """
        super().__init__(message)
        self.field = field

    def __str__(self) -> str:
        """String representation including field name."""
        if self.field:
            return f"Validation error for '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


class RateLimitError(PDFKnowledgebaseError):
    """Rate limiting errors, typically from external APIs."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        """Initialize rate limit error.

        Args:
            message: Error message.
            retry_after: Optional seconds to wait before retrying.
        """
        super().__init__(message)
        self.retry_after = retry_after

    def __str__(self) -> str:
        """String representation including retry delay."""
        base_msg = super().__str__()
        if self.retry_after:
            return f"{base_msg} (retry after {self.retry_after}s)"
        return base_msg


class ChunkingError(PDFKnowledgebaseError):
    """Errors during text chunking operations."""

    def __init__(self, message: str, strategy: Optional[str] = None, cause: Optional[Exception] = None):
        """Initialize chunking error.

        Args:
            message: Error message.
            strategy: Optional chunking strategy that failed.
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.strategy = strategy

    def __str__(self) -> str:
        """String representation including strategy."""
        base_msg = super().__str__()
        if self.strategy:
            return f"{base_msg} (strategy: {self.strategy})"
        return base_msg


class FileMonitorError(PDFKnowledgebaseError):
    """Errors during file monitoring operations."""

    def __init__(self, message: str, operation: Optional[str] = None, cause: Optional[Exception] = None):
        """Initialize file monitor error.

        Args:
            message: Error message.
            operation: Optional operation that failed.
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.operation = operation

    def __str__(self) -> str:
        """String representation including operation."""
        base_msg = super().__str__()
        if self.operation:
            return f"{base_msg} (operation: {self.operation})"
        return base_msg
