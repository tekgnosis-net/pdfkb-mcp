"""Base interface for document summarization services."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentSummary:
    """Container for document summarization results."""

    title: str
    short_description: str
    long_description: str
    confidence: Optional[float] = None  # Confidence score if available

    def __post_init__(self):
        """Validate summary fields."""
        if not self.title or not self.title.strip():
            raise ValueError("Title cannot be empty")
        if not self.short_description or not self.short_description.strip():
            raise ValueError("Short description cannot be empty")
        if not self.long_description or not self.long_description.strip():
            raise ValueError("Long description cannot be empty")


class SummarizerService(ABC):
    """Abstract base class for document summarization services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the summarization service."""
        pass

    @abstractmethod
    async def summarize_document(self, content: str, filename: str = "") -> DocumentSummary:
        """Summarize a document and generate title and descriptions.

        Args:
            content: The document content to summarize.
            filename: Optional filename for context.

        Returns:
            DocumentSummary with title, short description, and long description.
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the summarization service is available.

        Returns:
            True if service is working, False otherwise.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the current summarization model.

        Returns:
            Dictionary with model information.
        """
        pass
