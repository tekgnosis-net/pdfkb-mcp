"""Abstract base class for text chunkers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

from ..models import Chunk

if TYPE_CHECKING:
    from ..parsers.parser import PageContent

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result of chunking a markdown document."""

    chunks: List[Chunk]
    metadata: Dict[str, Any]


class Chunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(self, cache_dir: str = None):
        """Initialize the chunker with optional cache directory.

        Args:
            cache_dir: Directory to cache chunked results.
        """
        self.cache_dir = cache_dir

    @abstractmethod
    def chunk(self, markdown_content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk markdown content into smaller pieces.

        Args:
            markdown_content: Markdown text to chunk.
            metadata: Document metadata.

        Returns:
            List of Chunk objects.
        """
        pass

    def chunk_pages(self, pages: List["PageContent"], metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk page-aware content into smaller pieces.

        Default implementation combines pages and uses regular chunking.
        Subclasses can override for page-aware chunking strategies.

        Args:
            pages: List of PageContent objects with per-page content.
            metadata: Document-level metadata.

        Returns:
            List of Chunk objects with page metadata preserved.
        """
        # Default implementation: combine pages with page headers
        combined_parts = []
        for page in pages:
            combined_parts.append(f"# Page {page.page_number}\n")
            combined_parts.append(page.markdown_content)

        combined_markdown = "\n\n".join(combined_parts)

        # Use regular chunking on combined content
        chunks = self.chunk(combined_markdown, metadata)

        # Try to enrich chunks with page information based on page headers
        for chunk in chunks:
            # Look for page header in chunk text
            import re

            page_match = re.search(r"# Page (\d+)", chunk.text)
            if page_match:
                chunk.metadata["page_number"] = int(page_match.group(1))

        return chunks
