"""Abstract base class for text chunkers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from ..models import Chunk

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
