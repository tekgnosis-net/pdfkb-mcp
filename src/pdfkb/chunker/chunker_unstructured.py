"""Chunker using the unstructured library for markdown text chunking."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..models import Chunk
from .chunker import Chunker

logger = logging.getLogger(__name__)


class ChunkerUnstructured(Chunker):
    """Chunker using the unstructured library for markdown text chunking."""

    def __init__(self, cache_dir: str = None, min_chunk_size: int = 0):
        """Initialize the unstructured chunker with zero configuration.

        Args:
            cache_dir: Directory to cache chunked results (not used in this implementation).
            min_chunk_size: Minimum size for chunks (0 = disabled).
        """
        super().__init__(cache_dir=cache_dir, min_chunk_size=min_chunk_size)

        try:
            import unstructured
            from unstructured.partition.text import partition_text

            self.partition_text = partition_text
            # Handle cases where __version__ might not exist
            self.unstructured_version = getattr(unstructured, "__version__", "unknown")
        except ImportError:
            raise ImportError(
                "Unstructured library not available. Install with: pip install pdfkb-mcp[unstructured_chunker]"
            )

    def chunk(self, markdown_content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk markdown content using unstructured library.

        Args:
            markdown_content: Markdown text to chunk.
            metadata: Document metadata.

        Returns:
            List of Chunk objects.
        """
        try:
            if not markdown_content or not markdown_content.strip():
                logger.warning("Empty markdown content provided to chunker")
                return []

            # Partition text using unstructured with built-in chunking
            elements = self.partition_text(
                text=markdown_content,
                chunking_strategy="by_title",
                max_characters=1000,
                new_after_n_chars=800,
                combine_text_under_n_chars=150,
            )

            # Convert elements to Chunk objects
            chunks = []
            for i, element in enumerate(elements):
                chunk_text = str(element).strip()
                if not chunk_text:  # Skip empty chunks
                    continue

                # Create metadata for this chunk
                chunk_metadata = {
                    "chunk_strategy": "unstructured_by_title",
                    "element_type": element.__class__.__name__,
                    "unstructured_version": self.unstructured_version,
                    "max_characters": 1000,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                # Add any provided metadata
                chunk_metadata.update(metadata)

                chunk = Chunk(text=chunk_text, chunk_index=i, metadata=chunk_metadata)
                chunks.append(chunk)

            # Apply minimum chunk size filtering
            chunks = self._filter_small_chunks(chunks)

            logger.info(f"Created {len(chunks)} chunks using unstructured library")
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk markdown content with unstructured: {e}")
            raise RuntimeError(f"Failed to chunk text: {e}") from e
