"""Chunker using LangChain's MarkdownHeaderTextSplitter."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..models import Chunk
from .chunker import Chunker

logger = logging.getLogger(__name__)


class LangChainChunker(Chunker):
    """Chunker using LangChain's MarkdownHeaderTextSplitter."""

    def __init__(
        self,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cache_dir: str = None,
        min_chunk_size: int = 0,
    ):
        """Initialize the LangChain chunker.

        Args:
            headers_to_split_on: List of (header_tag, header_name) tuples.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between chunks.
            cache_dir: Optional cache directory.
            min_chunk_size: Minimum size for chunks (0 = disabled).
        """
        super().__init__(cache_dir=cache_dir, min_chunk_size=min_chunk_size)
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

            self.header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        except ImportError:
            raise ImportError("LangChain text splitters not available. Install with: pip install pdfkb-mcp[langchain]")

    def chunk(self, markdown_content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk markdown content using LangChain's splitter.

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

            # Split the markdown content by headers
            header_splits = self.header_splitter.split_text(markdown_content)

            # Process each header split and further split if needed
            final_chunks = []
            for split in header_splits:
                # Extract the text content from the split
                if hasattr(split, "page_content"):
                    # LangChain Document object
                    text_content = split.page_content
                    split_metadata = getattr(split, "metadata", {})
                else:
                    # Plain string
                    text_content = str(split)
                    split_metadata = {}

                # If the split is small enough, use it as is
                if len(text_content) <= self.chunk_size:
                    final_chunks.append({"text": text_content, "metadata": split_metadata})
                else:
                    # Split large sections into smaller chunks
                    sub_chunks = self.text_splitter.split_text(text_content)
                    for sub_chunk in sub_chunks:
                        final_chunks.append({"text": sub_chunk, "metadata": split_metadata})

            # Convert to Chunk objects with metadata
            chunks = []
            for i, chunk_data in enumerate(final_chunks):
                chunk_text = chunk_data["text"].strip()
                if not chunk_text:  # Skip empty chunks
                    continue

                chunk_metadata = {
                    "chunk_strategy": "langchain_markdown_header",
                    "headers_to_split_on": self.headers_to_split_on,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                # Add header metadata if available
                if chunk_data["metadata"]:
                    chunk_metadata.update(chunk_data["metadata"])

                # Add any provided metadata
                chunk_metadata.update(metadata)

                chunk = Chunk(text=chunk_text, chunk_index=i, metadata=chunk_metadata)
                chunks.append(chunk)

            # Apply minimum chunk size filtering
            chunks = self._filter_small_chunks(chunks)

            logger.info(f"Created {len(chunks)} chunks using LangChain MarkdownHeaderTextSplitter")
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk markdown content with LangChain: {e}")
            # Fallback to simple text splitting
            return self._fallback_chunk(markdown_content, metadata)

    def _fallback_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Fallback chunking method using only RecursiveCharacterTextSplitter.

        Args:
            text: Text content to chunk.
            metadata: Document metadata.

        Returns:
            List of Chunk objects.
        """
        try:
            logger.warning("Using fallback chunking method")
            chunks = []
            text_chunks = self.text_splitter.split_text(text)

            for i, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue

                chunk_metadata = {
                    "chunk_strategy": "langchain_fallback",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                chunk_metadata.update(metadata)

                chunk = Chunk(text=chunk_text.strip(), chunk_index=i, metadata=chunk_metadata)
                chunks.append(chunk)

            # Apply minimum chunk size filtering
            chunks = self._filter_small_chunks(chunks)

            logger.info(f"Created {len(chunks)} chunks using fallback method")
            return chunks

        except Exception as e:
            logger.error(f"Fallback chunking also failed: {e}")
            raise RuntimeError(f"Failed to chunk text: {e}") from e
