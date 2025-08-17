"""Page-based chunker that creates chunks on page boundaries."""

import re
from typing import Any, Dict, List, Optional

from pdfkb.chunker.chunker import Chunker
from pdfkb.models import Chunk
from pdfkb.parsers.parser import PageContent


class PageChunker(Chunker):
    """Chunker that creates chunks based on page boundaries.

    This chunker is designed to work with page-aware parsers that output
    PageContent objects. Each page becomes a separate chunk, preserving
    the natural document structure.
    """

    def __init__(
        self,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        merge_small: bool = True,
        global_min_chunk_size: int = 0,
        cache_dir: str = None,
    ):
        """Initialize the page chunker.

        Args:
            min_chunk_size: Minimum size for a chunk. Small pages may be merged.
            max_chunk_size: Maximum size for a chunk. Large pages may be split.
            merge_small: Whether to merge small consecutive pages.
            global_min_chunk_size: Global minimum chunk size (filters out unmergeable small chunks).
            cache_dir: Optional cache directory.
        """
        super().__init__(cache_dir=cache_dir, min_chunk_size=global_min_chunk_size)
        self.page_min_chunk_size = min_chunk_size  # Rename to avoid confusion
        self.max_chunk_size = max_chunk_size
        self.merge_small = merge_small

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Fallback method for non-page-aware content.

        This method is provided for compatibility but should not typically be used.
        PageChunker is designed to work with page-aware content via chunk_pages.

        Args:
            text: The text to chunk (entire document as string)
            metadata: Optional metadata for the document

        Returns:
            Single chunk containing the entire text
        """
        chunks = [
            Chunk(
                text=text,
                metadata={
                    **(metadata or {}),
                    "chunk_strategy": "page",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "warning": "PageChunker used without page-aware content",
                },
                chunk_index=0,
                document_id="",
            )
        ]

        # Apply global minimum chunk size filtering
        return self._filter_small_chunks(chunks)

    def chunk_pages(self, pages: List[PageContent], metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk page-aware content by creating one chunk per page.

        Args:
            pages: List of PageContent objects from a page-aware parser
            metadata: Optional document-level metadata

        Returns:
            List of chunks, typically one per page (may merge small pages)
        """
        if not pages:
            return []

        chunks = []
        current_chunk_text = ""
        current_chunk_pages = []

        for page in pages:
            page_text = page.markdown_content

            # Remove page boundary marker from the beginning of the content
            # This handles markers like --[PAGE: 1]-- at the start of the page
            page_marker_pattern = r"^--\[PAGE:\s*\d+\]--\s*\n?"
            page_text = re.sub(page_marker_pattern, "", page_text, count=1)

            # Also check if page has a marker in metadata and clean it
            if page.metadata.get("has_page_marker") and page.metadata.get("page_marker"):
                marker = page.metadata["page_marker"]
                if page_text.startswith(marker):
                    page_text = page_text[len(marker) :].lstrip()

            page_size = len(page_text)

            # Check if we should merge this page with the current chunk
            should_merge = False
            if self.merge_small and self.page_min_chunk_size and current_chunk_text:
                current_size = len(current_chunk_text)
                combined_size = current_size + page_size + 2  # +2 for "\n\n"

                # Only merge if:
                # 1. Current chunk is below minimum size AND
                # 2. The page itself is also small (below minimum) AND
                # 3. Combined size won't exceed max (if set)
                if current_size < self.page_min_chunk_size and page_size < self.page_min_chunk_size:
                    if not self.max_chunk_size or combined_size <= self.max_chunk_size:
                        should_merge = True

            if should_merge:
                # Merge with current chunk
                if page_text.strip():  # Only add if there's actual content
                    current_chunk_text += "\n\n" + page_text
                    current_chunk_pages.append(page.page_number)
            else:
                # Finalize current chunk if it exists
                if current_chunk_text:
                    chunks.append(self._create_chunk(current_chunk_text, current_chunk_pages, len(chunks), metadata))

                # Start new chunk with current page
                current_chunk_text = page_text
                current_chunk_pages = [page.page_number]

        # Don't forget the last chunk
        if current_chunk_text:
            chunks.append(self._create_chunk(current_chunk_text, current_chunk_pages, len(chunks), metadata))

        # Handle large chunks if max_chunk_size is set
        if self.max_chunk_size:
            chunks = self._split_large_chunks(chunks)

        # Apply global minimum chunk size filtering
        chunks = self._filter_small_chunks(chunks)

        # Update total chunks count
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata["total_chunks"] = total_chunks
            chunk.chunk_index = i

        return chunks

    def _create_chunk(
        self, text: str, page_numbers: List[int], chunk_index: int, metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """Create a chunk with appropriate metadata.

        Args:
            text: The chunk text
            page_numbers: List of page numbers included in this chunk
            chunk_index: Index of this chunk
            metadata: Optional document-level metadata

        Returns:
            A Chunk object with page metadata
        """
        chunk_metadata = {
            **(metadata or {}),
            "chunk_strategy": "page",
            "chunk_index": chunk_index,
            "page_numbers": page_numbers,
            "page_count": len(page_numbers),
        }

        # Add specific page metadata
        if len(page_numbers) == 1:
            chunk_metadata["page_number"] = page_numbers[0]
            chunk_metadata["single_page"] = True
        else:
            chunk_metadata["page_range"] = f"{page_numbers[0]}-{page_numbers[-1]}"
            chunk_metadata["merged_pages"] = True

        return Chunk(
            text=text,
            metadata=chunk_metadata,
            chunk_index=chunk_index,
            document_id="",
        )

    def _split_large_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Split chunks that exceed max_chunk_size.

        Args:
            chunks: List of chunks to potentially split

        Returns:
            List of chunks with large ones split
        """
        if not self.max_chunk_size:
            return chunks

        result = []
        for chunk in chunks:
            if len(chunk.text) <= self.max_chunk_size:
                result.append(chunk)
            else:
                # Split by sentences or paragraphs
                parts = self._split_text(chunk.text, self.max_chunk_size)
                for i, part in enumerate(parts):
                    new_chunk = Chunk(
                        text=part,
                        metadata={
                            **chunk.metadata,
                            "split_from_large_page": True,
                            "split_part": i + 1,
                            "split_total": len(parts),
                        },
                        chunk_index=0,  # Will be updated later
                        document_id=chunk.document_id,
                    )
                    result.append(new_chunk)

        return result

    def _split_text(self, text: str, max_size: int) -> List[str]:
        """Split text into parts not exceeding max_size.

        Tries to split on paragraph boundaries first, then sentences.

        Args:
            text: Text to split
            max_size: Maximum size for each part

        Returns:
            List of text parts
        """
        # Try splitting by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        parts = []
        current_part = ""

        for para in paragraphs:
            if len(current_part) + len(para) + 2 <= max_size:
                if current_part:
                    current_part += "\n\n" + para
                else:
                    current_part = para
            else:
                if current_part:
                    parts.append(current_part)

                # If paragraph itself is too large, split it further
                if len(para) > max_size:
                    # Simple sentence split (crude but functional)
                    sentences = para.replace(". ", ".|").split("|")
                    para_part = ""
                    for sent in sentences:
                        if len(para_part) + len(sent) <= max_size:
                            para_part += sent
                        else:
                            if para_part:
                                parts.append(para_part)
                            para_part = sent
                    if para_part:
                        parts.append(para_part)
                else:
                    current_part = para

        if current_part:
            parts.append(current_part)

        return parts
