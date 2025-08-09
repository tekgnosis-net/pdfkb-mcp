"""PDF parser using pymupdf4llm library."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .parser import ParseResult, PDFParser

logger = logging.getLogger(__name__)


class PyMuPDF4LLMParser(PDFParser):
    """PDF parser using pymupdf4llm library."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Optional[Path] = None):
        """Initialize the PyMuPDF4LLM parser.

        Args:
            config: Configuration options for pymupdf4llm.
            cache_dir: Directory to cache parsed markdown files.
        """
        super().__init__(cache_dir)
        self.config = config or {}

    async def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF file using pymupdf4llm library.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParseResult with markdown content and metadata.
        """
        try:
            # Check cache first
            cache_path = None
            if self.cache_dir:
                cache_path = self._get_cache_path(file_path)
                if self._is_cache_valid(file_path, cache_path):
                    logger.debug(f"Loading parsed content from cache: {cache_path}")
                    markdown_content = self._load_from_cache(cache_path)
                    metadata = self._load_metadata_from_cache(cache_path)
                    if markdown_content is not None and metadata:
                        return ParseResult(markdown_content=markdown_content, metadata=metadata)

            import pymupdf4llm

            logger.debug(f"Partitioning PDF with PyMuPDF4LLM: {file_path}")

            # Extract text using pymupdf4llm with configuration
            # Create a copy of config and remove the keys we're handling explicitly
            config_copy = dict(self.config)

            # Remove keys that we handle explicitly to avoid duplicate keyword arguments
            config_copy.pop("page_chunks", None)
            config_copy.pop("show_progress", None)

            # Use pymupdf4llm to convert PDF to markdown
            loop = asyncio.get_running_loop()
            md_text = await loop.run_in_executor(
                None,
                lambda: pymupdf4llm.to_markdown(
                    str(file_path),
                    page_chunks=True,  # Process by page for better chunking
                    show_progress=True,
                    **config_copy,
                ),
            )

            # Convert markdown text to proper format
            markdown_content = self._process_markdown_text(md_text)

            # Extract metadata
            metadata = self._extract_metadata(file_path)

            # Add processing information
            metadata["processing_timestamp"] = "N/A"  # Will be set by PDFProcessor
            metadata["processor_version"] = "pymupdf4llm"
            metadata["source_filename"] = file_path.name
            metadata["source_directory"] = str(file_path.parent)

            # Save to cache if enabled
            if cache_path:
                logger.debug(f"Saving parsed content to cache: {cache_path}")
                self._save_to_cache(cache_path, markdown_content)
                self._save_metadata_to_cache(cache_path, metadata)

            logger.debug("Extracted markdown content from PDF using PyMuPDF4LLM")
            return ParseResult(markdown_content=markdown_content, metadata=metadata)

        except ImportError:
            raise ImportError("PyMuPDF4LLM library not available. Install with: pip install pymupdf4llm")
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with PyMuPDF4LLM: {e}") from e

    def _process_markdown_text(self, md_text: Any) -> str:
        """Process markdown text from pymupdf4llm.

        Args:
            md_text: Markdown text extracted from PDF.

        Returns:
            Processed markdown content.
        """
        # Handle both string and list (page_chunks=True) cases
        if isinstance(md_text, list):
            # Process each page separately and combine
            pages_content = []
            for page_data in md_text:
                if isinstance(page_data, dict):
                    page_content = page_data.get("text", "")
                    # Add page header if available
                    page_number = page_data.get("metadata", {}).get("page_number")
                    if page_number:
                        pages_content.append(f"# Page {page_number}\n\n{page_content}")
                    else:
                        pages_content.append(page_content)
                else:
                    pages_content.append(str(page_data))
            return "\n\n".join(pages_content)
        else:
            return str(md_text)

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary of extracted metadata.
        """
        metadata = {}

        # Try to get page count and other metadata using PyMuPDF directly
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            metadata["page_count"] = doc.page_count

            # Try to extract document metadata
            doc_metadata = doc.metadata
            if doc_metadata:
                metadata.update(
                    {
                        "title": doc_metadata.get("title", ""),
                        "author": doc_metadata.get("author", ""),
                        "subject": doc_metadata.get("subject", ""),
                        "creator": doc_metadata.get("creator", ""),
                        "producer": doc_metadata.get("producer", ""),
                        "creationDate": doc_metadata.get("creationDate", ""),
                        "modDate": doc_metadata.get("modDate", ""),
                    }
                )
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to extract detailed metadata with PyMuPDF: {e}")
            metadata["page_count"] = 1  # Default fallback

        return metadata
