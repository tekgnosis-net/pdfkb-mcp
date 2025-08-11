"""PDF parser using pymupdf4llm library with robust error handling."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parser import ParseResult, PDFParser

logger = logging.getLogger(__name__)


class PyMuPDF4LLMParser(PDFParser):
    """PDF parser using pymupdf4llm library with robust error handling for complex PDFs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Optional[Path] = None):
        """Initialize the PyMuPDF4LLM parser.

        Args:
            config: Configuration options for pymupdf4llm.
            cache_dir: Directory to cache parsed markdown files.
        """
        super().__init__(cache_dir)
        self.config = config or {}
        # Configure robust extraction options
        self.enable_ocr_fallback = self.config.get("enable_ocr_fallback", True)
        self.min_text_length = self.config.get("min_text_length", 20)  # Min chars to consider page has text
        self.min_words_count = self.config.get("min_words_count", 5)  # Min words to consider page has text

    async def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF file using pymupdf4llm library with robust error handling.

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

            # Try the fast path first with pymupdf4llm
            try:
                result = await self._parse_with_pymupdf4llm(file_path)

                # Save to cache if enabled
                if cache_path:
                    logger.debug(f"Saving parsed content to cache: {cache_path}")
                    self._save_to_cache(cache_path, result.markdown_content)
                    self._save_metadata_to_cache(cache_path, result.metadata)

                return result

            except Exception as e:
                if "not a textpage" in str(e).lower():
                    logger.warning(f"PyMuPDF4LLM fast path failed with textpage error, trying robust extraction: {e}")
                    # Fall back to robust page-by-page extraction
                    result = await self._parse_robust(file_path)

                    # Save to cache if enabled
                    if cache_path:
                        logger.debug(f"Saving parsed content to cache: {cache_path}")
                        self._save_to_cache(cache_path, result.markdown_content)
                        self._save_metadata_to_cache(cache_path, result.metadata)

                    return result
                else:
                    raise

        except ImportError:
            raise ImportError("PyMuPDF4LLM library not available. Install with: pip install pymupdf4llm")
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with PyMuPDF4LLM: {e}") from e

    async def _parse_with_pymupdf4llm(self, file_path: Path) -> ParseResult:
        """Try standard pymupdf4llm parsing.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParseResult with markdown content and metadata.
        """
        import pymupdf4llm

        logger.debug(f"Attempting standard PyMuPDF4LLM parsing: {file_path}")

        # Extract text using pymupdf4llm with configuration
        config_copy = dict(self.config)
        config_copy.pop("page_chunks", None)
        config_copy.pop("show_progress", None)
        config_copy.pop("enable_ocr_fallback", None)
        config_copy.pop("min_text_length", None)
        config_copy.pop("min_words_count", None)

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
        metadata["extraction_method"] = "standard"
        metadata["source_filename"] = file_path.name
        metadata["source_directory"] = str(file_path.parent)

        logger.debug("Successfully extracted markdown content using standard PyMuPDF4LLM")
        return ParseResult(markdown_content=markdown_content, metadata=metadata)

    async def _parse_robust(self, file_path: Path) -> ParseResult:
        """Robust page-by-page extraction with error handling and OCR fallback.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParseResult with markdown content and metadata.
        """
        import fitz  # PyMuPDF

        logger.info(f"Using robust page-by-page extraction for: {file_path}")

        # Open document
        doc = fitz.open(str(file_path))
        pages_content = []
        ocr_pages = []
        failed_pages = []

        try:
            total_pages = doc.page_count

            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    page_text = ""
                    extraction_method = "native"

                    # Try native text extraction first
                    try:
                        # Use high-level get_text to avoid textpage issues
                        page_text = page.get_text("text")

                        # Check if we got meaningful text
                        if not self._has_meaningful_text(page_text):
                            # Try blocks extraction for better structure
                            blocks = page.get_text("blocks")
                            if blocks:
                                page_text = self._blocks_to_text(blocks)

                            # If still not enough text, try OCR if enabled
                            if not self._has_meaningful_text(page_text) and self.enable_ocr_fallback:
                                page_text = self._extract_with_ocr(page)
                                extraction_method = "ocr"
                                ocr_pages.append(page_num + 1)

                    except Exception as e:
                        logger.warning(f"Native extraction failed for page {page_num + 1}: {e}")

                        # Try OCR fallback if enabled
                        if self.enable_ocr_fallback:
                            try:
                                page_text = self._extract_with_ocr(page)
                                extraction_method = "ocr"
                                ocr_pages.append(page_num + 1)
                            except Exception as ocr_e:
                                logger.error(f"OCR also failed for page {page_num + 1}: {ocr_e}")
                                page_text = f"[Page {page_num + 1}: Extraction failed]"
                                failed_pages.append(page_num + 1)
                        else:
                            page_text = f"[Page {page_num + 1}: Extraction failed]"
                            failed_pages.append(page_num + 1)

                    # Format page content as markdown
                    page_md = self._format_page_markdown(page_num + 1, page_text, extraction_method)
                    pages_content.append(page_md)

                    # Show progress
                    if page_num % 10 == 0:
                        logger.debug(f"Processed {page_num + 1}/{total_pages} pages")

                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")
                    pages_content.append(f"# Page {page_num + 1}\n\n[Extraction failed: {e}]\n")
                    failed_pages.append(page_num + 1)

            # Combine all pages
            markdown_content = "\n\n".join(pages_content)

            # Extract metadata
            metadata = self._extract_metadata_from_doc(doc, file_path)

            # Add extraction statistics
            metadata["processing_timestamp"] = "N/A"
            metadata["processor_version"] = "pymupdf4llm_robust"
            metadata["extraction_method"] = "robust"
            metadata["total_pages"] = total_pages
            metadata["ocr_pages"] = ocr_pages
            metadata["failed_pages"] = failed_pages
            metadata["ocr_page_count"] = len(ocr_pages)
            metadata["failed_page_count"] = len(failed_pages)
            metadata["source_filename"] = file_path.name
            metadata["source_directory"] = str(file_path.parent)

            if ocr_pages:
                logger.info(f"Used OCR for {len(ocr_pages)} pages: {ocr_pages[:10]}...")
            if failed_pages:
                logger.warning(f"Failed to extract {len(failed_pages)} pages: {failed_pages[:10]}...")

            return ParseResult(markdown_content=markdown_content, metadata=metadata)

        finally:
            doc.close()

    def _has_meaningful_text(self, text: str) -> bool:
        """Check if extracted text has meaningful content.

        Args:
            text: Extracted text to check.

        Returns:
            True if text seems meaningful, False otherwise.
        """
        if not text:
            return False

        cleaned = text.strip()
        if len(cleaned) < self.min_text_length:
            return False

        # Count words (simple split)
        words = cleaned.split()
        if len(words) < self.min_words_count:
            return False

        return True

    def _blocks_to_text(self, blocks: List) -> str:
        """Convert blocks to text.

        Args:
            blocks: List of text blocks from PyMuPDF.

        Returns:
            Combined text from blocks.
        """
        text_parts = []
        for block in blocks:
            if isinstance(block, (list, tuple)) and len(block) >= 5:
                # Block format: (x0, y0, x1, y1, text, block_no, block_type)
                block_text = str(block[4]) if len(block) > 4 else ""
                if block_text.strip():
                    text_parts.append(block_text.strip())
        return "\n".join(text_parts)

    def _extract_with_ocr(self, page) -> str:
        """Extract text from page using OCR.

        Args:
            page: PyMuPDF page object.

        Returns:
            OCR-extracted text.
        """
        try:
            # Try to get OCR textpage
            tp = page.get_textpage_ocr(dpi=300, full=True)
            text = page.get_text("text", textpage=tp)
            return text or "[OCR extraction produced no text]"
        except Exception as e:
            logger.debug(f"OCR extraction failed: {e}")
            # As a last resort, note that the page likely contains only images
            return "[Page appears to contain only images or non-text content]"

    def _format_page_markdown(self, page_num: int, text: str, extraction_method: str) -> str:
        """Format page content as markdown.

        Args:
            page_num: Page number (1-indexed).
            text: Page text content.
            extraction_method: Method used to extract text.

        Returns:
            Formatted markdown for the page.
        """
        header = f"# Page {page_num}"
        if extraction_method == "ocr":
            header += " [OCR]"
        elif extraction_method == "failed":
            header += " [Failed]"

        return f"{header}\n\n{text}"

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
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            metadata = self._extract_metadata_from_doc(doc, file_path)
            doc.close()
            return metadata
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return {"page_count": 1}

    def _extract_metadata_from_doc(self, doc, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from an open PyMuPDF document.

        Args:
            doc: Open PyMuPDF document.
            file_path: Path to the PDF file.

        Returns:
            Dictionary of extracted metadata.
        """
        metadata = {}

        try:
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
        except Exception as e:
            logger.warning(f"Failed to extract detailed metadata: {e}")
            metadata["page_count"] = 1  # Default fallback

        return metadata
