"""PDF parser using pymupdf4llm library with robust error handling."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parser import DocumentParser, PageContent, ParseResult

logger = logging.getLogger(__name__)


class PyMuPDF4LLMParser(DocumentParser):
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
        """Parse a PDF file using pymupdf4llm library with page-aware extraction.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParseResult with page-aware content and metadata.
        """
        try:
            # Check cache first
            cache_path = None
            if self.cache_dir:
                cache_path = self._get_cache_path(file_path)
                if self._is_cache_valid(file_path, cache_path):
                    logger.debug(f"Loading parsed content from cache: {cache_path}")
                    cached_pages = self._load_pages_from_cache(cache_path)
                    metadata = self._load_metadata_from_cache(cache_path)
                    if cached_pages is not None and metadata:
                        return ParseResult(pages=cached_pages, metadata=metadata)

            # Try the fast path first with pymupdf4llm
            try:
                result = await self._parse_with_pymupdf4llm(file_path)

                # Save to cache if enabled
                if cache_path:
                    logger.debug(f"Saving parsed content to cache: {cache_path}")
                    self._save_pages_to_cache(cache_path, result.pages)
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
                        self._save_pages_to_cache(cache_path, result.pages)
                        self._save_metadata_to_cache(cache_path, result.metadata)

                    return result
                else:
                    raise

        except ImportError:
            raise ImportError("PyMuPDF4LLM library not available. Install with: pip install pymupdf4llm")
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with PyMuPDF4LLM: {e}") from e

    async def _parse_with_pymupdf4llm(self, file_path: Path) -> ParseResult:
        """Try standard pymupdf4llm parsing with page-by-page extraction.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParseResult with page-aware content and metadata.
        """
        import fitz  # PyMuPDF
        import pymupdf4llm

        logger.debug(f"Attempting page-by-page PyMuPDF4LLM parsing: {file_path}")

        # Extract text using pymupdf4llm with configuration
        config_copy = dict(self.config)
        config_copy.pop("page_chunks", None)
        config_copy.pop("show_progress", None)
        config_copy.pop("enable_ocr_fallback", None)
        config_copy.pop("min_text_length", None)
        config_copy.pop("min_words_count", None)

        # Open the PDF to get page count
        doc = fitz.open(str(file_path))
        total_pages = doc.page_count
        doc.close()

        # Extract each page individually
        pages = []
        loop = asyncio.get_running_loop()

        for page_num in range(total_pages):
            try:
                # Extract markdown for this specific page
                page_md = await loop.run_in_executor(
                    None,
                    lambda pn=page_num: pymupdf4llm.to_markdown(
                        str(file_path),
                        pages=[pn],  # Extract only this page
                        **config_copy,
                    ),
                )

                # Process the markdown content for this page
                if isinstance(page_md, list) and page_md:
                    # If it returns a list, get the first element
                    page_content = page_md[0].get("text", "") if isinstance(page_md[0], dict) else str(page_md[0])
                else:
                    page_content = str(page_md) if page_md else ""

                # Create PageContent for this page
                page = PageContent(
                    page_number=page_num + 1,  # 1-indexed page numbers
                    markdown_content=page_content,
                    metadata={"page_index": page_num},
                )
                pages.append(page)

                # Log progress every 10 pages
                if (page_num + 1) % 10 == 0:
                    logger.debug(f"Processed {page_num + 1}/{total_pages} pages")

            except Exception as e:
                logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                # Add empty page with error message
                pages.append(
                    PageContent(
                        page_number=page_num + 1,
                        markdown_content=f"[Page {page_num + 1}: Extraction failed - {e}]",
                        metadata={"extraction_error": str(e), "page_index": page_num},
                    )
                )

        # Extract document metadata
        metadata = self._extract_metadata(file_path)

        # Add processing information
        metadata["processing_timestamp"] = "N/A"  # Will be set by PDFProcessor
        metadata["processor_version"] = "pymupdf4llm"
        metadata["extraction_method"] = "page-by-page"
        metadata["source_filename"] = file_path.name
        metadata["source_directory"] = str(file_path.parent)
        metadata["total_pages"] = total_pages

        logger.debug(f"Successfully extracted {len(pages)} pages using PyMuPDF4LLM")
        return ParseResult(pages=pages, metadata=metadata)

    async def _parse_robust(self, file_path: Path) -> ParseResult:
        """Robust page-by-page extraction with error handling and OCR fallback.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParseResult with page-aware content and metadata.
        """
        import fitz  # PyMuPDF

        logger.info(f"Using robust page-by-page extraction for: {file_path}")

        # Open document
        doc = fitz.open(str(file_path))
        pages = []
        ocr_pages = []
        failed_pages = []

        try:
            total_pages = doc.page_count

            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    page_text = ""
                    extraction_method = "native"
                    page_metadata = {"page_index": page_num}

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
                                page_text = "[Extraction failed]"
                                page_metadata["extraction_error"] = str(ocr_e)
                                failed_pages.append(page_num + 1)
                        else:
                            page_text = "[Extraction failed]"
                            page_metadata["extraction_error"] = str(e)
                            failed_pages.append(page_num + 1)

                    # Add extraction method to metadata
                    page_metadata["extraction_method"] = extraction_method

                    # Create PageContent for this page
                    page_content = PageContent(
                        page_number=page_num + 1, markdown_content=page_text, metadata=page_metadata
                    )
                    pages.append(page_content)

                    # Show progress
                    if page_num % 10 == 0:
                        logger.debug(f"Processed {page_num + 1}/{total_pages} pages")

                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")
                    pages.append(
                        PageContent(
                            page_number=page_num + 1,
                            markdown_content=f"[Extraction failed: {e}]",
                            metadata={"extraction_error": str(e), "page_index": page_num},
                        )
                    )
                    failed_pages.append(page_num + 1)

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

            return ParseResult(pages=pages, metadata=metadata)

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

    def _save_pages_to_cache(self, cache_path: Path, pages: List[PageContent]) -> None:
        """Save pages to cache as JSON.

        Args:
            cache_path: Path to the cache file.
            pages: List of PageContent objects to save.
        """
        try:
            pages_data = []
            for page in pages:
                pages_data.append(
                    {
                        "page_number": page.page_number,
                        "markdown_content": page.markdown_content,
                        "metadata": page.metadata,
                    }
                )

            pages_path = cache_path.with_suffix(".pages.json")
            with open(pages_path, "w", encoding="utf-8") as f:
                json.dump(pages_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save pages to cache {pages_path}: {e}")

    def _load_pages_from_cache(self, cache_path: Path) -> Optional[List[PageContent]]:
        """Load pages from cache.

        Args:
            cache_path: Path to the cache file.

        Returns:
            List of PageContent objects or None if cache is invalid.
        """
        try:
            pages_path = cache_path.with_suffix(".pages.json")
            if not pages_path.exists():
                return None

            with open(pages_path, "r", encoding="utf-8") as f:
                pages_data = json.load(f)

            pages = []
            for page_data in pages_data:
                pages.append(
                    PageContent(
                        page_number=page_data["page_number"],
                        markdown_content=page_data["markdown_content"],
                        metadata=page_data.get("metadata", {}),
                    )
                )

            return pages
        except Exception as e:
            logger.warning(f"Failed to load pages from cache {cache_path}: {e}")
            return None
