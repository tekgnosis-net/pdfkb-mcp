"""PDF parser using OpenRouter LLM integration for image-to-text transcription."""

import asyncio
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from .parser import DocumentParser, PageContent, ParseResult

logger = logging.getLogger(__name__)


class LLMParser(DocumentParser):
    """PDF parser using OpenRouter LLM integration for image-to-text transcription."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Path = None):
        """Initialize the LLM parser.

        Args:
            config: Configuration options for LLM parsing.
            cache_dir: Directory to cache parsed markdown files.
        """
        super().__init__(cache_dir)
        self.config = config or {}

        # Configuration with defaults
        self.model = self.config.get("model", "google/gemini-2.5-flash")
        self.openrouter_api_key = self.config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
        self.concurrency = self.config.get("concurrency", 5)
        self.dpi = self.config.get("dpi", 150)
        self.max_retries = self.config.get("max_retries", 3)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        # Validate configuration
        if not self.openrouter_api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass it in config."
            )

        # Setup Jinja2 environment
        self._setup_template_environment()

        # Import dependencies
        try:
            import fitz  # PyMuPDF
            import httpx
            from PIL import Image

            self.httpx = httpx
            self.fitz = fitz
            self.Image = Image
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not available: {e}. Install with: pip install httpx pymupdf pillow"
            )

    def _setup_template_environment(self):
        """Setup Jinja2 template environment."""
        if not JINJA2_AVAILABLE:
            logger.warning("Jinja2 not available, using hardcoded prompts. Install with: pip install jinja2")
            self.jinja_env = None
            return

        try:
            # Templates directory relative to this file
            templates_dir = Path(__file__).parent / "templates"

            if not templates_dir.exists():
                logger.warning(f"Templates directory not found: {templates_dir}. Using hardcoded prompts.")
                self.jinja_env = None
                return

            # Setup Jinja2 environment
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                autoescape=False,  # We're generating plain text prompts
                trim_blocks=True,
                lstrip_blocks=True,
            )

            logger.debug(f"Jinja2 environment initialized with templates from: {templates_dir}")

        except Exception as e:
            logger.warning(f"Failed to setup Jinja2 environment: {e}. Using hardcoded prompts.")
            self.jinja_env = None

    def _render_template(self, template_name: str, **kwargs) -> str:
        """Render a Jinja2 template with the given variables.

        Args:
            template_name: Name of the template file.
            **kwargs: Variables to pass to the template.

        Returns:
            Rendered template string.
        """
        if not self.jinja_env:
            # Fallback to hardcoded prompts if Jinja2 not available
            return self._get_fallback_prompt(template_name, **kwargs)

        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**kwargs)
        except TemplateNotFound:
            logger.warning(f"Template {template_name} not found, using fallback prompt")
            return self._get_fallback_prompt(template_name, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to render template {template_name}: {e}. Using fallback prompt.")
            return self._get_fallback_prompt(template_name, **kwargs)

    def _get_fallback_prompt(self, template_name: str, **kwargs) -> str:
        """Get fallback hardcoded prompts when templates are not available.

        Args:
            template_name: Name of the template.
            **kwargs: Template variables (ignored for fallback).

        Returns:
            Hardcoded prompt string.
        """
        if template_name == "page_transcription.j2":
            return """Convert this PDF page image to markdown format. Follow these guidelines:

1. Maintain accurate markdown syntax and structure
2. Preserve document hierarchy (headers, lists, tables)
3. Replace images with descriptive text in brackets like [Image: description]
4. Ensure 1:1 transcription accuracy (do not summarize)
5. Maintain proper formatting for tables, lists, and text blocks
6. Use appropriate header levels (# ## ###) based on visual hierarchy
7. Preserve any mathematical formulas or equations in text format
8. Include any footnotes or references found on the page

Return only the markdown content, no additional commentary."""

        elif template_name == "metadata_extraction.j2":
            return """Analyze these first few pages of a PDF document and extract:

1. **Title**: The main title of the document (not headers/footers)
2. **Summary**: A concise 2-3 sentence summary of what this document is about

Look for:
- Main document title (often largest text on first page)
- Abstract, introduction, or executive summary sections
- Key topics and themes

Return your response in this exact format:
TITLE: [extracted title]
SUMMARY: [2-3 sentence summary]

If you cannot determine a clear title or summary, return "TITLE: " or "SUMMARY: " with no content after the colon."""

        else:
            logger.warning(f"Unknown template name for fallback: {template_name}")
            return "Please analyze the provided content."

    async def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF file using LLM image-to-text transcription.

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
                        # Create page-aware result
                        # TODO: Implement proper page extraction for this parser
                        pages = [PageContent(page_number=1, markdown_content=markdown_content, metadata={})]
                        return ParseResult(pages=pages, metadata=metadata)

            logger.debug(f"Parsing PDF with LLM: {file_path}")
            start_time = time.time()

            # Open PDF and extract pages
            doc = self.fitz.open(str(file_path))
            total_pages = doc.page_count
            logger.debug(f"PDF has {total_pages} pages")

            # Convert pages to images
            page_images = await self._convert_pages_to_images(doc)
            doc.close()

            # Process pages concurrently
            page_transcriptions = await self._process_pages_concurrently(page_images)

            # Combine transcriptions into final markdown
            markdown_content = self._combine_page_transcriptions(page_transcriptions)

            # Extract metadata from first 5 pages
            metadata = await self._extract_metadata(file_path, page_images[:5], total_pages)

            # Add processing information
            processing_time = time.time() - start_time
            metadata.update(
                {
                    "processing_timestamp": "N/A",  # Will be set by PDFProcessor
                    "processor_version": "llm",
                    "source_filename": file_path.name,
                    "source_directory": str(file_path.parent),
                    "processing_time_seconds": processing_time,
                    "llm_model": self.model,
                    "page_count": total_pages,
                }
            )

            # Save to cache if enabled
            if cache_path:
                logger.debug(f"Saving parsed content to cache: {cache_path}")
                self._save_to_cache(cache_path, markdown_content)
                self._save_metadata_to_cache(cache_path, metadata)

            logger.debug(f"Successfully parsed PDF with LLM in {processing_time:.2f} seconds")
            # Create page-aware result
            # TODO: Implement proper page extraction for this parser
            pages = [PageContent(page_number=1, markdown_content=markdown_content, metadata={})]
            return ParseResult(pages=pages, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to parse PDF with LLM: {e}")
            raise RuntimeError(f"Failed to parse PDF with LLM: {e}") from e

    async def _convert_pages_to_images(self, doc) -> List[Tuple[int, str]]:
        """Convert PDF pages to PNG images.

        Args:
            doc: PyMuPDF document object.

        Returns:
            List of tuples (page_number, base64_encoded_image).
        """
        page_images = []

        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]

                # Convert page to image (150 DPI, RGB)
                mat = self.fitz.Matrix(self.dpi / 72, self.dpi / 72)  # Scale factor for DPI
                pix = page.get_pixmap(matrix=mat, colorspace=self.fitz.csRGB)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = self.Image.open(io.BytesIO(img_data))

                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                page_images.append((page_num + 1, img_base64))
                logger.debug(f"Converted page {page_num + 1} to image")

            except Exception as e:
                logger.warning(f"Failed to convert page {page_num + 1} to image: {e}")
                # Add placeholder for failed page
                page_images.append((page_num + 1, None))

        return page_images

    async def _process_pages_concurrently(self, page_images: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """Process pages concurrently with configurable concurrency.

        Args:
            page_images: List of tuples (page_number, base64_encoded_image).

        Returns:
            List of tuples (page_number, transcribed_markdown).
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_single_page(page_data: Tuple[int, str]) -> Tuple[int, str]:
            async with semaphore:
                page_num, img_base64 = page_data
                if img_base64 is None:
                    return (
                        page_num,
                        f"# Page {page_num}\n\n*[Error: Could not process this page]*\n",
                    )

                try:
                    transcription = await self._transcribe_image_to_markdown(img_base64, page_num)
                    return (page_num, transcription)
                except Exception as e:
                    logger.warning(f"Failed to transcribe page {page_num}: {e}")
                    return (
                        page_num,
                        f"# Page {page_num}\n\n*[Error: Could not transcribe this page: {str(e)}]*\n",
                    )

        # Process all pages concurrently
        tasks = [process_single_page(page_data) for page_data in page_images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions and sort by page number
        transcriptions = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Page processing failed: {result}")
                transcriptions.append((0, "*[Error: Page processing failed]*\n"))
            else:
                transcriptions.append(result)

        # Sort by page number
        transcriptions.sort(key=lambda x: x[0])
        return transcriptions

    async def _transcribe_image_to_markdown(self, img_base64: str, page_num: int) -> str:
        """Transcribe a single image to markdown using OpenRouter API.

        Args:
            img_base64: Base64 encoded image.
            page_num: Page number for context.

        Returns:
            Transcribed markdown content.
        """
        # Render prompt using template
        prompt = self._render_template(
            "page_transcription.j2",
            page_number=page_num,
            image_description_format="description",
            strict_transcription=True,
            preserve_formatting=True,
        )

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.1,  # Low temperature for consistent transcription
        }

        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                async with self.httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(self.base_url, json=payload, headers=headers)

                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        return f"# Page {page_num}\n\n{content}\n"

                    elif response.status_code == 429:  # Rate limit
                        wait_time = 2**attempt  # Exponential backoff
                        logger.warning(
                            f"Rate limited on page {page_num}, waiting {wait_time}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    elif response.status_code == 400:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "Unknown API error")
                        raise RuntimeError(f"API error: {error_msg}")

                    else:
                        response.raise_for_status()

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"API call failed for page {page_num}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to transcribe page {page_num} after {self.max_retries} attempts: {e}")

        # If all retries failed, return error placeholder
        raise RuntimeError(f"Failed to transcribe page {page_num} after {self.max_retries} attempts: {last_exception}")

    def _combine_page_transcriptions(self, page_transcriptions: List[Tuple[int, str]]) -> str:
        """Combine page transcriptions into final markdown document.

        Args:
            page_transcriptions: List of tuples (page_number, transcribed_markdown).

        Returns:
            Combined markdown content.
        """
        combined_content = []

        for page_num, transcription in page_transcriptions:
            if transcription and transcription.strip():
                combined_content.append(transcription.strip())

        return "\n\n---\n\n".join(combined_content)

    async def _extract_metadata(
        self, file_path: Path, first_pages: List[Tuple[int, str]], total_pages: int
    ) -> Dict[str, Any]:
        """Extract title and summary from first 5 pages using LLM.

        Args:
            file_path: Path to the PDF file.
            first_pages: First 5 pages as (page_number, base64_image) tuples.
            total_pages: Total number of pages in the document.

        Returns:
            Dictionary of extracted metadata.
        """
        metadata = {
            "page_count": total_pages,
            "title": "",
            "summary": "",
        }

        # Try to extract basic metadata using PyMuPDF
        try:
            doc = self.fitz.open(str(file_path))
            doc_metadata = doc.metadata
            if doc_metadata:
                metadata.update(
                    {
                        "author": doc_metadata.get("author", ""),
                        "subject": doc_metadata.get("subject", ""),
                        "creator": doc_metadata.get("creator", ""),
                        "producer": doc_metadata.get("producer", ""),
                        "creationDate": doc_metadata.get("creationDate", ""),
                        "modDate": doc_metadata.get("modDate", ""),
                    }
                )
                # Use PDF title if available
                if doc_metadata.get("title"):
                    metadata["title"] = doc_metadata["title"]
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to extract basic metadata: {e}")

        # Extract title and summary using LLM from first pages
        try:
            # Combine first few pages for context
            pages_for_metadata = [page for page in first_pages if page[1] is not None][:5]

            if pages_for_metadata:
                title, summary = await self._extract_title_and_summary(pages_for_metadata)
                if title:
                    metadata["title"] = title
                if summary:
                    metadata["summary"] = summary

        except Exception as e:
            logger.warning(f"Failed to extract title and summary with LLM: {e}")

        # Fallback title from filename if still empty
        if not metadata["title"]:
            metadata["title"] = file_path.stem.replace("_", " ").replace("-", " ").title()

        return metadata

    async def _extract_title_and_summary(self, pages: List[Tuple[int, str]]) -> Tuple[str, str]:
        """Extract title and summary from first pages using LLM.

        Args:
            pages: List of (page_number, base64_image) tuples.

        Returns:
            Tuple of (title, summary).
        """
        if not pages:
            return "", ""

        # Render prompt using template
        prompt = self._render_template(
            "metadata_extraction.j2",
            summary_length="2-3",
            page_count=len(pages),
            focus_areas=["Document structure and organization", "Main themes and objectives"],
            fallback_instructions=(
                "If the document appears to be academic, look for author names and publication details."
            ),
        )

        # Create message with multiple images
        message_content = [{"type": "text", "text": prompt}]

        for page_num, img_base64 in pages:
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        try:
            async with self.httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                    # Parse title and summary
                    title = ""
                    summary = ""

                    for line in content.split("\n"):
                        line = line.strip()
                        if line.startswith("TITLE:"):
                            title = line[6:].strip()
                        elif line.startswith("SUMMARY:"):
                            summary = line[8:].strip()

                    return title, summary
                else:
                    logger.warning(f"Failed to extract metadata, status: {response.status_code}")
                    return "", ""

        except Exception as e:
            logger.warning(f"Failed to extract title and summary: {e}")
            return "", ""
