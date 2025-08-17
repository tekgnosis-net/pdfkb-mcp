"""Markdown document parser implementation."""

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .parser import DocumentParser, PageContent, ParseResult

logger = logging.getLogger(__name__)


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents.

    This parser reads markdown files directly and extracts metadata from
    YAML/TOML frontmatter if present. Since the content is already in markdown
    format, no conversion is needed - just metadata extraction and validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Optional[Path] = None):
        """Initialize the Markdown parser.

        Args:
            config: Optional configuration dict with:
                - parse_frontmatter: Whether to parse YAML/TOML frontmatter (default: True)
                - extract_title: Whether to extract title from first H1 (default: True)
                - page_boundary_pattern: Regex pattern to detect page boundaries (default: r'--\\[PAGE:\\s*(\\d+)\\]--')
                - split_on_page_boundaries: Whether to split content into pages (default: True)
            cache_dir: Not used for markdown (no caching needed for direct reads)
        """
        # Don't pass cache_dir to parent since we don't cache markdown parsing
        super().__init__(cache_dir=None)
        self.config = config or {}
        self.parse_frontmatter = self.config.get("parse_frontmatter", True)
        self.extract_title = self.config.get("extract_title", True)
        self.page_boundary_pattern = self.config.get("page_boundary_pattern", r"--\[PAGE:\s*(\d+)\]--")
        self.split_on_page_boundaries = self.config.get("split_on_page_boundaries", True)

    async def parse(self, file_path: Path) -> ParseResult:
        """Parse a markdown file and extract content and metadata.

        Args:
            file_path: Path to the markdown file.

        Returns:
            ParseResult with markdown content and metadata.
        """
        try:
            # Read file content asynchronously
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, self._read_file, file_path)

            # Extract frontmatter and content
            metadata, markdown_content = self._extract_frontmatter(content)

            # Add file metadata
            file_stats = file_path.stat()
            metadata.update(
                {
                    "source_filename": file_path.name,
                    "source_directory": str(file_path.parent),
                    "file_size": file_stats.st_size,
                    "modified_time": datetime.fromtimestamp(file_stats.st_mtime, tz=timezone.utc).isoformat(),
                    "document_type": "markdown",
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Extract title if configured
            if self.extract_title and "title" not in metadata:
                title = self._extract_title_from_content(markdown_content)
                if title:
                    metadata["title"] = title

            # Count basic statistics
            lines = markdown_content.split("\n")
            metadata["line_count"] = len(lines)
            metadata["word_count"] = len(markdown_content.split())
            metadata["char_count"] = len(markdown_content)

            # Count markdown elements
            metadata["heading_count"] = len(re.findall(r"^#+\s", markdown_content, re.MULTILINE))
            metadata["link_count"] = len(re.findall(r"\[([^\]]+)\]\(([^)]+)\)", markdown_content))
            metadata["code_block_count"] = len(re.findall(r"^```", markdown_content, re.MULTILINE)) // 2

            logger.info(f"Successfully parsed markdown file: {file_path.name}")

            # Split content into pages if configured
            if self.split_on_page_boundaries and self.page_boundary_pattern:
                pages = self._split_into_pages(markdown_content)
            else:
                # Single page for entire document
                pages = [PageContent(page_number=1, markdown_content=markdown_content, metadata={})]

            # Update page count in metadata
            metadata["page_count"] = len(pages)

            return ParseResult(pages=pages, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to parse markdown file {file_path}: {e}")
            raise

    def _read_file(self, file_path: Path) -> str:
        """Read file content synchronously.

        Args:
            file_path: Path to the file.

        Returns:
            File content as string.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML/TOML frontmatter from markdown content.

        Args:
            content: Full markdown content.

        Returns:
            Tuple of (metadata dict, markdown content without frontmatter).
        """
        metadata = {}
        markdown_content = content

        if not self.parse_frontmatter:
            return metadata, markdown_content

        # Check for YAML frontmatter (--- ... ---)
        yaml_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        yaml_match = re.match(yaml_pattern, content, re.DOTALL)

        if yaml_match:
            try:
                frontmatter_text = yaml_match.group(1)
                metadata = yaml.safe_load(frontmatter_text) or {}
                # Remove frontmatter from content
                markdown_content = content[yaml_match.end() :]
                logger.debug(f"Extracted YAML frontmatter with {len(metadata)} fields")
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse YAML frontmatter: {e}")
                # Keep original content if parsing fails

        # Alternative: Check for TOML frontmatter (+++ ... +++)
        if not metadata:
            toml_pattern = r"^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n"
            toml_match = re.match(toml_pattern, content, re.DOTALL)

            if toml_match:
                try:
                    import toml

                    frontmatter_text = toml_match.group(1)
                    metadata = toml.loads(frontmatter_text) or {}
                    # Remove frontmatter from content
                    markdown_content = content[toml_match.end() :]
                    logger.debug(f"Extracted TOML frontmatter with {len(metadata)} fields")
                except (ImportError, Exception) as e:
                    logger.warning(f"Failed to parse TOML frontmatter: {e}")
                    # Keep original content if parsing fails

        return metadata, markdown_content

    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from markdown content.

        Looks for the first H1 heading in the content.

        Args:
            content: Markdown content.

        Returns:
            Title string if found, None otherwise.
        """
        # Look for first H1 heading
        h1_pattern = r"^#\s+(.+)$"
        match = re.search(h1_pattern, content, re.MULTILINE)

        if match:
            title = match.group(1).strip()
            # Remove any markdown formatting from title
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)  # Bold
            title = re.sub(r"\*(.+?)\*", r"\1", title)  # Italic
            title = re.sub(r"`(.+?)`", r"\1", title)  # Code
            title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)  # Links
            return title

        # No fallback - only return title if we found an H1
        return None

    def _split_into_pages(self, content: str) -> list[PageContent]:
        """Split markdown content into pages based on page boundary markers.

        Args:
            content: Markdown content potentially containing page markers.

        Returns:
            List of PageContent objects, one per page.
        """
        pages = []

        # Find all page markers and their positions
        pattern = re.compile(self.page_boundary_pattern)
        matches = list(pattern.finditer(content))

        if not matches:
            # No page markers found, return single page
            return [PageContent(page_number=1, markdown_content=content, metadata={"has_page_marker": False})]

        # Extract page numbers and positions
        page_info = []
        for match in matches:
            try:
                # Try to extract page number from the match
                if match.groups():
                    page_num = int(match.group(1))
                else:
                    # If no capture group, use sequential numbering
                    page_num = len(page_info) + 1

                page_info.append(
                    {"page_number": page_num, "start": match.start(), "end": match.end(), "marker": match.group(0)}
                )
            except (ValueError, IndexError):
                # If we can't extract page number, use sequential
                page_info.append(
                    {
                        "page_number": len(page_info) + 1,
                        "start": match.start(),
                        "end": match.end(),
                        "marker": match.group(0),
                    }
                )

        # Handle content before first page marker
        if page_info[0]["start"] > 0:
            pre_content = content[: page_info[0]["start"]].strip()
            if pre_content:
                # Check if it's just frontmatter that was already extracted
                if not (self.parse_frontmatter and (pre_content.startswith("---") or pre_content.startswith("+++"))):
                    pages.append(
                        PageContent(
                            page_number=0,  # Page 0 for content before first marker
                            markdown_content=pre_content,
                            metadata={"has_page_marker": False, "before_first_page": True},
                        )
                    )

        # Extract content for each page
        for i, info in enumerate(page_info):
            # Determine where this page's content ends
            if i < len(page_info) - 1:
                # Content goes until the next page marker
                content_end = page_info[i + 1]["start"]
            else:
                # Last page - content goes to end of document
                content_end = len(content)

            # Extract page content (including the marker)
            page_content = content[info["start"] : content_end].strip()

            pages.append(
                PageContent(
                    page_number=info["page_number"],
                    markdown_content=page_content,
                    metadata={"has_page_marker": True, "page_marker": info["marker"]},
                )
            )

        # Sort pages by page number (in case they're out of order)
        pages.sort(key=lambda p: p.page_number)

        # Renumber pages sequentially if needed
        for i, page in enumerate(pages):
            if page.page_number == 0:
                # Keep page 0 as is (content before first marker)
                continue
            expected_num = i if pages[0].page_number == 0 else i + 1
            if page.page_number != expected_num:
                logger.debug(f"Renumbering page {page.page_number} to {expected_num}")
                page.metadata["original_page_number"] = page.page_number
                page.page_number = expected_num

        logger.info(f"Split markdown into {len(pages)} pages using pattern: {self.page_boundary_pattern}")

        return pages
