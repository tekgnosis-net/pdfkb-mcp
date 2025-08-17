"""Abstract base class for document parsers (PDF, Markdown, etc.)."""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Content and metadata for a single page."""

    page_number: int
    markdown_content: str
    metadata: Dict[str, Any]


@dataclass
class ParseResult:
    """Result of parsing a document with page-aware content."""

    pages: List[PageContent]
    metadata: Dict[str, Any]

    def get_combined_markdown(self) -> str:
        """Get combined markdown content from all pages.

        Returns:
            Combined markdown content with page headers.
        """
        if not self.pages:
            return ""

        combined_parts = []
        for page in self.pages:
            # Add page header
            combined_parts.append(f"# Page {page.page_number}\n")
            combined_parts.append(page.markdown_content)

        return "\n\n".join(combined_parts)


class DocumentParser(ABC):
    """Abstract base class for document parsers.

    Supports parsing various document formats (PDF, Markdown, etc.) into
    a common markdown representation for downstream processing.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the parser with optional cache directory.

        Args:
            cache_dir: Directory to cache parsed markdown files.
        """
        self.cache_dir = cache_dir
        if self.cache_dir:
            (self.cache_dir / "markdown").mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def parse(self, file_path: Path) -> ParseResult:
        """Parse a document file and extract text and metadata.

        Args:
            file_path: Path to the document file.

        Returns:
            ParseResult with markdown content and metadata.
        """
        pass

    def _get_cache_path(self, file_path: Path) -> Path:
        """Get the cache path for a parsed markdown file.

        Args:
            file_path: Path to the document file.

        Returns:
            Path to the cached markdown file.
        """
        if not self.cache_dir:
            raise ValueError("Cache directory not configured")

        # Create a hash of the file path to use as cache key
        file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()
        return self.cache_dir / "markdown" / f"{file_hash}.md"

    def _is_cache_valid(self, file_path: Path, cache_path: Path) -> bool:
        """Check if the cached file is valid (newer than the source file).

        Args:
            file_path: Path to the source document file.
            cache_path: Path to the cached markdown file.

        Returns:
            True if cache is valid, False otherwise.
        """
        if not cache_path.exists():
            return False

        try:
            pdf_mtime = file_path.stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime
            return cache_mtime > pdf_mtime
        except OSError:
            return False

    def _load_from_cache(self, cache_path: Path) -> str:
        """Load markdown content from cache.

        Args:
            cache_path: Path to the cached markdown file.

        Returns:
            Markdown content from cache.
        """
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to load from cache {cache_path}: {e}")
            return ""

    def _save_to_cache(self, cache_path: Path, content: str) -> None:
        """Save markdown content to cache.

        Args:
            cache_path: Path to the cached markdown file.
            content: Markdown content to save.
        """
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_path}: {e}")

    def _save_metadata_to_cache(self, cache_path: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata to cache as JSON file.

        Args:
            cache_path: Path to the cached markdown file (used as base for metadata path).
            metadata: Metadata to save.
        """
        try:
            metadata_path = cache_path.with_suffix(".metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save metadata to cache {metadata_path}: {e}")

    def _load_metadata_from_cache(self, cache_path: Path) -> Dict[str, Any]:
        """Load metadata from cache.

        Args:
            cache_path: Path to the cached markdown file (used as base for metadata path).

        Returns:
            Metadata from cache or empty dict if not found.
        """
        try:
            metadata_path = cache_path.with_suffix(".metadata.json")
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)  # type: ignore[no-any-return]
            return {}
        except Exception as e:
            logger.warning(f"Failed to load metadata from cache {metadata_path}: {e}")
            return {}


# Backward compatibility alias
PDFParser = DocumentParser
