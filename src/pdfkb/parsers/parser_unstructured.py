"""PDF parser using the Unstructured library."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from .parser import ParseResult, PDFParser

logger = logging.getLogger(__name__)


class UnstructuredPDFParser(PDFParser):
    """PDF parser using the Unstructured library."""

    def __init__(self, strategy: str = "fast", cache_dir: Path = None):
        """Initialize the Unstructured parser.

        Args:
            strategy: PDF processing strategy ("fast" or "hi_res").
            cache_dir: Directory to cache parsed markdown files.
        """
        super().__init__(cache_dir)
        self.strategy = strategy

    async def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF file using Unstructured library.

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

            from unstructured.partition.pdf import partition_pdf

            logger.debug(f"Partitioning PDF with Unstructured using '{self.strategy}' strategy: {file_path}")

            # Partition the PDF with enhanced options including strategy
            elements = partition_pdf(
                filename=str(file_path),
                strategy=self.strategy,  # Use configured strategy ("fast" or "hi_res")
                extract_images_in_pdf=False,  # Skip image extraction for performance
                infer_table_structure=True,  # Extract table structure
                chunking_strategy="by_title",  # Group by document structure
                max_characters=1000,  # Default value, will be overridden by config
                new_after_n_chars=800,  # Default value, will be overridden by config
                combine_text_under_n_chars=100,  # Combine small elements
            )

            # Convert elements to markdown
            markdown_content = self._elements_to_markdown(elements)

            # Extract metadata from elements
            metadata = self._extract_metadata_from_elements(elements)

            # Add processing information
            metadata["processing_timestamp"] = "N/A"  # Will be set by PDFProcessor
            metadata["processor_version"] = "unstructured"
            metadata["source_filename"] = file_path.name
            metadata["source_directory"] = str(file_path.parent)

            # Save to cache if enabled
            if cache_path:
                logger.debug(f"Saving parsed content to cache: {cache_path}")
                self._save_to_cache(cache_path, markdown_content)
                self._save_metadata_to_cache(cache_path, metadata)

            logger.debug(f"Extracted {len(elements)} elements from PDF using Unstructured")
            return ParseResult(markdown_content=markdown_content, metadata=metadata)

        except ImportError:
            raise ImportError("Unstructured library not available. Install with: pip install unstructured[pdf]")
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with Unstructured: {e}") from e

    def _elements_to_markdown(self, elements: List[Any]) -> str:
        """Convert Unstructured elements to Markdown format.

        Args:
            elements: List of Unstructured elements.

        Returns:
            Markdown formatted text.
        """
        markdown_lines = []

        for element in elements:
            element_text = str(element).strip()
            if not element_text:
                continue

            element_type = type(element).__name__

            # Format based on element type
            if element_type in ["Title", "Header"]:
                # Add as headers
                header_level = min(len(element_text.split()), 6)  # Max 6 levels
                markdown_lines.append(f"{'#' * header_level} {element_text}")
            elif element_type == "Table":
                # Format tables
                markdown_lines.append(f"[TABLE]\n{element_text}\n[/TABLE]")
            elif element_type == "ListItem":
                # Format list items
                markdown_lines.append(f"- {element_text}")
            else:
                # Regular text
                markdown_lines.append(element_text)

        return "\n\n".join(markdown_lines)

    def _extract_metadata_from_elements(self, elements: List[Any]) -> Dict[str, Any]:
        """Extract metadata from Unstructured elements.

        Args:
            elements: List of Unstructured elements.

        Returns:
            Dictionary of extracted metadata.
        """
        metadata = {}

        # Count pages and element types
        pages = set()
        element_types = {}

        for element in elements:
            # Track page numbers
            if hasattr(element, "metadata") and element.metadata:
                # Handle both dict and ElementMetadata object
                try:
                    if hasattr(element.metadata, "page_number"):
                        pages.add(element.metadata.page_number)
                    elif isinstance(element.metadata, dict) and "page_number" in element.metadata:
                        pages.add(element.metadata["page_number"])
                except (AttributeError, KeyError, TypeError):
                    pass

                # Extract document-level metadata from first element
                if not metadata:  # Only extract once
                    element_meta = element.metadata
                    try:
                        # Try attribute access first (ElementMetadata object)
                        if hasattr(element_meta, "filename"):
                            metadata["source_filename"] = element_meta.filename
                        elif isinstance(element_meta, dict) and "filename" in element_meta:
                            metadata["source_filename"] = element_meta["filename"]

                        if hasattr(element_meta, "file_directory"):
                            metadata["source_directory"] = element_meta.file_directory
                        elif isinstance(element_meta, dict) and "file_directory" in element_meta:
                            metadata["source_directory"] = element_meta["file_directory"]
                    except (AttributeError, KeyError, TypeError):
                        pass

            # Count element types
            element_type = str(type(element).__name__)
            element_types[element_type] = element_types.get(element_type, 0) + 1

        metadata["page_count"] = len(pages) if pages else 1
        metadata["element_types"] = element_types
        metadata["total_elements"] = len(elements)

        return metadata
