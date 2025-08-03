"""PDF parser using the Marker library."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .parser import ParseResult, PDFParser

logger = logging.getLogger(__name__)


class MarkerPDFParser(PDFParser):
    """PDF parser using the Marker library."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Path = None):
        """Initialize the Marker parser.

        Args:
            config: Configuration options for Marker parser, including LLM settings.
            cache_dir: Directory to cache parsed markdown files.
        """
        super().__init__(cache_dir)
        self.config = config or {}

    async def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF file using Marker library.

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

            from marker.config.parser import ConfigParser
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            logger.debug(f"Partitioning PDF with Marker: {file_path}")

            # Prepare configuration for Marker
            marker_config = {}

            # Configure LLM if enabled
            if self.config.get("use_llm", False):
                marker_config["use_llm"] = True

                # Configure OpenAI service to use OpenRouter
                if self.config.get("openrouter_api_key"):
                    marker_config["openai_api_key"] = self.config["openrouter_api_key"]
                    marker_config["openai_base_url"] = "https://openrouter.ai/api/v1"
                    marker_config["openai_model"] = self.config.get("llm_model", "gpt-4o")
                    marker_config["llm_service"] = "marker.services.openai.OpenAIService"
                else:
                    logger.warning(
                        "LLM enabled for Marker but no OpenRouter API key provided. Falling back to non-LLM processing."
                    )
                    marker_config["use_llm"] = False

            # Create ConfigParser with our configuration
            config_parser = ConfigParser(marker_config)

            # Create converter with LLM configuration
            converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=(config_parser.get_llm_service() if marker_config.get("use_llm") else None),
            )
            rendered = converter(str(file_path))
            markdown_content, _, _ = text_from_rendered(rendered)

            # Extract metadata from rendered output
            metadata = self._extract_metadata(file_path, rendered)

            # Add processing information
            metadata["processing_timestamp"] = "N/A"  # Will be set by PDFProcessor
            metadata["processor_version"] = "marker"
            metadata["llm_enabled"] = marker_config.get("use_llm", False)
            if marker_config.get("use_llm", False):
                metadata["llm_model"] = marker_config.get("openai_model", "unknown")
            metadata["source_filename"] = file_path.name
            metadata["source_directory"] = str(file_path.parent)

            # Save to cache if enabled
            if cache_path:
                logger.debug(f"Saving parsed content to cache: {cache_path}")
                self._save_to_cache(cache_path, markdown_content)
                self._save_metadata_to_cache(cache_path, metadata)

            logger.debug("Extracted markdown content from PDF using Marker")
            return ParseResult(markdown_content=markdown_content, metadata=metadata)

        except ImportError:
            raise ImportError("Marker library not available. Install with: pip install marker-pdf")
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with Marker: {e}") from e

    def _extract_metadata(self, file_path: Path, rendered) -> Dict[str, Any]:
        """Extract metadata from Marker's rendered output.

        Args:
            file_path: Path to the PDF file.
            rendered: Rendered output from Marker.

        Returns:
            Dictionary of extracted metadata.
        """
        metadata = {}

        # Extract page count and other metadata from rendered output
        try:
            # Try to get page count from metadata if available
            if hasattr(rendered, "metadata"):
                rendered_metadata = rendered.metadata
                if hasattr(rendered_metadata, "page_stats") and rendered_metadata.page_stats:
                    metadata["page_count"] = len(rendered_metadata.page_stats)
                elif hasattr(rendered_metadata, "table_of_contents") and rendered_metadata.table_of_contents:
                    # Estimate page count from table of contents
                    page_ids = set()
                    for toc_entry in rendered_metadata.table_of_contents:
                        page_ids.add(toc_entry.page_id)
                    metadata["page_count"] = len(page_ids) if page_ids else 1
                else:
                    # Default to 1 if no specific page info
                    metadata["page_count"] = 1

                # Add all metadata from rendered output
                if hasattr(rendered_metadata, "__dict__"):
                    metadata.update(rendered_metadata.__dict__)
                elif isinstance(rendered_metadata, dict):
                    metadata.update(rendered_metadata)
            else:
                # Fallback if no metadata object
                metadata["page_count"] = 1
        except Exception as e:
            logger.warning(f"Failed to extract detailed metadata from Marker output: {e}")
            metadata["page_count"] = 1

        return metadata
