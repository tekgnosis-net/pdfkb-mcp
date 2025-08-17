"""PDF parser using the Marker library."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .parser import DocumentParser, PageContent, ParseResult

logger = logging.getLogger(__name__)


class MarkerPDFParser(DocumentParser):
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
                        # Create page-aware result
                        # TODO: Implement proper page extraction for this parser
                        pages = [PageContent(page_number=1, markdown_content=markdown_content, metadata={})]
                        return ParseResult(pages=pages, metadata=metadata)

            from marker.config.parser import ConfigParser
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            logger.debug(f"Partitioning PDF with Marker: {file_path}")

            # Prepare configuration for Marker
            marker_config = {
                "output_format": "markdown",  # Specify output format to avoid KeyError
                "parallel_factor": 1,  # Process pages sequentially to reduce memory usage
                "disable_ocr": False,  # Allow OCR but can be disabled if needed
                "debug": False,  # Disable debug mode for better performance
                # Table recognition will be attempted, but we have a fallback if it fails
            }

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

            # Run the blocking Marker conversion in a thread executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()

            def run_marker_conversion():
                """Run the Marker conversion synchronously."""
                logger.info(f"Starting Marker conversion for {file_path.name}")
                try:
                    # Suppress verbose output from Marker by redirecting tqdm
                    import os
                    from contextlib import redirect_stderr
                    from io import StringIO

                    # Capture output to reduce verbosity
                    with redirect_stderr(StringIO()):
                        # Keep stdout for important messages but reduce tqdm verbosity
                        old_env = os.environ.get("TQDM_DISABLE")
                        try:
                            # Reduce tqdm verbosity
                            os.environ["TQDM_DISABLE"] = "0"  # Keep tqdm but less verbose
                            rendered = converter(str(file_path))
                        finally:
                            if old_env is not None:
                                os.environ["TQDM_DISABLE"] = old_env
                            elif "TQDM_DISABLE" in os.environ:
                                del os.environ["TQDM_DISABLE"]
                    return rendered
                except RuntimeError as e:
                    if "stack expects a non-empty TensorList" in str(e):
                        logger.warning(
                            f"Table recognition failed for {file_path.name}, retrying without table recognition"
                        )
                        # Retry without table recognition
                        marker_config["disable_table_rec"] = True
                        config_parser_retry = ConfigParser(marker_config)
                        converter_retry = PdfConverter(
                            config=config_parser_retry.generate_config_dict(),
                            artifact_dict=create_model_dict(),
                            processor_list=config_parser_retry.get_processors(),
                            renderer=config_parser_retry.get_renderer(),
                            llm_service=(
                                config_parser_retry.get_llm_service() if marker_config.get("use_llm") else None
                            ),
                        )
                        with redirect_stderr(StringIO()):
                            old_env = os.environ.get("TQDM_DISABLE")
                            try:
                                os.environ["TQDM_DISABLE"] = "0"
                                rendered = converter_retry(str(file_path))
                            finally:
                                if old_env is not None:
                                    os.environ["TQDM_DISABLE"] = old_env
                                elif "TQDM_DISABLE" in os.environ:
                                    del os.environ["TQDM_DISABLE"]
                        return rendered
                    else:
                        logger.error(f"Marker conversion failed: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Marker conversion failed: {e}")
                    raise

            # Run with a timeout to prevent hanging
            try:
                rendered = await asyncio.wait_for(
                    loop.run_in_executor(None, run_marker_conversion), timeout=300  # 5 minute timeout
                )
                logger.info(f"Marker conversion completed for {file_path.name}")
            except asyncio.TimeoutError:
                raise RuntimeError(f"Marker conversion timed out after 5 minutes for {file_path.name}")

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
            # Create page-aware result
            # TODO: Implement proper page extraction for this parser
            pages = [PageContent(page_number=1, markdown_content=markdown_content, metadata={})]
            return ParseResult(pages=pages, metadata=metadata)

        except ImportError:
            raise ImportError("Marker library not available. Install with: pip install pdfkb-mcp[marker]")
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
