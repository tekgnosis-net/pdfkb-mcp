"""PDF parser using MinerU CLI tool."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parser import DocumentParser, PageContent, ParseResult

logger = logging.getLogger(__name__)


class MinerUPDFParser(DocumentParser):
    """PDF parser using MinerU CLI tool."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Optional[Path] = None):
        """Initialize the MinerU parser.

        Args:
            config: Configuration options for MinerU CLI.
            cache_dir: Directory to cache parsed markdown files.
        """
        super().__init__(cache_dir)

        # Default configuration with basic settings
        self.default_config = {
            "backend": "pipeline",
            "method": "auto",
            "lang": "en",
            "formula": True,
            "table": True,
            "vram": 16,
        }

        # Merge user config with defaults
        self.config = {**self.default_config, **(config or {})}

        # Check if mineru is available
        self._check_mineru_availability()

    def _check_mineru_availability(self) -> None:
        """Check if mineru CLI tool is available."""
        try:
            result = subprocess.run(["mineru", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.debug(f"MinerU CLI available: {result.stdout.strip()}")
            else:
                raise RuntimeError("MinerU CLI returned non-zero exit code")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            raise ImportError(
                "MinerU CLI not available. Please ensure MinerU is installed and accessible via 'mineru' command."
            ) from e

    async def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF file using MinerU CLI.

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
                    if markdown_content is not None:
                        # Create page-aware result
                        # TODO: Implement proper page extraction for this parser
                        pages = [PageContent(page_number=1, markdown_content=markdown_content, metadata={})]
                        return ParseResult(pages=pages, metadata=metadata)

            logger.debug(f"Parsing PDF with MinerU CLI: {file_path}")

            # Create temporary directory for MinerU output
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output_dir = Path(temp_dir) / "mineru_output"
                temp_output_dir.mkdir()

                # Execute MinerU CLI
                markdown_content, metadata = self._execute_mineru(file_path, temp_output_dir)

                # Add processing information
                metadata["processing_timestamp"] = "N/A"  # Will be set by PDFProcessor
                metadata["processor_version"] = "mineru"
                metadata["source_filename"] = file_path.name
                metadata["source_directory"] = str(file_path.parent)

                # Save to cache if enabled
                if cache_path:
                    logger.debug(f"Saving parsed content to cache: {cache_path}")
                    self._save_to_cache(cache_path, markdown_content)
                    self._save_metadata_to_cache(cache_path, metadata)

                logger.debug("Successfully extracted content from PDF using MinerU")
                # Create page-aware result
                # TODO: Implement proper page extraction for this parser
                pages = [PageContent(page_number=1, markdown_content=markdown_content, metadata={})]
                return ParseResult(pages=pages, metadata=metadata)

        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with MinerU: {e}") from e

    def _execute_mineru(self, file_path: Path, output_dir: Path) -> tuple[str, Dict[str, Any]]:
        """Execute MinerU CLI and process results.

        Args:
            file_path: Path to the PDF file.
            output_dir: Directory for MinerU output.

        Returns:
            Tuple of (markdown_content, metadata).
        """
        # Build MinerU command
        cmd = self._build_mineru_command(file_path, output_dir)

        logger.info(f"Executing MinerU command: {' '.join(cmd)}")

        # Execute command without timeout
        try:
            # Use Popen for real-time output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                cwd=Path.cwd(),
                bufsize=1,
                universal_newlines=True,
            )

            # Capture and log output in real-time
            output_lines = []

            # Read output line by line for real-time logging
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        output_lines.append(line)
                        logger.info(f"MinerU: {line.rstrip()}")

            # Wait for process to complete and get return code
            return_code = process.wait()

            # Check return code
            if return_code != 0:
                error_msg = f"MinerU CLI failed with exit code {return_code}"
                if output_lines:
                    error_msg += f": {''.join(output_lines[-10:])}"  # Last 10 lines of output
                raise RuntimeError(error_msg)

            logger.info("MinerU CLI execution completed successfully")

        except Exception as e:
            raise RuntimeError(f"MinerU CLI execution failed: {e}")

        # Process output files
        return self._process_mineru_output(file_path, output_dir)

    def _build_mineru_command(self, file_path: Path, output_dir: Path) -> List[str]:
        """Build MinerU CLI command with configuration.

        Args:
            file_path: Path to the PDF file.
            output_dir: Directory for MinerU output.

        Returns:
            List of command arguments.
        """
        cmd = ["mineru", "-p", str(file_path), "-o", str(output_dir)]

        # Add configuration options
        if "backend" in self.config:
            cmd.extend(["-b", self.config["backend"]])

        if "method" in self.config:
            cmd.extend(["-m", self.config["method"]])

        if "lang" in self.config:
            cmd.extend(["-l", self.config["lang"]])

        if "formula" in self.config:
            cmd.extend(["-f", str(self.config["formula"]).lower()])

        if "table" in self.config:
            cmd.extend(["-t", str(self.config["table"]).lower()])

        # Add other optional parameters
        for param in ["start", "end", "device", "vram", "url", "source"]:
            if param in self.config:
                flag = f"-{param[0]}" if param in ["start", "end", "device"] else f"--{param}"
                cmd.extend([flag, str(self.config[param])])

        return cmd

    def _process_mineru_output(self, file_path: Path, output_dir: Path) -> tuple[str, Dict[str, Any]]:
        """Process MinerU output files and extract content and metadata.

        Args:
            file_path: Original PDF file path.
            output_dir: MinerU output directory.

        Returns:
            Tuple of (markdown_content, metadata).
        """
        # Find markdown output file
        markdown_content = self._find_and_read_markdown(file_path, output_dir)

        # Extract metadata from JSON files
        metadata = self._extract_metadata_from_json_files(file_path, output_dir)

        return markdown_content, metadata

    def _find_and_read_markdown(self, file_path: Path, output_dir: Path) -> str:
        """Find and read the markdown output file from MinerU.

        Args:
            file_path: Original PDF file path.
            output_dir: MinerU output directory.

        Returns:
            Markdown content as string.
        """
        # MinerU typically creates output with the same name as input but .md extension
        expected_md_name = file_path.stem + ".md"

        # Look for markdown files in output directory (recursively to handle MinerU's nested structure)
        markdown_files = list(output_dir.glob("**/*.md"))

        if not markdown_files:
            raise RuntimeError(f"No markdown output found in {output_dir}")

        # Prefer file with matching name, otherwise use first found
        target_file = None
        for md_file in markdown_files:
            if md_file.name == expected_md_name:
                target_file = md_file
                break

        if not target_file:
            target_file = markdown_files[0]
            logger.warning(f"Expected {expected_md_name} but using {target_file.name}")

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                raise RuntimeError("Markdown output file is empty")

            logger.debug(f"Read {len(content)} characters from {target_file.name}")
            return content

        except Exception as e:
            raise RuntimeError(f"Failed to read markdown output {target_file}: {e}") from e

    def _extract_metadata_from_json_files(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extract metadata from MinerU JSON output files.

        Args:
            file_path: Original PDF file path.
            output_dir: MinerU output directory.

        Returns:
            Dictionary of extracted metadata.
        """
        metadata = {}

        # Look for middle.json and model.json files
        json_files = list(output_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                if "middle" in json_file.name:
                    # Extract metadata from middle.json
                    middle_metadata = self._extract_from_middle_json(json_data)
                    metadata.update(middle_metadata)

                elif "model" in json_file.name:
                    # Extract metadata from model.json
                    model_metadata = self._extract_from_model_json(json_data)
                    metadata.update(model_metadata)

                logger.debug(f"Extracted metadata from {json_file.name}")

            except Exception as e:
                logger.warning(f"Failed to extract metadata from {json_file}: {e}")

        # Add basic fallback metadata if no JSON files were processed
        if not metadata:
            metadata = {
                "page_count": 1,  # Default fallback
                "element_types": {},
                "total_elements": 0,
            }

        return metadata

    def _extract_from_middle_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from middle.json file.

        Args:
            json_data: Parsed JSON data from middle.json.

        Returns:
            Dictionary of extracted metadata.
        """
        metadata = {}

        # Extract parse type and version
        if "_parse_type" in json_data:
            metadata["parse_type"] = json_data["_parse_type"]

        if "_version_name" in json_data:
            metadata["mineru_version"] = json_data["_version_name"]

        # Extract page information
        pdf_info = json_data.get("pdf_info", [])
        if pdf_info:
            metadata["page_count"] = len(pdf_info)

            # Count elements and types
            total_elements = 0
            element_types = {}

            for page in pdf_info:
                # Count para_blocks
                para_blocks = page.get("para_blocks", [])
                total_elements += len(para_blocks)

                # Count block types
                for block in para_blocks:
                    block_type = block.get("type", "unknown")
                    element_types[block_type] = element_types.get(block_type, 0) + 1

                # Count images and tables
                images = page.get("images", [])
                tables = page.get("tables", [])
                equations = page.get("interline_equations", [])

                total_elements += len(images) + len(tables) + len(equations)

                if images:
                    element_types["image"] = element_types.get("image", 0) + len(images)
                if tables:
                    element_types["table"] = element_types.get("table", 0) + len(tables)
                if equations:
                    element_types["equation"] = element_types.get("equation", 0) + len(equations)

            metadata["total_elements"] = total_elements
            metadata["element_types"] = element_types

        return metadata

    def _extract_from_model_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from model.json file.

        Args:
            json_data: Parsed JSON data from model.json.

        Returns:
            Dictionary of extracted metadata.
        """
        metadata = {}

        # Extract inference results information
        if isinstance(json_data, list):
            metadata["inference_pages"] = len(json_data)

            # Count layout detection results
            total_detections = 0
            category_counts = {}

            for page_result in json_data:
                layout_dets = page_result.get("layout_dets", [])
                total_detections += len(layout_dets)

                for detection in layout_dets:
                    category_id = detection.get("category_id")
                    if category_id is not None:
                        category_counts[category_id] = category_counts.get(category_id, 0) + 1

                # Extract page info
                page_info = page_result.get("page_info", {})
                if page_info and "page_count" not in metadata:
                    # Use the highest page number + 1 as page count
                    page_no = page_info.get("page_no", 0)
                    metadata["page_count"] = max(metadata.get("page_count", 0), page_no + 1)

            metadata["total_detections"] = total_detections
            metadata["category_counts"] = category_counts

        return metadata
