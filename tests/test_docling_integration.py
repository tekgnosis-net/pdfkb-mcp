"""Integration tests for DoclingParser with the existing system."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.pdfkb.config import ServerConfig
from src.pdfkb.parsers.parser_docling import DoclingParser
from src.pdfkb.pdf_processor import PDFProcessor


class TestDoclingIntegration:
    """Test DoclingParser integration with PDFProcessor and ServerConfig."""

    @pytest.fixture
    def temp_pdf_file(self):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n%Test PDF content for integration testing\n")
            temp_path = Path(f.name)

        yield temp_path

        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]] * 5  # Mock embeddings
        return service

    def test_docling_parser_instantiation(self):
        """Test that DoclingParser can be instantiated without errors."""
        parser = DoclingParser()

        assert parser is not None
        # Do not assert non-existent key; ensure defaults exist instead
        assert "ocr_enabled" in parser.config
        # Actually, processor_version is added during parsing, not in config
        assert parser.config["ocr_enabled"] is True
        assert parser.available_features is not None

    def test_config_system_docling_support(self):
        """Test that ServerConfig properly handles docling configuration."""
        # Test that docling is accepted as a valid parser
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-12345",
                "PDFKB_PDF_PARSER": "docling",
                "DOCLING_OCR_ENGINE": "tesseract",
                "DOCLING_TABLE_MODE": "ACCURATE",
                "DOCLING_FORMULA_ENRICHMENT": "true",
            },
            clear=True,
        ):
            config = ServerConfig.from_env()

            assert config.pdf_parser == "docling"
            assert hasattr(config, "docling_config")

            docling_config = getattr(config, "docling_config", {})
            assert docling_config.get("ocr_engine") == "tesseract"
            assert docling_config.get("table_processing_mode") == "ACCURATE"
            assert docling_config.get("formula_enrichment") is True

    @patch("src.pdfkb.parsers.parser_docling.DoclingParser._check_ocr_engine_available")
    def test_pdf_processor_docling_integration(self, mock_ocr_check, mock_embedding_service, temp_pdf_file):
        """Test DoclingParser integration with PDFProcessor."""
        mock_ocr_check.return_value = True

        # Create config for docling parser
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-12345",
                "PDFKB_PDF_PARSER": "docling",
            },
            clear=True,
        ):
            config = ServerConfig.from_env()

        # Create PDFProcessor with docling parser
        processor = PDFProcessor(config, mock_embedding_service)

        # Verify the correct parser was created
        assert isinstance(processor.parser, DoclingParser)
        assert processor.parser.config["ocr_enabled"] is True

    def test_docling_parser_fallback_logic(self, mock_embedding_service):
        """Test that DoclingParser fallback works when docling is not available."""
        # Create config for docling parser
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-12345",
                "PDFKB_PDF_PARSER": "docling",
            },
            clear=True,
        ):
            config = ServerConfig.from_env()

        # Mock docling import failure
        with patch(
            "src.pdfkb.parsers.parser_docling.DoclingParser.__init__",
            side_effect=ImportError("No module named 'docling'"),
        ):

            # PDFProcessor should fallback to another parser
            processor = PDFProcessor(config, mock_embedding_service)

            # Should fallback to UnstructuredPDFParser or PyMuPDF4LLMParser
            assert not isinstance(processor.parser, DoclingParser)
            # The specific fallback parser depends on what's available

    @patch("src.pdfkb.parsers.parser_docling.DoclingParser._check_ocr_engine_available")
    async def test_docling_parser_with_pdf_processor_processing(
        self, mock_ocr_check, mock_embedding_service, temp_pdf_file
    ):
        """Test end-to-end processing with DoclingParser through PDFProcessor."""
        mock_ocr_check.return_value = True

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-12345",
                "PDFKB_PDF_PARSER": "docling",
            },
            clear=True,
        ):
            config = ServerConfig.from_env()

        # Mock docling components for testing
        with (
            patch("docling.document_converter.DocumentConverter") as mock_converter_class,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_options_class,
        ):

            # Setup mock conversion result
            mock_result = Mock()
            mock_result.status = Mock()
            mock_result.status.__eq__ = lambda self, other: False  # Not FAILURE

            mock_doc = Mock()
            mock_doc.pages = [Mock()]
            mock_doc.export_to_markdown.return_value = "# Test Document\n\nIntegration test content."
            mock_result.document = mock_doc

            mock_converter = Mock()
            mock_converter_class.return_value = mock_converter
            mock_converter.convert.return_value = mock_result

            mock_options = Mock()
            mock_options_class.return_value = mock_options

            # Create processor and process PDF
            processor = PDFProcessor(config, mock_embedding_service)

            # Verify we have the right parser
            assert isinstance(processor.parser, DoclingParser)

            # Process the PDF
            result = await processor.process_pdf(temp_pdf_file)

            # Verify processing was successful
            assert result.success is True
            assert result.document is not None
            assert result.document.title is not None
            assert len(result.document.chunks) > 0

            # Verify docling-specific metadata
            metadata = result.document.metadata
            assert metadata["processor_version"] == "docling"
            assert "docling_processing_time" in metadata
            assert "docling_features_used" in metadata


class TestDoclingEnvironmentVariables:
    """Test docling-specific environment variable handling."""

    def test_all_docling_environment_variables(self):
        """Test that all docling environment variables are properly parsed."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-12345",
                "PDFKB_PDF_PARSER": "docling",
                "DOCLING_OCR_ENGINE": "easyocr",
                "DOCLING_OCR_LANGUAGES": "en,es,fr",
                "DOCLING_TABLE_MODE": "FAST",
                "DOCLING_FORMULA_ENRICHMENT": "true",
                "DOCLING_PROCESSING_TIMEOUT": "600",
                "DOCLING_DEVICE": "cuda",
                "DOCLING_MAX_PAGES": "100",
            },
            clear=True,
        ):
            config = ServerConfig.from_env()

            assert config.pdf_parser == "docling"

            docling_config = getattr(config, "docling_config", {})
            assert docling_config["ocr_engine"] == "easyocr"
            assert docling_config["ocr_languages"] == ["en", "es", "fr"]
            assert docling_config["table_processing_mode"] == "FAST"
            assert docling_config["formula_enrichment"] is True
            assert docling_config["processing_timeout"] == 600
            assert docling_config["device_selection"] == "cuda"
            assert docling_config["max_pages"] == 100

    def test_invalid_docling_environment_variables(self):
        """Test handling of invalid docling environment variables."""
        # Test invalid timeout
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-12345",
                "DOCLING_PROCESSING_TIMEOUT": "invalid",
            },
        ):
            with pytest.raises(Exception):  # Should raise ConfigurationError
                ServerConfig.from_env()

        # Test invalid max pages
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-12345",
                "DOCLING_MAX_PAGES": "invalid",
            },
        ):
            with pytest.raises(Exception):  # Should raise ConfigurationError
                ServerConfig.from_env()


class TestDoclingParserRegistry:
    """Test that DoclingParser is properly registered in the parser system."""

    def test_docling_parser_in_imports(self):
        """Test that DoclingParser is available in parser imports."""
        from src.pdfkb.parsers import DoclingParser as ImportedDoclingParser
        from src.pdfkb.parsers.parser_docling import DoclingParser as DirectDoclingParser

        assert ImportedDoclingParser is DirectDoclingParser

    def test_docling_parser_in_all_exports(self):
        """Test that DoclingParser is in __all__ exports."""
        from src.pdfkb.parsers import __all__ as parser_exports

        assert "DoclingParser" in parser_exports


@pytest.mark.performance
class TestDoclingParserPerformance:
    """Performance tests for DoclingParser."""

    @pytest.fixture
    def larger_pdf_content(self):
        """Create larger PDF content for performance testing."""
        # Create a larger mock PDF content
        content = b"%PDF-1.4\n"
        content += b"Mock PDF content " * 1000  # Repeat content to simulate larger file
        return content

    def test_memory_usage_estimation(self):
        """Test that DoclingParser has reasonable memory usage configuration."""
        parser = DoclingParser(
            config={
                "max_file_size": 50 * 1024 * 1024,  # 50MB limit
                "processing_timeout": 120,  # 2 minute timeout
                "table_processing_mode": "FAST",  # Use faster processing
            }
        )

        assert parser.config["max_file_size"] == 50 * 1024 * 1024
        assert parser.config["processing_timeout"] == 120
        assert parser.config["table_processing_mode"] == "FAST"

    def test_timeout_configuration(self):
        """Test timeout configuration for different processing stages."""
        parser = DoclingParser(config={"processing_timeout": 300})

        # Mock PdfPipelineOptions to test timeout application
        mock_options = Mock()
        mock_options.ocr_options = Mock()
        mock_options.table_options = Mock()

        configured_options = parser._apply_resource_limits(mock_options)

        # Verify that the options object was returned (even if timeouts couldn't be set)
        assert configured_options is mock_options
