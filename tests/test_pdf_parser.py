"""Tests for the PDF parser module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.exceptions import PDFProcessingError
from pdfkb.parsers import ParseResult, PyMuPDF4LLMParser, UnstructuredPDFParser
from pdfkb.pdf_processor import PDFProcessor


class TestPDFParser:
    """Test cases for PDFParser classes."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            chunk_size=1000,
            chunk_overlap=200,
        )

    @pytest.fixture
    def embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        service.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        return service

    @pytest.mark.asyncio
    async def test_unstructured_parser_creation(self):
        """Test UnstructuredPDFParser creation."""
        parser = UnstructuredPDFParser(strategy="fast")
        assert parser.strategy == "fast"

    @pytest.mark.asyncio
    async def test_pymupdf4llm_parser_creation(self):
        """Test PyMuPDF4LLMParser creation."""
        parser = PyMuPDF4LLMParser()
        assert parser.config == {}

    @pytest.mark.asyncio
    async def test_unstructured_parser_parse(self, tmp_path):
        """Test UnstructuredPDFParser parse method."""
        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Mock the unstructured partition function at the correct location
        with patch("unstructured.partition.pdf.partition_pdf") as mock_partition:
            mock_partition.return_value = ["test element 1", "test element 2"]

            parser = UnstructuredPDFParser(strategy="fast")
            result = await parser.parse(pdf_file)

            assert isinstance(result, ParseResult)
            assert len(result.markdown_content) > 0
            assert "processor_version" in result.metadata
            assert result.metadata["processor_version"] == "unstructured"

    @pytest.mark.asyncio
    async def test_pymupdf4llm_parser_parse(self, tmp_path):
        """Test PyMuPDF4LLMParser parse method."""
        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Mock the pymupdf4llm to_markdown function at the correct location
        with patch("pymupdf4llm.to_markdown") as mock_to_markdown:
            mock_to_markdown.return_value = "# Test Title\n\nThis is test content."

            parser = PyMuPDF4LLMParser()
            result = await parser.parse(pdf_file)

            assert isinstance(result, ParseResult)
            assert len(result.markdown_content) > 0
            assert "processor_version" in result.metadata
            assert result.metadata["processor_version"] == "pymupdf4llm"

    @pytest.mark.asyncio
    async def test_pymupdf4llm_parser_parse_with_pages(self, tmp_path):
        """Test PyMuPDF4LLMParser parse method with page chunks."""
        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Mock the pymupdf4llm to_markdown function at the correct location
        with patch("pymupdf4llm.to_markdown") as mock_to_markdown:
            mock_to_markdown.return_value = [
                {"text": "# Page 1 Title\n\nPage 1 content."},
                {"text": "# Page 2 Title\n\nPage 2 content."},
            ]

            parser = PyMuPDF4LLMParser()
            result = await parser.parse(pdf_file)

            assert isinstance(result, ParseResult)
            assert len(result.markdown_content) > 0
            assert "processor_version" in result.metadata

    @pytest.mark.asyncio
    async def test_pdf_processor_with_unstructured_parser(self, config, embedding_service, tmp_path):
        """Test PDFProcessor with Unstructured parser."""
        # Create a config with unstructured parser
        config.pdf_parser = "unstructured"

        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Mock the unstructured partition function
        with patch("unstructured.partition.pdf.partition_pdf") as mock_partition:
            mock_partition.return_value = ["test element 1", "test element 2"]

            processor = PDFProcessor(config, embedding_service)
            assert isinstance(processor.parser, UnstructuredPDFParser)

    @pytest.mark.asyncio
    async def test_pdf_processor_with_pymupdf4llm_parser(self, config, embedding_service, tmp_path):
        """Test PDFProcessor with PyMuPDF4LLM parser."""
        # Create a config with pymupdf4llm parser
        config.pdf_parser = "pymupdf4llm"

        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Mock the pymupdf4llm to_markdown function
        with patch("pymupdf4llm.to_markdown") as mock_to_markdown:
            mock_to_markdown.return_value = "# Test Title\n\nThis is test content."

            processor = PDFProcessor(config, embedding_service)
            assert isinstance(processor.parser, PyMuPDF4LLMParser)

    @pytest.mark.asyncio
    async def test_pdf_processor_parser_fallback(self, config, embedding_service, tmp_path):
        """Test PDFProcessor parser fallback when primary parser is not available."""
        # Create a config with pymupdf4llm parser
        config.pdf_parser = "pymupdf4llm"

        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Mock the PyMuPDF4LLMParser to raise ImportError during construction
        with patch("pdfkb.parsers.parser_pymupdf4llm.PyMuPDF4LLMParser.__init__") as mock_pymupdf_init:
            mock_pymupdf_init.side_effect = ImportError("pymupdf4llm not available")

            # Mock the unstructured partition function
            with patch("unstructured.partition.pdf.partition_pdf") as mock_partition:
                mock_partition.return_value = ["test element 1", "test element 2"]

                processor = PDFProcessor(config, embedding_service)
                # Should fall back to Unstructured parser
                assert isinstance(processor.parser, UnstructuredPDFParser)

    @pytest.mark.asyncio
    async def test_pdf_processor_both_parsers_unavailable(self, config, embedding_service):
        """Test PDFProcessor when both parsers are unavailable."""
        # Create a config with pymupdf4llm parser
        config.pdf_parser = "pymupdf4llm"

        # Mock both parsers to be unavailable
        with patch("pdfkb.parsers.parser_pymupdf4llm.PyMuPDF4LLMParser.__init__") as mock_pymupdf_init:
            mock_pymupdf_init.side_effect = ImportError("pymupdf4llm not available")

            with patch("pdfkb.parsers.parser_unstructured.UnstructuredPDFParser.__init__") as mock_unstructured_init:
                mock_unstructured_init.side_effect = ImportError("unstructured not available")

                with pytest.raises(PDFProcessingError):
                    PDFProcessor(config, embedding_service)
