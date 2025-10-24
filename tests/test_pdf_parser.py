"""Tests for the PDF parser module."""

import shutil
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.document_processor import DocumentProcessor as PDFProcessor
from pdfkb.exceptions import PDFProcessingError
from pdfkb.parsers import ParseResult, PyMuPDF4LLMParser, UnstructuredPDFParser


class TestPDFParser:
    """Test cases for PDFParser classes."""

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Provide a copy of the sample PDF for testing."""
        # Copy the sample PDF to a temp location to avoid modifying the original
        sample_pdf_path = Path(__file__).parent / "sample.pdf"
        test_pdf_path = tmp_path / "test.pdf"
        shutil.copy(sample_pdf_path, test_pdf_path)
        return test_pdf_path

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
    async def test_unstructured_parser_parse(self, sample_pdf, monkeypatch):
        """Test UnstructuredPDFParser parse method."""
        pdf_file = sample_pdf
        # Prepare lightweight stubs for unstructured.partition.pdf so the
        # parser can import and call partition_pdf without pulling heavy
        # optional dependencies into the test environment.
        un_pkg = types.ModuleType("unstructured")
        un_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "unstructured", un_pkg)

        part_pkg = types.ModuleType("unstructured.partition")
        part_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "unstructured.partition", part_pkg)

        pdf_mod = types.ModuleType("unstructured.partition.pdf")
        pdf_mod.partition_pdf = MagicMock(return_value=["test element 1", "test element 2"])
        monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", pdf_mod)

        parser = UnstructuredPDFParser(strategy="fast")
        result = await parser.parse(pdf_file)

        assert isinstance(result, ParseResult)
        assert len(result.pages) > 0
        assert len(result.pages[0].markdown_content) > 0
        assert "processor_version" in result.metadata
        assert result.metadata["processor_version"] == "unstructured"

    @pytest.mark.asyncio
    async def test_pymupdf4llm_parser_parse(self, sample_pdf):
        """Test PyMuPDF4LLMParser parse method."""
        pdf_file = sample_pdf

        # Test with real PDF parsing
        parser = PyMuPDF4LLMParser()
        result = await parser.parse(pdf_file)

        assert isinstance(result, ParseResult)
        assert len(result.pages) > 0
        # Some pages might be empty (e.g., cover pages), so check that at least one page has content
        pages_with_content = [p for p in result.pages if len(p.markdown_content) > 0]
        assert len(pages_with_content) > 0, "At least one page should have content"
        assert "processor_version" in result.metadata
        assert result.metadata["processor_version"] == "pymupdf4llm"

    @pytest.mark.asyncio
    async def test_pymupdf4llm_parser_parse_with_pages(self, sample_pdf):
        """Test PyMuPDF4LLMParser parse method with page chunks."""
        pdf_file = sample_pdf

        # Test with real PDF parsing - sample.pdf should have multiple pages
        parser = PyMuPDF4LLMParser()
        result = await parser.parse(pdf_file)

        assert isinstance(result, ParseResult)
        assert len(result.pages) > 0  # Should have at least one page

        # Check that pages are properly numbered and at least some have content
        pages_with_content = 0
        for page in result.pages:
            assert page.page_number > 0
            if len(page.markdown_content) > 0:
                pages_with_content += 1

        assert pages_with_content > 0, "At least one page should have content"
        assert "processor_version" in result.metadata

    @pytest.mark.asyncio
    async def test_pdf_processor_with_unstructured_parser(self, config, embedding_service, sample_pdf, monkeypatch):
        """Test PDFProcessor with Unstructured parser."""
        # Create a config with unstructured parser
        config.pdf_parser = "unstructured"
        # Inject stub modules so the processor can instantiate the
        # Unstructured parser without installing heavy optional deps.
        un_pkg = types.ModuleType("unstructured")
        un_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "unstructured", un_pkg)

        part_pkg = types.ModuleType("unstructured.partition")
        part_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "unstructured.partition", part_pkg)

        pdf_mod = types.ModuleType("unstructured.partition.pdf")
        pdf_mod.partition_pdf = MagicMock(return_value=["test element 1", "test element 2"])
        monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", pdf_mod)

        processor = PDFProcessor(config, embedding_service)
        assert isinstance(processor.parser, UnstructuredPDFParser)

    @pytest.mark.asyncio
    async def test_pdf_processor_with_pymupdf4llm_parser(self, config, embedding_service, sample_pdf):
        """Test PDFProcessor with PyMuPDF4LLM parser."""
        # Create a config with pymupdf4llm parser
        config.pdf_parser = "pymupdf4llm"

        # Just test that the processor creates the right parser type
        processor = PDFProcessor(config, embedding_service)
        assert isinstance(processor.parser, PyMuPDF4LLMParser)

    @pytest.mark.asyncio
    async def test_pdf_processor_parser_fallback(self, config, embedding_service, sample_pdf, monkeypatch):
        """Test PDFProcessor parser fallback when primary parser is not available."""
        # Create a config with pymupdf4llm parser
        config.pdf_parser = "pymupdf4llm"

        # Mock the PyMuPDF4LLMParser to raise ImportError during construction
        with patch("pdfkb.parsers.parser_pymupdf4llm.PyMuPDF4LLMParser.__init__") as mock_pymupdf_init:
            mock_pymupdf_init.side_effect = ImportError("pymupdf4llm not available")

            # Inject stubs for unstructured.partition.pdf so fallback can be
            # exercised without installing heavy deps.
            un_pkg = types.ModuleType("unstructured")
            un_pkg.__path__ = []
            monkeypatch.setitem(sys.modules, "unstructured", un_pkg)

            part_pkg = types.ModuleType("unstructured.partition")
            part_pkg.__path__ = []
            monkeypatch.setitem(sys.modules, "unstructured.partition", part_pkg)

            pdf_mod = types.ModuleType("unstructured.partition.pdf")
            pdf_mod.partition_pdf = MagicMock(return_value=["test element 1", "test element 2"])
            monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", pdf_mod)

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
