"""Tests for the PDF processor module."""

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from pdfkb.config import ServerConfig
from pdfkb.document_processor import DocumentProcessor as PDFProcessor


class TestPDFProcessor:
    """Test cases for PDFProcessor class."""

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

    @pytest.fixture
    def processor(self, config, embedding_service):
        """Create a PDFProcessor instance."""
        return PDFProcessor(config, embedding_service)

    @pytest.mark.asyncio
    async def test_process_pdf_file_not_found(self, processor):
        """Test processing a non-existent PDF file."""
        non_existent_file = Path("non_existent.pdf")
        result = await processor.process_pdf(non_existent_file)

        assert not result.success
        assert "File not found" in result.error

    @pytest.mark.asyncio
    async def test_validate_pdf_valid_file(self, processor, sample_pdf):
        """Test validating a valid PDF file."""
        is_valid = await processor.validate_pdf(sample_pdf)
        assert is_valid

    @pytest.mark.asyncio
    async def test_validate_pdf_invalid_extension(self, processor, tmp_path):
        """Test validating a file with wrong extension."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")

        is_valid = await processor.validate_pdf(txt_file)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_validate_pdf_empty_file(self, processor, tmp_path):
        """Test validating an empty file."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")

        is_valid = await processor.validate_pdf(empty_file)
        assert not is_valid

    # TODO: Add more comprehensive tests when real implementation is added
    # - Test actual PDF text extraction
    # - Test chunking strategies
    # - Test embedding generation
    # - Test error handling scenarios
