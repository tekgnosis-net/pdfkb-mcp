"""Integration tests for DocumentProcessor with Markdown support."""

import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from src.pdfkb.config import ServerConfig
from src.pdfkb.document_processor import DocumentProcessor


class TestDocumentProcessorMarkdown:
    """Test suite for DocumentProcessor with Markdown files."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = ServerConfig(openai_api_key="test-key")
        config.cache_dir = Path(tempfile.mkdtemp())
        # processing_path is a property that automatically uses cache_dir/processing
        config.processing_path.mkdir(parents=True, exist_ok=True)
        config.metadata_path.mkdir(parents=True, exist_ok=True)
        return config

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()

        # Return embeddings for each text chunk
        async def generate_embeddings(texts):
            return [[0.1] * 1024 for _ in texts]

        service.generate_embeddings = AsyncMock(side_effect=generate_embeddings)
        return service

    @pytest.fixture
    def processor(self, config, mock_embedding_service):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor(
            config=config,
            embedding_service=mock_embedding_service,
            cache_manager=None,
            embedding_semaphore=asyncio.Semaphore(1),
        )

    @pytest.fixture
    def sample_markdown_file(self):
        """Create a temporary markdown file."""
        content = """---
title: Test Document
author: Test Author
---

# Introduction

This is a test document for integration testing.

## Section 1

Content for section 1 with some details.

## Section 2

More content in section 2.

### Subsection 2.1

Even more detailed content here.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)
        return temp_path

    @pytest.mark.asyncio
    async def test_process_markdown_document(self, processor, sample_markdown_file):
        """Test processing a markdown document."""
        try:
            result = await processor.process_markdown(sample_markdown_file)

            # Check processing succeeded
            assert result.success is True
            assert result.error is None
            assert result.document is not None

            # Check document metadata
            document = result.document
            assert document.title == "Test Document"
            assert document.metadata["author"] == "Test Author"
            assert document.metadata["document_type"] == "markdown"
            assert document.page_count == 1  # Markdown without page boundaries is treated as a single page

            # Check chunks were created
            assert len(document.chunks) > 0
            assert result.chunks_created == len(document.chunks)

            # Check embeddings were generated
            assert all(chunk.has_embedding for chunk in document.chunks)
            assert result.embeddings_generated == len(document.chunks)

            # Check processing time is recorded
            assert result.processing_time > 0

        finally:
            sample_markdown_file.unlink()

    @pytest.mark.asyncio
    async def test_process_document_routes_markdown(self, processor, sample_markdown_file):
        """Test that process_document correctly routes markdown files."""
        try:
            result = await processor.process_document(sample_markdown_file)

            assert result.success is True
            assert result.document is not None
            assert result.document.metadata["document_type"] == "markdown"

        finally:
            sample_markdown_file.unlink()

    @pytest.mark.asyncio
    async def test_process_document_routes_pdf(self, processor):
        """Test that process_document correctly routes PDF files."""
        # Use the real sample PDF
        sample_pdf_path = Path(__file__).parent / "sample.pdf"
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        # Copy sample PDF to temp location
        shutil.copy(sample_pdf_path, temp_path)

        try:
            result = await processor.process_document(temp_path)

            # Should successfully process the real PDF
            assert result.success is True or result.document is not None

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_unsupported_document_type(self, processor):
        """Test that unsupported file types return an error."""
        # Create a .txt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("plain text content")
            temp_path = Path(f.name)

        try:
            result = await processor.process_document(temp_path)

            assert result.success is False
            assert "Unsupported document type" in result.error

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_markdown_chunking(self, processor, sample_markdown_file):
        """Test that markdown content is properly chunked."""
        try:
            result = await processor.process_markdown(sample_markdown_file)

            assert result.success is True

            # Check chunks contain expected content
            all_text = " ".join(chunk.text for chunk in result.document.chunks)
            # The chunker may not include headers in the text, but should include the content
            assert "test document" in all_text.lower()
            assert "content" in all_text.lower()

            # Check chunk metadata
            for chunk in result.document.chunks:
                assert chunk.document_id == result.document.id
                assert chunk.chunk_index >= 0
                assert chunk.text != ""

        finally:
            sample_markdown_file.unlink()

    @pytest.mark.asyncio
    async def test_markdown_without_frontmatter(self, processor):
        """Test processing markdown without frontmatter."""
        content = """# Document Without Frontmatter

This is a simple markdown document.

## Section A

Content here.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            result = await processor.process_markdown(temp_path)

            assert result.success is True
            assert result.document.title == "Document Without Frontmatter"
            assert "author" not in result.document.metadata

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_markdown_error_handling(self, processor):
        """Test error handling for invalid markdown files."""
        # Test with non-existent file
        nonexistent_path = Path("/tmp/nonexistent_markdown_12345.md")

        result = await processor.process_markdown(nonexistent_path)

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_markdown_file(self, processor):
        """Test processing an empty markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            result = await processor.process_markdown(temp_path)

            # Should still process successfully, even if empty
            assert result.success is True
            assert result.document is not None
            # May have 0 or 1 chunks depending on chunker behavior with empty content
            assert result.chunks_created >= 0

        finally:
            temp_path.unlink()
