"""Integration tests for PageChunker with document processing pipeline."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from pdfkb.config import ServerConfig
from pdfkb.document_processor import DocumentProcessor


class TestPageChunkerIntegration:
    """Integration tests for PageChunker with the document processing pipeline."""

    @pytest.fixture
    def config_with_page_chunker(self):
        """Create config with page chunker."""
        return ServerConfig(
            openai_api_key="test-key",
            document_chunker="page",
            page_chunker_min_chunk_size=50,
            page_chunker_merge_small=True,
            markdown_page_boundary_pattern=r"--\[PAGE:\s*(\d+)\]--",
            markdown_split_on_page_boundaries=True,
        )

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()

        async def generate_embeddings(texts):
            return [[0.1] * 1024 for _ in texts]

        service.generate_embeddings = AsyncMock(side_effect=generate_embeddings)
        return service

    @pytest.mark.asyncio
    async def test_process_markdown_with_page_markers(self, config_with_page_chunker, mock_embedding_service):
        """Test processing markdown with page markers using page chunker."""
        # Create markdown content with page markers
        content = """---
title: Test Document with Pages
author: Test Author
---

--[PAGE: 1]--
# Introduction

This is the introduction on page 1.
It contains important background information.

--[PAGE: 2]--
## Methods

The methods section on page 2.
Describes the approach taken.

--[PAGE: 3]--
## Results

Results are presented on page 3.
Shows the findings of the study.

--[PAGE: 4]--
## Conclusion

The conclusion on page 4.
Summarizes the key points."""

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            # Create processor
            processor = DocumentProcessor(
                config=config_with_page_chunker,
                embedding_service=mock_embedding_service,
            )

            # Process the document
            result = await processor.process_markdown(temp_path)

            # Verify processing succeeded
            assert result.success is True
            assert result.document is not None

            # Check we got 4 chunks (one per page)
            assert len(result.document.chunks) == 4

            # Verify each chunk corresponds to a page
            for i, chunk in enumerate(result.document.chunks):
                page_num = i + 1
                assert f"page {page_num}" in chunk.text.lower() or f"[PAGE: {page_num}]" in chunk.text
                assert chunk.metadata.get("page_number") == page_num or chunk.metadata.get("page_numbers") == [page_num]
                assert chunk.metadata.get("chunk_strategy") == "page"

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_markdown_without_page_markers(self, config_with_page_chunker, mock_embedding_service):
        """Test processing markdown without page markers."""
        content = """# Document Title

This is a document without page markers.

## Section 1

Content for section 1.

## Section 2

Content for section 2."""

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            # Create processor
            processor = DocumentProcessor(
                config=config_with_page_chunker,
                embedding_service=mock_embedding_service,
            )

            # Process the document
            result = await processor.process_markdown(temp_path)

            # Verify processing succeeded
            assert result.success is True

            # Should have 1 chunk (entire document as single page)
            assert len(result.document.chunks) == 1
            assert result.document.chunks[0].metadata.get("chunk_strategy") == "page"
            assert "Document Title" in result.document.chunks[0].text

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_page_merging_with_small_pages(self, mock_embedding_service):
        """Test that small pages are properly merged."""
        config = ServerConfig(
            openai_api_key="test-key",
            document_chunker="page",
            page_chunker_min_chunk_size=100,
            page_chunker_merge_small=True,
            markdown_page_boundary_pattern=r"--\[PAGE:\s*(\d+)\]--",
            markdown_split_on_page_boundaries=True,
        )

        content = """--[PAGE: 1]--
Short content.

--[PAGE: 2]--
Also short.

--[PAGE: 3]--
This is a much longer page with enough content to exceed the minimum chunk size threshold.

--[PAGE: 4]--
Another long page with substantial content that won't be merged with others."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            processor = DocumentProcessor(
                config=config,
                embedding_service=mock_embedding_service,
            )

            result = await processor.process_markdown(temp_path)

            assert result.success is True

            # After removing page markers, pages 1, 2, and 3 are all small enough to be merged
            # Should have 2 chunks: pages 1+2+3 merged, page 4 separate
            assert len(result.document.chunks) == 2

            # First chunk should be merged pages 1, 2, and 3
            assert "Short content" in result.document.chunks[0].text
            assert "Also short" in result.document.chunks[0].text
            assert "much longer page" in result.document.chunks[0].text

            # Check metadata indicates merging
            if "page_numbers" in result.document.chunks[0].metadata:
                assert result.document.chunks[0].metadata["page_numbers"] == [1, 2, 3]

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_pdf_processing_with_page_chunker(self, config_with_page_chunker, mock_embedding_service):
        """Test that page chunker works with PDF processing."""
        # Use the real sample PDF
        sample_pdf_path = Path(__file__).parent / "sample.pdf"
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        # Copy sample PDF to temp location
        shutil.copy(sample_pdf_path, temp_path)

        try:
            # Create processor
            processor = DocumentProcessor(
                config=config_with_page_chunker,
                embedding_service=mock_embedding_service,
            )

            # Mock the parser to return page-aware content
            from pdfkb.parsers.parser import PageContent, ParseResult

            pages = [
                PageContent(page_number=1, markdown_content="Content from PDF page 1", metadata={}),
                PageContent(page_number=2, markdown_content="Content from PDF page 2", metadata={}),
            ]

            mock_parse_result = ParseResult(pages=pages, metadata={"page_count": 2})

            processor.parser.parse = AsyncMock(return_value=mock_parse_result)

            # Process the PDF
            result = await processor.process_pdf(temp_path)

            # Should work with PDF content too
            assert result.success is True

            # With min_chunk_size=50 and merge_small=True, the two small pages
            # ("Content from PDF page 1" and "Content from PDF page 2") should be merged
            assert len(result.document.chunks) == 1

            # Check the merged chunk contains both pages
            chunk = result.document.chunks[0]
            assert "Content from PDF page 1" in chunk.text
            assert "Content from PDF page 2" in chunk.text
            assert chunk.metadata.get("chunk_strategy") == "page"
            assert chunk.metadata.get("page_numbers") == [1, 2]

        finally:
            temp_path.unlink()


class TestPageChunkerConfiguration:
    """Test configuration validation for page chunker."""

    def test_valid_page_chunker_config(self):
        """Test valid page chunker configuration."""
        config = ServerConfig(
            openai_api_key="test-key",
            document_chunker="page",
            page_chunker_min_chunk_size=100,
            page_chunker_max_chunk_size=5000,
            page_chunker_merge_small=True,
        )

        assert config.document_chunker == "page"
        assert config.pdf_chunker == "page"  # Backward compatibility
        assert config.page_chunker_min_chunk_size == 100
        assert config.page_chunker_max_chunk_size == 5000
        assert config.page_chunker_merge_small is True

    def test_page_chunker_from_env(self, monkeypatch):
        """Test loading page chunker config from environment."""
        monkeypatch.setenv("PDFKB_DOCUMENT_CHUNKER", "page")
        monkeypatch.setenv("PDFKB_PAGE_CHUNKER_MIN_CHUNK_SIZE", "200")
        monkeypatch.setenv("PDFKB_PAGE_CHUNKER_MAX_CHUNK_SIZE", "10000")
        monkeypatch.setenv("PDFKB_PAGE_CHUNKER_MERGE_SMALL", "false")

        config = ServerConfig.from_env()

        assert config.document_chunker == "page"
        assert config.page_chunker_min_chunk_size == 200
        assert config.page_chunker_max_chunk_size == 10000
        assert config.page_chunker_merge_small is False

    def test_markdown_page_boundary_config(self, monkeypatch):
        """Test markdown page boundary configuration."""
        monkeypatch.setenv("PDFKB_MARKDOWN_PAGE_BOUNDARY_PATTERN", r"<<<PAGE (\d+)>>>")
        monkeypatch.setenv("PDFKB_MARKDOWN_SPLIT_ON_PAGE_BOUNDARIES", "true")

        config = ServerConfig.from_env()

        assert config.markdown_page_boundary_pattern == r"<<<PAGE (\d+)>>>"
        assert config.markdown_split_on_page_boundaries is True
