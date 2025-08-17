"""Tests for minimum chunk size filtering functionality."""

import os
from unittest.mock import patch

import pytest

from pdfkb.chunker.chunker_langchain import LangChainChunker
from pdfkb.chunker.chunker_page import PageChunker
from pdfkb.chunker.chunker_unstructured import ChunkerUnstructured
from pdfkb.config import ServerConfig
from pdfkb.models import Chunk
from pdfkb.parsers.parser import PageContent


class TestMinChunkSizeFiltering:
    """Test cases for minimum chunk size filtering across all chunkers."""

    def test_config_parsing_min_chunk_size(self):
        """Test that PDFKB_MIN_CHUNK_SIZE is parsed correctly."""
        with patch.dict(
            os.environ,
            {"PDFKB_MIN_CHUNK_SIZE": "150", "PDFKB_OPENAI_API_KEY": "sk-test-key", "PDFKB_EMBEDDING_PROVIDER": "local"},
        ):
            config = ServerConfig.from_env()
            assert config.min_chunk_size == 150

    def test_config_default_min_chunk_size(self):
        """Test that min_chunk_size defaults to 0 (disabled)."""
        # Disable dotenv loading to test defaults
        with patch("pdfkb.config.load_dotenv"):
            with patch.dict(
                os.environ, {"PDFKB_OPENAI_API_KEY": "sk-test-key", "PDFKB_EMBEDDING_PROVIDER": "local"}, clear=True
            ):
                config = ServerConfig.from_env()
                assert config.min_chunk_size == 0

    def test_page_chunker_filtering(self):
        """Test that PageChunker filters out small chunks."""
        chunker = PageChunker(
            min_chunk_size=100,  # Page-level merging threshold
            global_min_chunk_size=150,  # Global filtering threshold
        )

        # Create test pages with different sizes
        pages = [
            PageContent(page_number=1, markdown_content="A" * 50, metadata={}),  # Too small (50 chars)
            PageContent(
                page_number=2, markdown_content="B" * 120, metadata={}
            ),  # Small but above page threshold (120 chars)
            PageContent(page_number=3, markdown_content="C" * 200, metadata={}),  # Good size (200 chars)
            PageContent(page_number=4, markdown_content="D" * 30, metadata={}),  # Too small (30 chars)
        ]

        chunks = chunker.chunk_pages(pages, {})

        # Should filter out chunks smaller than 150 characters
        assert len(chunks) == 1  # Only the 200-char chunk should remain
        assert "C" in chunks[0].text
        assert len(chunks[0].text.strip()) >= 150

    def test_page_chunker_no_filtering_when_disabled(self):
        """Test that PageChunker doesn't filter when min_chunk_size is 0."""
        chunker = PageChunker(
            min_chunk_size=None,
            global_min_chunk_size=0,  # Disabled
        )

        pages = [
            PageContent(page_number=1, markdown_content="Small", metadata={}),
            PageContent(page_number=2, markdown_content="Also small", metadata={}),
        ]

        chunks = chunker.chunk_pages(pages, {})

        # Should keep all chunks when filtering is disabled
        assert len(chunks) == 2

    def test_langchain_chunker_filtering(self):
        """Test that LangChainChunker filters out small chunks."""
        chunker = LangChainChunker(
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=150,
        )

        # Create markdown with sections that will produce small chunks
        markdown = (
            """
# Small Section
Short content.

# Large Section
"""
            + "This is a much larger section with plenty of content. " * 10
        )

        chunks = chunker.chunk(markdown, {})

        # Should filter out chunks smaller than 150 characters
        for chunk in chunks:
            assert len(chunk.text.strip()) >= 150

    def test_unstructured_chunker_filtering(self):
        """Test that ChunkerUnstructured filters out small chunks."""
        try:
            chunker = ChunkerUnstructured(min_chunk_size=150)

            # Create markdown that will produce small chunks
            markdown = """
Short.

This is a much longer paragraph with significantly more content that should be above the 150 character threshold.

Tiny.
"""

            chunks = chunker.chunk(markdown, {})

            # Should filter out chunks smaller than 150 characters
            for chunk in chunks:
                assert len(chunk.text.strip()) >= 150

        except ImportError:
            pytest.skip("Unstructured library not available")

    def test_base_chunker_filter_method(self):
        """Test the base chunker's _filter_small_chunks method."""
        from pdfkb.chunker.chunker import Chunker

        # Create a concrete chunker for testing
        class TestChunker(Chunker):
            def chunk(self, markdown_content, metadata):
                return []

        chunker = TestChunker(min_chunk_size=100)

        # Create test chunks with different sizes
        chunks = [
            Chunk(text="A" * 50, chunk_index=0, metadata={}),  # Too small
            Chunk(text="B" * 150, chunk_index=1, metadata={}),  # Good
            Chunk(text="C" * 30, chunk_index=2, metadata={}),  # Too small
            Chunk(text="D" * 200, chunk_index=3, metadata={}),  # Good
        ]

        filtered = chunker._filter_small_chunks(chunks)

        assert len(filtered) == 2
        assert "B" in filtered[0].text
        assert "D" in filtered[1].text

    def test_filtering_disabled_when_zero(self):
        """Test that filtering is disabled when min_chunk_size is 0."""
        from pdfkb.chunker.chunker import Chunker

        class TestChunker(Chunker):
            def chunk(self, markdown_content, metadata):
                return []

        chunker = TestChunker(min_chunk_size=0)  # Disabled

        chunks = [
            Chunk(text="Tiny", chunk_index=0, metadata={}),
            Chunk(text="Also tiny", chunk_index=1, metadata={}),
        ]

        filtered = chunker._filter_small_chunks(chunks)

        # Should keep all chunks when filtering is disabled
        assert len(filtered) == 2

    def test_page_chunker_merging_vs_filtering(self):
        """Test the interaction between page merging and global filtering."""
        chunker = PageChunker(
            min_chunk_size=80,  # Page-level merge threshold
            global_min_chunk_size=150,  # Global filter threshold
            merge_small=True,
        )

        # Create pages that will be merged but still might be too small
        pages = [
            PageContent(page_number=1, markdown_content="A" * 70, metadata={}),  # Will try to merge
            PageContent(page_number=2, markdown_content="B" * 70, metadata={}),  # Will merge with page 1
            PageContent(page_number=3, markdown_content="C" * 50, metadata={}),  # Too small even after potential merge
            PageContent(page_number=4, markdown_content="D" * 200, metadata={}),  # Good size
        ]

        chunks = chunker.chunk_pages(pages, {})

        # Should have merged pages 1+2 (140 chars total, below 150, so filtered)
        # Page 3 is too small (filtered)
        # Page 4 is good size (kept)
        assert len(chunks) == 1
        assert "D" in chunks[0].text

    def test_empty_chunks_handling(self):
        """Test that empty or whitespace-only chunks are handled correctly."""
        chunker = PageChunker(global_min_chunk_size=30)

        pages = [
            PageContent(page_number=1, markdown_content="   \n\t  ", metadata={}),  # Whitespace only
            PageContent(page_number=2, markdown_content="", metadata={}),  # Empty
            PageContent(
                page_number=3,
                markdown_content="Good content here with enough characters to pass the filter",
                metadata={},
            ),
        ]

        chunks = chunker.chunk_pages(pages, {})

        # Should only keep the non-empty chunk with sufficient content
        assert len(chunks) == 1
        assert "Good content" in chunks[0].text
