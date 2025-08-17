"""Tests for the PageChunker implementation."""

from pdfkb.chunker.chunker_page import PageChunker
from pdfkb.parsers.parser import PageContent


class TestPageChunker:
    """Test suite for PageChunker."""

    def test_basic_page_chunking(self):
        """Test basic page chunking with one chunk per page."""
        chunker = PageChunker()

        pages = [
            PageContent(page_number=1, markdown_content="Page 1 content", metadata={}),
            PageContent(page_number=2, markdown_content="Page 2 content", metadata={}),
            PageContent(page_number=3, markdown_content="Page 3 content", metadata={}),
        ]

        chunks = chunker.chunk_pages(pages)

        assert len(chunks) == 3
        assert chunks[0].text == "Page 1 content"
        assert chunks[0].metadata["page_number"] == 1
        assert chunks[1].text == "Page 2 content"
        assert chunks[1].metadata["page_number"] == 2
        assert chunks[2].text == "Page 3 content"
        assert chunks[2].metadata["page_number"] == 3

    def test_merge_small_pages(self):
        """Test merging of small pages."""
        chunker = PageChunker(min_chunk_size=50, merge_small=True)

        pages = [
            PageContent(page_number=1, markdown_content="Small", metadata={}),
            PageContent(page_number=2, markdown_content="Also small", metadata={}),
            PageContent(
                page_number=3,
                markdown_content="This is a much longer page with lots of content that exceeds the minimum",
                metadata={},
            ),
        ]

        chunks = chunker.chunk_pages(pages)

        # First two pages should be merged
        assert len(chunks) == 2
        assert "Small" in chunks[0].text
        assert "Also small" in chunks[0].text
        assert chunks[0].metadata["page_numbers"] == [1, 2]
        assert chunks[0].metadata["merged_pages"] is True

        # Third page should be separate
        assert chunks[1].metadata["page_number"] == 3
        assert chunks[1].metadata["single_page"] is True

    def test_split_large_pages(self):
        """Test splitting of large pages."""
        chunker = PageChunker(max_chunk_size=50)

        long_content = "This is a very long page. " * 10  # Creates content > 50 chars

        pages = [
            PageContent(page_number=1, markdown_content=long_content, metadata={}),
        ]

        chunks = chunker.chunk_pages(pages)

        # Should be split into multiple chunks
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 50
            if "split_from_large_page" in chunk.metadata:
                assert chunk.metadata["split_from_large_page"] is True

    def test_no_merge_when_disabled(self):
        """Test that pages are not merged when merge_small is False."""
        chunker = PageChunker(min_chunk_size=50, merge_small=False)

        pages = [
            PageContent(page_number=1, markdown_content="Small", metadata={}),
            PageContent(page_number=2, markdown_content="Also small", metadata={}),
        ]

        chunks = chunker.chunk_pages(pages)

        # Should not merge even though they're small
        assert len(chunks) == 2
        assert chunks[0].text == "Small"
        assert chunks[1].text == "Also small"

    def test_empty_pages_list(self):
        """Test handling of empty pages list."""
        chunker = PageChunker()

        chunks = chunker.chunk_pages([])

        assert chunks == []

    def test_fallback_chunk_method(self):
        """Test the fallback chunk() method for non-page-aware content."""
        chunker = PageChunker()

        text = "This is some text content"
        chunks = chunker.chunk(text, {"title": "Test"})

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].metadata["title"] == "Test"
        assert chunks[0].metadata["warning"] == "PageChunker used without page-aware content"

    def test_metadata_preservation(self):
        """Test that document-level metadata is preserved in chunks."""
        chunker = PageChunker()

        pages = [
            PageContent(page_number=1, markdown_content="Page 1", metadata={"source": "pdf"}),
        ]

        doc_metadata = {"title": "Test Document", "author": "Test Author"}
        chunks = chunker.chunk_pages(pages, doc_metadata)

        assert len(chunks) == 1
        assert chunks[0].metadata["title"] == "Test Document"
        assert chunks[0].metadata["author"] == "Test Author"
        assert chunks[0].metadata["chunk_strategy"] == "page"

    def test_chunk_indexing(self):
        """Test that chunks are properly indexed."""
        chunker = PageChunker()

        pages = [
            PageContent(page_number=1, markdown_content="Page 1", metadata={}),
            PageContent(page_number=2, markdown_content="Page 2", metadata={}),
            PageContent(page_number=3, markdown_content="Page 3", metadata={}),
        ]

        chunks = chunker.chunk_pages(pages)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == 3

    def test_page_range_metadata(self):
        """Test page range metadata for merged chunks."""
        chunker = PageChunker(min_chunk_size=50, merge_small=True)

        pages = [
            PageContent(page_number=1, markdown_content="A", metadata={}),
            PageContent(page_number=2, markdown_content="B", metadata={}),
            PageContent(page_number=3, markdown_content="C", metadata={}),
        ]

        chunks = chunker.chunk_pages(pages)

        # All pages should be merged into one chunk
        assert len(chunks) == 1
        assert chunks[0].metadata["page_range"] == "1-3"
        assert chunks[0].metadata["page_count"] == 3
        assert chunks[0].metadata["merged_pages"] is True
