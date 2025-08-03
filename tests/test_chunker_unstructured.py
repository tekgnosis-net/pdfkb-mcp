"""Tests for the ChunkerUnstructured implementation."""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pdfkb.chunker.chunker_unstructured import ChunkerUnstructured
from pdfkb.config import ServerConfig
from pdfkb.exceptions import PDFProcessingError
from pdfkb.models import Chunk
from pdfkb.pdf_processor import PDFProcessor


class TestChunkerUnstructured:
    """Test cases for ChunkerUnstructured class."""

    @pytest.fixture
    def sample_markdown_content(self) -> str:
        """Create sample markdown content for testing."""
        return """# Document Title

## Introduction

This is the introduction section with some content.

## Main Content

### Subsection 1

This is the first subsection with detailed information.
- Bullet point 1
- Bullet point 2
- Bullet point 3

### Subsection 2

This is the second subsection with more information.

## Conclusion

This is the conclusion of the document.
"""

    @pytest.fixture
    def sample_metadata(self) -> Dict[str, Any]:
        """Create sample metadata for testing."""
        return {
            "document_id": "test_doc_123",
            "source": "test.pdf",
            "page_count": 5,
        }

    @pytest.fixture
    def chunker(self):
        """Create a ChunkerUnstructured instance."""
        try:
            return ChunkerUnstructured()
        except ImportError:
            pytest.skip("Unstructured library not available")

    def test_chunker_initialization_success(self):
        """Test successful initialization of ChunkerUnstructured."""
        try:
            chunker = ChunkerUnstructured()
            assert chunker is not None
            assert hasattr(chunker, "partition_text")
        except ImportError:
            pytest.skip("Unstructured library not available")

    def test_chunker_initialization_failure(self):
        """Test initialization failure when unstructured library is not available."""
        with patch("unstructured.partition.text.partition_text"):
            # Mock the import to raise ImportError
            with patch.dict(
                "sys.modules",
                {
                    "unstructured": None,
                    "unstructured.partition": None,
                    "unstructured.partition.text": None,
                },
            ):
                # Also patch the import statements directly
                with patch(
                    "builtins.__import__",
                    side_effect=ImportError("Unstructured library not available"),
                ):
                    with pytest.raises(ImportError, match="Unstructured library not available"):
                        ChunkerUnstructured()

    def test_chunk_success(self, chunker, sample_markdown_content, sample_metadata):
        """Test successful chunking of markdown content."""
        # Mock the unstructured partition_text function
        mock_element_1 = Mock()
        mock_element_1.__str__ = Mock(return_value="This is the first chunk content.")
        mock_element_1.__class__.__name__ = "NarrativeText"

        mock_element_2 = Mock()
        mock_element_2.__str__ = Mock(return_value="This is the second chunk content.")
        mock_element_2.__class__.__name__ = "Title"

        with patch.object(chunker, "partition_text", return_value=[mock_element_1, mock_element_2]):
            chunks = chunker.chunk(sample_markdown_content, sample_metadata)

            assert isinstance(chunks, list)
            assert len(chunks) == 2
            assert all(isinstance(chunk, Chunk) for chunk in chunks)

            # Check first chunk
            assert chunks[0].text == "This is the first chunk content."
            assert chunks[0].chunk_index == 0
            assert "chunk_strategy" in chunks[0].metadata
            assert chunks[0].metadata["chunk_strategy"] == "unstructured_by_title"
            assert chunks[0].metadata["element_type"] == "NarrativeText"

            # Check second chunk
            assert chunks[1].text == "This is the second chunk content."
            assert chunks[1].chunk_index == 1
            assert chunks[1].metadata["element_type"] == "Title"

            # Check that provided metadata is included
            assert "document_id" in chunks[0].metadata
            assert chunks[0].metadata["document_id"] == "test_doc_123"

    def test_chunk_empty_content(self, chunker, sample_metadata):
        """Test chunking with empty content."""
        chunks = chunker.chunk("", sample_metadata)
        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_chunk_whitespace_only_content(self, chunker, sample_metadata):
        """Test chunking with whitespace-only content."""
        chunks = chunker.chunk("   \n\t  \n  ", sample_metadata)
        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_chunk_with_none_content(self, chunker, sample_metadata):
        """Test chunking with None content."""
        chunks = chunker.chunk(None, sample_metadata)
        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_chunk_metadata_creation(self, chunker, sample_markdown_content, sample_metadata):
        """Test proper metadata creation for chunks."""
        mock_element = Mock()
        mock_element.__str__ = Mock(return_value="Test chunk content.")
        mock_element.__class__.__name__ = "ListItem"

        with patch.object(chunker, "partition_text", return_value=[mock_element]):
            chunks = chunker.chunk(sample_markdown_content, sample_metadata)

            assert len(chunks) == 1
            chunk = chunks[0]

            # Check required metadata fields
            assert "chunk_strategy" in chunk.metadata
            assert "element_type" in chunk.metadata
            assert "unstructured_version" in chunk.metadata
            assert "max_characters" in chunk.metadata
            assert "created_at" in chunk.metadata

            # Check metadata values
            assert chunk.metadata["chunk_strategy"] == "unstructured_by_title"
            assert chunk.metadata["element_type"] == "ListItem"
            assert chunk.metadata["max_characters"] == 1000

            # Check provided metadata is included
            assert chunk.metadata["document_id"] == "test_doc_123"
            assert chunk.metadata["source"] == "test.pdf"

    def test_chunk_skip_empty_elements(self, chunker, sample_markdown_content, sample_metadata):
        """Test that empty elements are skipped during chunking."""
        mock_element_1 = Mock()
        mock_element_1.__str__ = Mock(return_value="Valid content")
        mock_element_1.__class__.__name__ = "NarrativeText"

        mock_element_2 = Mock()
        mock_element_2.__str__ = Mock(return_value="")  # Empty content
        mock_element_2.__class__.__name__ = "NarrativeText"

        mock_element_3 = Mock()
        mock_element_3.__str__ = Mock(return_value="   ")  # Whitespace only
        mock_element_3.__class__.__name__ = "Title"

        mock_element_4 = Mock()
        mock_element_4.__str__ = Mock(return_value="Another valid content")
        mock_element_4.__class__.__name__ = "ListItem"

        with patch.object(
            chunker,
            "partition_text",
            return_value=[mock_element_1, mock_element_2, mock_element_3, mock_element_4],
        ):
            chunks = chunker.chunk(sample_markdown_content, sample_metadata)

            # Should only have 2 chunks (skipping empty ones)
            assert len(chunks) == 2
            assert chunks[0].text == "Valid content"
            assert chunks[1].text == "Another valid content"

    def test_chunk_error_handling(self, chunker, sample_markdown_content, sample_metadata):
        """Test error handling during chunking."""
        with patch.object(chunker, "partition_text", side_effect=Exception("Test error")):
            with pytest.raises(RuntimeError, match="Failed to chunk text"):
                chunker.chunk(sample_markdown_content, sample_metadata)

    def test_chunk_markdown_structures(self, chunker, sample_metadata):
        """Test chunking with different markdown structures."""
        # Test with headers
        markdown_with_headers = """# Main Title

## Section 1

Content with **bold** and *italic* text.

### Subsection

More content with [link](http://example.com) and `code`.

## Section 2

- List item 1
- List item 2
  - Nested item

> Blockquote content

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2   |
"""

        mock_elements = []
        for i, content in enumerate(
            [
                "Main Title",
                "Section 1",
                "Content with bold and italic text.",
                "Subsection",
                "More content with link and code.",
                "Section 2",
                "List item 1",
                "List item 2",
                "Nested item",
                "Blockquote content",
                "Column 1",
                "Column 2",
                "Cell 1",
                "Cell 2",
            ]
        ):
            mock_element = Mock()
            mock_element.__str__ = Mock(return_value=content)
            mock_element.__class__.__name__ = "NarrativeText"
            mock_elements.append(mock_element)

        with patch.object(chunker, "partition_text", return_value=mock_elements):
            chunks = chunker.chunk(markdown_with_headers, sample_metadata)

            assert isinstance(chunks, list)
            assert len(chunks) == 14  # All elements should create chunks
            assert all(isinstance(chunk, Chunk) for chunk in chunks)


class TestPDFProcessorWithUnstructuredChunker:
    """Test cases for PDFProcessor integration with Unstructured chunker."""

    @pytest.fixture
    def config(self):
        """Create a test configuration with unstructured chunker."""
        return ServerConfig(openai_api_key="sk-test-key", pdf_chunker="unstructured")

    @pytest.fixture
    def embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        service.generate_embeddings = Mock(return_value=[[0.1, 0.2, 0.3]])
        return service

    def test_pdf_processor_chunker_selection_unstructured(self, config, embedding_service):
        """Test PDFProcessor selects Unstructured chunker when configured."""
        try:
            processor = PDFProcessor(config, embedding_service)
            assert isinstance(processor.chunker, ChunkerUnstructured)
        except ImportError:
            pytest.skip("Unstructured library not available")

    def test_pdf_processor_chunker_selection_fallback(self, config, embedding_service):
        """Test PDFProcessor fallback when Unstructured chunker is not available."""
        # Configure to use unstructured chunker
        config.pdf_chunker = "unstructured"

        # Mock ImportError for unstructured
        with patch("pdfkb.chunker.chunker_unstructured.ChunkerUnstructured.__init__") as mock_init:
            mock_init.side_effect = ImportError("unstructured not available")

            with pytest.raises(PDFProcessingError, match="Unstructured chunker not available"):
                PDFProcessor(config, embedding_service)

    @pytest.mark.asyncio
    async def test_pdf_processor_with_unstructured_chunker_integration(self, config, embedding_service, tmp_path):
        """Test PDFProcessor integration with Unstructured chunker."""
        try:
            processor = PDFProcessor(config, embedding_service)

            # Create a dummy PDF file
            pdf_file = tmp_path / "test.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\ntest content")

            # Mock the parser and chunker
            mock_parse_result = Mock()
            mock_parse_result.markdown_content = "# Test\n\nThis is test content."
            mock_parse_result.metadata = {"page_count": 1}

            # Use proper async mock for parse method
            processor.parser.parse = AsyncMock(return_value=mock_parse_result)

            # Mock unstructured partition_text
            mock_element = Mock()
            mock_element.__str__ = Mock(return_value="Test chunk content.")
            mock_element.__class__.__name__ = "NarrativeText"

            processor.chunker.partition_text = Mock(return_value=[mock_element])

            # Mock the embedding service generate_embeddings method to be proper async mock
            embedding_service.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

            # Process the PDF
            result = await processor.process_pdf(pdf_file)

            # Check that processing was successful
            assert result.success is True
            assert result.document is not None
            assert len(result.document.chunks) > 0

            # Check chunk properties
            chunk = result.document.chunks[0]
            assert chunk.text == "Test chunk content."
            assert "chunk_strategy" in chunk.metadata
            assert chunk.metadata["chunk_strategy"] == "unstructured_by_title"

        except ImportError:
            pytest.skip("Unstructured library not available")


class TestConfigChanges:
    """Test cases for configuration changes affecting chunker selection."""

    def test_config_pdf_chunker_unstructured(self):
        """Test configuration with pdf_chunker set to 'unstructured'."""
        config = ServerConfig(openai_api_key="sk-test-key", pdf_chunker="unstructured")
        assert config.pdf_chunker == "unstructured"

    def test_config_pdf_chunker_langchain(self):
        """Test configuration with pdf_chunker set to 'langchain'."""
        config = ServerConfig(openai_api_key="sk-test-key", pdf_chunker="langchain")
        assert config.pdf_chunker == "langchain"

    def test_config_invalid_chunker(self):
        """Test configuration with invalid chunker raises error."""
        with pytest.raises(Exception):  # ConfigurationError or validation error
            ServerConfig(openai_api_key="sk-test-key", pdf_chunker="invalid_chunker")
