"""Tests for deduplication functionality."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.pdfkb.config import ServerConfig
from src.pdfkb.models import Chunk, Document
from src.pdfkb.vector_store import VectorStore


class TestChunkDeduplication:
    """Test cases for chunk deduplication logic."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=Path("./test_pdfs"),
            cache_dir=Path("./test_cache"),
            chunk_size=500,
            chunk_overlap=50,
            embedding_model="text-embedding-3-small",
        )

    @pytest.fixture
    def sample_document(self):
        """Create a sample document with chunks."""
        doc = Document(
            path="/test/sample.pdf",
            title="Test Document",
            checksum="abc123",
            file_size=1000,
            page_count=2,
        )

        # Add chunks with deterministic content for testing
        chunk1 = Chunk(
            document_id=doc.id,
            text="This is the first chunk of text.",
            chunk_index=0,
            page_number=1,
            embedding=[0.1, 0.2, 0.3],
        )

        chunk2 = Chunk(
            document_id=doc.id,
            text="This is the second chunk of text.",
            chunk_index=1,
            page_number=1,
            embedding=[0.4, 0.5, 0.6],
        )

        doc.add_chunk(chunk1)
        doc.add_chunk(chunk2)

        return doc

    def test_chunk_deterministic_id_generation(self, sample_document):
        """Test that chunks generate deterministic IDs based on content."""
        doc = sample_document
        chunk1, chunk2 = doc.chunks

        # IDs should be deterministic and different
        assert chunk1.id.startswith("chunk_")
        assert chunk2.id.startswith("chunk_")
        assert chunk1.id != chunk2.id

        # Same content should produce same ID
        duplicate_chunk = Chunk(
            document_id=doc.id,
            text="This is the first chunk of text.",
            chunk_index=0,
            page_number=1,
            embedding=[0.7, 0.8, 0.9],
        )

        assert duplicate_chunk.id == chunk1.id

    def test_document_deterministic_id_generation(self):
        """Test that documents generate deterministic IDs based on path and checksum."""
        doc1 = Document(path="/test/sample.pdf", checksum="abc123")

        doc2 = Document(path="/test/sample.pdf", checksum="abc123")

        doc3 = Document(path="/test/different.pdf", checksum="abc123")

        # Same path and checksum should produce same ID
        assert doc1.id == doc2.id
        assert doc1.id.startswith("doc_")

        # Different path should produce different ID
        assert doc1.id != doc3.id

    @pytest.mark.asyncio
    async def test_vector_store_filter_existing_chunks(self, config):
        """Test that vector store correctly filters out existing chunks."""
        vector_store = VectorStore(config)

        # Mock the collection
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["chunk_existing1", "chunk_existing2"]}
        vector_store.collection = mock_collection

        # Create test chunks
        existing_chunk = Chunk(id="chunk_existing1", text="Existing chunk", embedding=[0.1, 0.2, 0.3])

        new_chunk = Chunk(id="chunk_new1", text="New chunk", embedding=[0.4, 0.5, 0.6])

        chunks = [existing_chunk, new_chunk]

        # Filter chunks
        new_chunks = await vector_store._filter_existing_chunks(chunks)

        # Should only return the new chunk
        assert len(new_chunks) == 1
        assert new_chunks[0].id == "chunk_new1"

        # Verify collection.get was called with correct IDs
        mock_collection.get.assert_called_once_with(ids=["chunk_existing1", "chunk_new1"], include=["metadatas"])

    @pytest.mark.asyncio
    async def test_add_document_with_duplicates(self, config, sample_document):
        """Test adding document with some duplicate chunks."""
        vector_store = VectorStore(config)

        # Mock the collection
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": [sample_document.chunks[0].id]}  # First chunk already exists
        mock_collection.add = Mock()
        vector_store.collection = mock_collection

        # Add document
        await vector_store.add_document(sample_document)

        # Should only add the second chunk (first is duplicate)
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert len(call_args["ids"]) == 1
        assert call_args["ids"][0] == sample_document.chunks[1].id

    @pytest.mark.asyncio
    async def test_add_document_all_duplicates(self, config, sample_document):
        """Test adding document where all chunks are duplicates."""
        vector_store = VectorStore(config)

        # Mock the collection - all chunks exist
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": [chunk.id for chunk in sample_document.chunks]}
        mock_collection.add = Mock()
        vector_store.collection = mock_collection

        # Add document
        await vector_store.add_document(sample_document)

        # Should not add any chunks
        mock_collection.add.assert_not_called()

    def test_chunk_content_changes_affect_id(self):
        """Test that changes to chunk content result in different IDs."""
        base_chunk = Chunk(document_id="doc_123", text="Original text", chunk_index=0, page_number=1)

        modified_text_chunk = Chunk(document_id="doc_123", text="Modified text", chunk_index=0, page_number=1)

        modified_index_chunk = Chunk(
            document_id="doc_123",
            text="Original text",
            chunk_index=1,  # Different index
            page_number=1,
        )

        # Different content should produce different IDs
        assert base_chunk.id != modified_text_chunk.id
        assert base_chunk.id != modified_index_chunk.id
        assert modified_text_chunk.id != modified_index_chunk.id


class TestDuplicateChunkScenarios:
    """Test various real-world duplicate chunk scenarios."""

    @pytest.mark.asyncio
    async def test_reprocessing_same_file(self):
        """Test that reprocessing the same file doesn't create duplicates."""
        # This would be tested with integration tests
        # where we process the same PDF twice and verify no duplicates
        pass

    @pytest.mark.asyncio
    async def test_file_monitor_and_manual_add_coordination(self):
        """Test that file monitor and manual add_document don't create duplicates."""
        # This would test the coordination between different processing paths
        pass
