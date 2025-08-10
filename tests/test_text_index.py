"""Unit tests for TextIndex class."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from whoosh import index

from pdfkb.config import ServerConfig
from pdfkb.models import Chunk, Document
from pdfkb.text_index import TextIndex


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test indexes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    config = MagicMock(spec=ServerConfig)
    config.whoosh_index_dir = str(temp_dir / "whoosh")
    config.whoosh_analyzer = "standard"
    config.whoosh_min_score = 0.0
    config.enable_hybrid_search = True
    return config


@pytest.fixture
def text_index(test_config):
    """Create a TextIndex instance."""
    return TextIndex(test_config)


@pytest.fixture
def sample_document():
    """Create a sample document with chunks."""
    doc = Document(id="doc_test123", path="/test/document.pdf", title="Test Document", checksum="abc123")

    chunks = [
        Chunk(
            id="chunk_001",
            document_id=doc.id,
            text="This is the first chunk about machine learning and AI.",
            page_number=1,
            chunk_index=0,
        ),
        Chunk(
            id="chunk_002",
            document_id=doc.id,
            text="The second chunk discusses neural networks and deep learning.",
            page_number=1,
            chunk_index=1,
        ),
        Chunk(
            id="chunk_003",
            document_id=doc.id,
            text="Python is a great language for data science.",
            page_number=2,
            chunk_index=2,
        ),
    ]

    doc.chunks = chunks
    return doc


@pytest.mark.asyncio
@pytest.mark.unit
class TestTextIndex:
    """Test suite for TextIndex class."""

    async def test_initialization(self, text_index, temp_dir):
        """Test TextIndex initialization."""
        await text_index.initialize()

        # Check that index was created
        assert text_index.index is not None
        assert Path(text_index.index_dir).exists()
        assert index.exists_in(str(text_index.index_dir))

    async def test_schema_creation(self, text_index):
        """Test schema creation with correct fields."""
        schema = text_index._create_schema()

        # Check schema fields
        assert "chunk_id" in schema
        assert "document_id" in schema
        assert "text" in schema
        assert "metadata" in schema
        assert "page_number" in schema
        assert "chunk_index" in schema

    async def test_add_document(self, text_index, sample_document):
        """Test adding a document to the index."""
        await text_index.initialize()
        await text_index.add_document(sample_document)

        # Verify chunks were added
        with text_index.index.searcher() as searcher:
            assert searcher.doc_count_all() == 3

            # Check specific chunk
            doc = searcher.document(chunk_id="chunk_001")
            assert doc is not None
            assert doc["document_id"] == sample_document.id
            assert "machine learning" in doc["text"]

    async def test_add_document_empty_chunks(self, text_index):
        """Test adding a document with no chunks."""
        doc = Document(id="empty_doc", path="/test/empty.pdf", checksum="xyz")

        await text_index.initialize()
        await text_index.add_document(doc)

        # Should handle gracefully
        with text_index.index.searcher() as searcher:
            assert searcher.doc_count_all() == 0

    async def test_search_basic(self, text_index, sample_document):
        """Test basic search functionality."""
        await text_index.initialize()
        await text_index.add_document(sample_document)

        # Search for a term
        results = await text_index.search("machine learning", limit=5)

        assert len(results) > 0
        assert results[0]["chunk_id"] == "chunk_001"
        assert "machine learning" in results[0]["text"].lower()
        assert results[0]["score"] > 0

    async def test_search_multiple_results(self, text_index, sample_document):
        """Test search returning multiple results."""
        await text_index.initialize()
        await text_index.add_document(sample_document)

        # Search for a common term
        results = await text_index.search("learning", limit=10)

        # Should find at least 2 chunks
        assert len(results) >= 2
        # Results should be sorted by score
        for i in range(1, len(results)):
            assert results[i - 1]["score"] >= results[i]["score"]

    async def test_search_no_results(self, text_index, sample_document):
        """Test search with no matching results."""
        await text_index.initialize()
        await text_index.add_document(sample_document)

        # Search for non-existent term
        results = await text_index.search("quantum computing blockchain", limit=5)

        assert len(results) == 0

    async def test_search_with_limit(self, text_index, sample_document):
        """Test search respects limit parameter."""
        await text_index.initialize()
        await text_index.add_document(sample_document)

        # Add more documents
        for i in range(5):
            doc = Document(id=f"doc_{i}", path=f"/test/doc_{i}.pdf", checksum=f"hash_{i}")
            doc.chunks = [
                Chunk(
                    id=f"chunk_{i}_{j}",
                    document_id=doc.id,
                    text=f"This is about learning and education in document {i}",
                    page_number=1,
                    chunk_index=j,
                )
                for j in range(3)
            ]
            await text_index.add_document(doc)

        # Search with limit
        results = await text_index.search("learning", limit=3)
        assert len(results) <= 3

    async def test_delete_document(self, text_index, sample_document):
        """Test deleting a document from the index."""
        await text_index.initialize()
        await text_index.add_document(sample_document)

        # Verify document exists
        with text_index.index.searcher() as searcher:
            # Count actual documents (not deleted)
            initial_docs = list(searcher.documents(document_id=sample_document.id))
            assert len(initial_docs) == 3

        # Delete document
        await text_index.delete_document(sample_document.id)

        # Verify deletion - check that documents are no longer returned
        with text_index.index.searcher() as searcher:
            # Check no documents with this ID are returned
            remaining_docs = list(searcher.documents(document_id=sample_document.id))
            assert len(remaining_docs) == 0
            # Check specific chunk is gone
            doc = searcher.document(chunk_id="chunk_001")
            assert doc is None

    async def test_get_document_count(self, text_index):
        """Test counting unique documents."""
        await text_index.initialize()

        # Add multiple documents
        for i in range(3):
            doc = Document(id=f"doc_{i}", path=f"/test/doc_{i}.pdf", checksum=f"hash_{i}")
            doc.chunks = [
                Chunk(id=f"chunk_{i}_{j}", document_id=doc.id, text=f"Content {j}", chunk_index=j) for j in range(2)
            ]
            await text_index.add_document(doc)

        count = await text_index.get_document_count()
        assert count == 3

    async def test_get_chunk_count(self, text_index):
        """Test counting total chunks."""
        await text_index.initialize()

        # Add documents with multiple chunks
        for i in range(2):
            doc = Document(id=f"doc_{i}", path=f"/test/doc_{i}.pdf", checksum=f"hash_{i}")
            doc.chunks = [
                Chunk(id=f"chunk_{i}_{j}", document_id=doc.id, text=f"Content {j}", chunk_index=j) for j in range(3)
            ]
            await text_index.add_document(doc)

        count = await text_index.get_chunk_count()
        assert count == 6  # 2 documents * 3 chunks each

    async def test_reset_index(self, text_index, sample_document):
        """Test resetting the index."""
        await text_index.initialize()
        await text_index.add_document(sample_document)

        # Verify data exists
        with text_index.index.searcher() as searcher:
            assert searcher.doc_count_all() > 0

        # Reset index
        await text_index.reset_index()

        # Verify index is empty
        with text_index.index.searcher() as searcher:
            assert searcher.doc_count_all() == 0

    async def test_update_existing_chunk(self, text_index, sample_document):
        """Test updating an existing chunk."""
        await text_index.initialize()
        await text_index.add_document(sample_document)

        # Modify and re-add the same document
        sample_document.chunks[0].text = "Updated text for the first chunk"
        await text_index.add_document(sample_document)

        # Verify update
        with text_index.index.searcher() as searcher:
            # Count documents for this document_id (should still be 3)
            docs = list(searcher.documents(document_id=sample_document.id))
            assert len(docs) == 3

            # Check updated content
            doc = searcher.document(chunk_id="chunk_001")
            assert doc is not None
            assert "Updated text" in doc["text"]

    async def test_concurrent_writes(self, text_index):
        """Test concurrent write operations are properly synchronized."""
        await text_index.initialize()

        # Create multiple documents
        docs = []
        for i in range(5):
            doc = Document(id=f"doc_{i}", path=f"/test/doc_{i}.pdf", checksum=f"hash_{i}")
            doc.chunks = [Chunk(id=f"chunk_{i}", document_id=doc.id, text=f"Content for document {i}", chunk_index=0)]
            docs.append(doc)

        # Add documents concurrently
        tasks = [text_index.add_document(doc) for doc in docs]
        await asyncio.gather(*tasks)

        # Verify all documents were added
        with text_index.index.searcher() as searcher:
            assert searcher.doc_count_all() == 5

    async def test_stemming_analyzer(self, temp_dir):
        """Test using stemming analyzer."""
        config = MagicMock(spec=ServerConfig)
        config.whoosh_index_dir = str(temp_dir / "whoosh_stemming")
        config.whoosh_analyzer = "stemming"
        config.whoosh_min_score = 0.0

        index = TextIndex(config)
        await index.initialize()

        # Add document with related terms
        doc = Document(id="doc_1", path="/test.pdf", checksum="abc")
        doc.chunks = [Chunk(id="chunk_1", document_id=doc.id, text="running runners run", chunk_index=0)]
        await index.add_document(doc)

        # Search should find stemmed variations
        results = await index.search("run", limit=5)
        assert len(results) > 0

    async def test_minimum_score_threshold(self, test_config, sample_document):
        """Test minimum score threshold filtering."""
        test_config.whoosh_min_score = 0.5
        index = TextIndex(test_config)

        await index.initialize()
        await index.add_document(sample_document)

        # Search with high threshold
        results = await index.search("the", limit=10)

        # Common words should have lower scores and be filtered
        for result in results:
            assert result["score"] >= 0.5

    async def test_error_handling_search(self, text_index):
        """Test error handling during search."""
        # Search without initialization should handle gracefully
        results = await text_index.search("test", limit=5)
        assert results == []

    async def test_close_index(self, text_index):
        """Test closing the index."""
        await text_index.initialize()
        assert text_index.index is not None

        await text_index.close()
        assert text_index.index is None
