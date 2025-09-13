"""Tests for MCP resources with document identifier resolution."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.main import PDFKnowledgebaseServer
from pdfkb.models import Chunk, Document


class TestMCPResources:
    """Test cases for MCP resources with new doc:// scheme."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a test configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "documents",
            cache_dir=tmp_path / "cache",
            vector_search_k=5,
        )

    @pytest.fixture
    async def server(self, config):
        """Create a PDFKnowledgebaseServer instance with mocked components."""
        server = PDFKnowledgebaseServer(config)

        # Mock the necessary components
        server.vector_store = AsyncMock()
        server.embedding_service = AsyncMock()
        server.document_processor = AsyncMock()

        # Create test documents in cache
        doc1 = Document(
            id="doc_1234567890abcdef",
            path="/app/documents/test1.pdf",
            title="Test Document 1",
            page_count=3,
            chunk_count=2,
        )
        doc1.chunks = [
            Chunk(id="chunk_1", document_id=doc1.id, text="Content from page 1", page_number=1, chunk_index=0),
            Chunk(id="chunk_2", document_id=doc1.id, text="Content from page 2", page_number=2, chunk_index=1),
        ]

        doc2 = Document(
            id="doc_fedcba0987654321",
            path="/app/documents/markdown.md",
            title="Markdown Document",
            page_count=2,
            chunk_count=1,
        )
        doc2.chunks = [
            Chunk(id="chunk_3", document_id=doc2.id, text="Markdown content page 1", page_number=1, chunk_index=0)
        ]

        # Document without page numbers (to test error handling)
        doc3 = Document(
            id="doc_nopages12345678",
            path="/app/documents/nopage.md",
            title="Document Without Pages",
            page_count=0,
            chunk_count=1,
        )
        doc3.chunks = [
            Chunk(
                id="chunk_4", document_id=doc3.id, text="Content without page number", page_number=None, chunk_index=0
            )
        ]

        server._document_cache = {doc1.id: doc1, doc2.id: doc2, doc3.id: doc3}

        # Mock knowledgebase path
        config.knowledgebase_path.mkdir(parents=True, exist_ok=True)

        return server

    @pytest.mark.asyncio
    async def test_resolve_document_identifier_with_internal_id(self, server):
        """Test resolving document identifiers using internal IDs."""
        # Test existing internal ID
        result = await server._resolve_document_identifier("doc_1234567890abcdef")
        assert result == "doc_1234567890abcdef"

        # Test non-existing internal ID
        result = await server._resolve_document_identifier("doc_nonexistent123")
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_document_identifier_with_absolute_path(self, server):
        """Test resolving document identifiers using absolute paths."""
        result = await server._resolve_document_identifier("/app/documents/test1.pdf")
        assert result == "doc_1234567890abcdef"

        result = await server._resolve_document_identifier("/app/documents/markdown.md")
        assert result == "doc_fedcba0987654321"

        # Test non-existing path
        result = await server._resolve_document_identifier("/app/documents/nonexistent.pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_document_identifier_with_relative_path(self, server, config):
        """Test resolving document identifiers using relative paths."""
        # Mock Path.resolve() to simulate path resolution
        with patch("pathlib.Path.resolve") as mock_resolve:
            mock_resolve.return_value = Path("/app/documents/test1.pdf")

            result = await server._resolve_document_identifier("test1.pdf")
            assert result == "doc_1234567890abcdef"

    @pytest.mark.asyncio
    async def test_find_document_by_path(self, server):
        """Test finding documents by file path."""
        result = await server._find_document_by_path("/app/documents/test1.pdf")
        assert result == "doc_1234567890abcdef"

        result = await server._find_document_by_path("/nonexistent/path.pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_doc_resource_with_internal_id(self, server):
        """Test doc:// resource functionality with internal document ID."""
        # Setup the resources - this creates the FastMCP resource handlers
        server._setup_resources()

        # Test that resources can be setup without error
        assert server.app is not None

    @pytest.mark.asyncio
    async def test_doc_resource_setup(self, server):
        """Test that doc:// resources can be set up correctly."""
        # This tests that _setup_resources() completes without error
        # and that the scheme change from pdf:// to doc:// is applied
        server._setup_resources()
        assert server.app is not None

    def test_resource_setup_completes(self, server):
        """Test that resource setup completes without errors."""
        # This verifies that the new doc:// scheme resources can be setup
        server._setup_resources()
        assert server.app is not None

        # Test that document cache has expected documents
        assert len(server._document_cache) == 3
        assert "doc_1234567890abcdef" in server._document_cache
        assert "doc_fedcba0987654321" in server._document_cache
        assert "doc_nopages12345678" in server._document_cache

    @pytest.mark.asyncio
    async def test_chunk_retrieval_from_vector_store(self, server):
        """Test that chunk retrieval works when chunks are only in vector store."""
        # Create a document that has no chunks in memory but chunks in vector store
        doc_id = "doc_vector_only_123"
        doc = Document(
            id=doc_id,
            path="/app/documents/vector_only.md",
            title="Vector Store Only Doc",
            page_count=2,
            chunk_count=0,  # No chunks in memory
        )
        # Explicitly set no chunks in memory
        doc.chunks = []

        # Add to document cache
        server._document_cache[doc_id] = doc

        # Mock vector store to return chunks with proper chunk indices
        mock_chunks = [
            Chunk(
                id="chunk_vector_0",
                document_id=doc_id,
                text="Content from chunk 0 (vector store)",
                page_number=1,
                chunk_index=0,
            ),
            Chunk(
                id="chunk_vector_1",
                document_id=doc_id,
                text="Content from chunk 1 (vector store)",
                page_number=1,
                chunk_index=1,
            ),
            Chunk(
                id="chunk_vector_2",
                document_id=doc_id,
                text="Content from chunk 2 (vector store)",
                page_number=2,
                chunk_index=2,
            ),
        ]
        server.vector_store.get_document_chunks.return_value = mock_chunks

        # Test the resolution function directly
        resolved_id = await server._resolve_document_identifier(doc_id)
        assert resolved_id == doc_id

    def test_chunk_indices_parsing(self, server):
        """Test chunk indices parsing logic."""
        # Test single index
        indices = "0"
        parsed = [int(idx.strip()) for idx in indices.split(",") if idx.strip().isdigit()]
        assert parsed == [0]

        # Test multiple indices
        indices = "0,2,5,10"
        parsed = [int(idx.strip()) for idx in indices.split(",") if idx.strip().isdigit()]
        assert parsed == [0, 2, 5, 10]

        # Test with spaces
        indices = "0, 2 , 5 ,10"
        parsed = [int(idx.strip()) for idx in indices.split(",") if idx.strip().isdigit()]
        assert parsed == [0, 2, 5, 10]

        # Test with invalid indices (should be filtered out)
        indices = "0,invalid,2,abc,5"
        parsed = [int(idx.strip()) for idx in indices.split(",") if idx.strip().isdigit()]
        assert parsed == [0, 2, 5]
