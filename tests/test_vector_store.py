"""Tests for the vector store module."""

import pytest

from pdfkb.config import ServerConfig
from pdfkb.models import Chunk, Document, SearchQuery
from pdfkb.vector_store import VectorStore


class TestVectorStore:
    """Test cases for VectorStore class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            vector_search_k=5,
        )

    @pytest.fixture
    def vector_store(self, config):
        """Create a VectorStore instance."""
        return VectorStore(config)

    @pytest.fixture
    def sample_document(self):
        """Create a sample document with chunks."""
        doc = Document(
            id="test-doc-1",
            path="/test/sample.pdf",
            title="Sample Document",
        )

        chunk = Chunk(
            id="test-chunk-1",
            document_id=doc.id,
            text="This is a sample text chunk.",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            page_number=1,
        )

        doc.add_chunk(chunk)
        return doc

    @pytest.mark.asyncio
    async def test_initialize_vector_store(self, vector_store, monkeypatch):
        """Test initializing the vector store with mocked chroma client."""

        class DummyCollection:
            def __init__(self):
                self.name = "pdf_knowledgebase"

            def add(self, *args, **kwargs):
                return None

            def delete(self, *args, **kwargs):
                return None

            def count(self):
                return 0

            def get(self, *args, **kwargs):
                return {"metadatas": []}

            def query(self, *args, **kwargs):
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            def get_or_create_collection(self, name, metadata=None):
                return DummyCollection()

        # Patch chromadb import used inside initialize()
        import importlib
        import types

        chroma_stub = types.SimpleNamespace(PersistentClient=DummyClient)
        original_import_module = importlib.import_module

        def fake_import(name, package=None):
            if name == "chromadb":
                return chroma_stub
            if name == "chromadb.config":
                Settings = type("Settings", (), {"__init__": lambda self, **kwargs: None})
                return types.SimpleNamespace(Settings=Settings)
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import)

        await vector_store.initialize()
        assert vector_store.collection_name == "pdf_knowledgebase"

    @pytest.mark.asyncio
    async def test_add_document_no_chunks(self, vector_store, config):
        """Test adding a document with no chunks."""
        doc = Document(id="empty-doc", path="/test/empty.pdf")

        # Should not raise an error
        await vector_store.add_document(doc)

    @pytest.mark.asyncio
    async def test_add_document_no_embeddings(self, vector_store):
        """Test adding a document with chunks but no embeddings."""
        doc = Document(id="no-embed-doc", path="/test/no_embed.pdf")
        chunk = Chunk(
            id="no-embed-chunk",
            document_id=doc.id,
            text="Text without embedding",
        )
        doc.add_chunk(chunk)

        # Should not raise an error
        await vector_store.add_document(doc)

    @pytest.mark.asyncio
    async def test_search_empty_query(self, vector_store):
        """Test searching with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            SearchQuery(query="")

    @pytest.mark.asyncio
    async def test_search_invalid_limit(self, vector_store):
        """Test searching with invalid limit."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            SearchQuery(query="test", limit=0)

    @pytest.mark.asyncio
    async def test_search_invalid_min_score(self, vector_store):
        """Test searching with invalid min_score."""
        with pytest.raises(ValueError, match="min_score must be between 0 and 1"):
            SearchQuery(query="test", min_score=1.5)

    @pytest.mark.asyncio
    async def test_delete_document(self, vector_store, monkeypatch):
        """Test deleting a document with mocked chroma client."""

        class DummyCollection:
            async def delete(self, *args, **kwargs):
                return None

        class DummyClient:
            def get_or_create_collection(self, name):
                return DummyCollection()

        import importlib
        import types

        chroma_stub = types.SimpleNamespace(PersistentClient=DummyClient)
        original_import_module = importlib.import_module

        def fake_import(name, package=None):
            if name == "chromadb":
                return chroma_stub
            if name == "chromadb.config":
                Settings = type("Settings", (), {"__init__": lambda self, **kwargs: None})
                return types.SimpleNamespace(Settings=Settings)
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import)

        await vector_store.initialize()
        # Should not raise an error
        await vector_store.delete_document("test-doc-id")

    @pytest.mark.asyncio
    async def test_get_document_count(self, vector_store, monkeypatch):
        """Test getting document count with mocked collection."""

        # Create a single DummyCollection instance so identity is preserved
        class DummyCollection:
            def get(self, include=None, ids=None, where=None):
                # Simulate metadatas for 42 unique documents
                metadatas = [{"document_id": f"doc-{i}"} for i in range(42)]
                return {"metadatas": metadatas}

            # Provide count as an alternative fallback
            def count(self):
                return 42

        dummy_collection = DummyCollection()

        class DummyClient:
            def __init__(self, *args, **kwargs):
                # Always return the same collection instance
                self._collection = dummy_collection

            def get_or_create_collection(self, name, metadata=None):
                return self._collection

            def delete_collection(self, name):
                return None

            def create_collection(self, name, metadata=None):
                return self._collection

        import importlib
        import types

        chroma_stub = types.SimpleNamespace(PersistentClient=DummyClient)
        original_import_module = importlib.import_module

        def fake_import(name, package=None):
            if name == "chromadb":
                return chroma_stub
            if name == "chromadb.config":
                Settings = type("Settings", (), {"__init__": lambda self, **kwargs: None})
                return types.SimpleNamespace(Settings=Settings)
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        await vector_store.initialize()
        # Ensure the exact dummy collection instance is used
        vector_store.collection = dummy_collection

        count = await vector_store.get_document_count()
        assert isinstance(count, int)
        assert count == 42

    @pytest.mark.asyncio
    async def test_get_chunk_count(self, vector_store, monkeypatch):
        """Test getting chunk count with mocked collection."""

        class DummyCollection:
            def count(self):
                return 17

            def get(self, include=None, ids=None, where=None):
                return {"ids": [f"id-{i}" for i in range(17)]}

        dummy_collection = DummyCollection()

        class DummyClient:
            def __init__(self, *args, **kwargs):
                self._collection = dummy_collection

            def get_or_create_collection(self, name, metadata=None):
                return self._collection

            def delete_collection(self, name):
                return None

            def create_collection(self, name, metadata=None):
                return self._collection

        import importlib
        import types

        chroma_stub = types.SimpleNamespace(PersistentClient=DummyClient)
        original_import_module = importlib.import_module

        def fake_import(name, package=None):
            if name == "chromadb":
                return chroma_stub
            if name == "chromadb.config":
                Settings = type("Settings", (), {"__init__": lambda self, **kwargs: None})
                return types.SimpleNamespace(Settings=Settings)
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        await vector_store.initialize()
        vector_store.collection = dummy_collection

        count = await vector_store.get_chunk_count()
        assert isinstance(count, int)
        assert count == 17

    def test_prepare_chunk_metadata(self, vector_store, sample_document):
        """Test preparing chunk metadata for storage."""
        chunk = sample_document.chunks[0]
        metadata = vector_store._prepare_chunk_metadata(chunk, sample_document)

        assert metadata["document_id"] == sample_document.id
        assert metadata["document_path"] == sample_document.path
        assert metadata["document_title"] == sample_document.title
        assert metadata["chunk_index"] == chunk.chunk_index
        assert metadata["page_number"] == chunk.page_number

    def test_chunk_from_metadata(self, vector_store):
        """Test creating chunk from metadata."""
        metadata = {
            "document_id": "test-doc",
            "page_number": 1,
            "chunk_index": 0,
        }

        chunk = vector_store._chunk_from_metadata("test-chunk", "test text", metadata)

        assert chunk.id == "test-chunk"
        assert chunk.document_id == "test-doc"
        assert chunk.text == "test text"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0

    def test_document_from_metadata(self, vector_store):
        """Test creating document from metadata."""
        metadata = {
            "document_id": "test-doc",
            "document_path": "/test/doc.pdf",
            "document_title": "Test Document",
        }

        doc = vector_store._document_from_metadata(metadata)

        assert doc.id == "test-doc"
        assert doc.path == "/test/doc.pdf"
        assert doc.title == "Test Document"

    # TODO: Add more comprehensive tests when real implementation is added
    # - Test actual Chroma operations
    # - Test search functionality
    # - Test error handling scenarios
    # - Test metadata filtering
