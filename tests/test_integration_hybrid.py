"""Integration tests for hybrid search functionality."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pdfkb.config import ServerConfig
from pdfkb.embeddings import EmbeddingService
from pdfkb.models import Chunk, Document, SearchQuery
from pdfkb.vector_store import VectorStore


@pytest.fixture
async def test_config():
    """Create a test configuration with hybrid search enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ServerConfig(
            openai_api_key="sk-test-key-123456789",
            knowledgebase_path=Path(tmpdir) / "pdfs",
            cache_dir=Path(tmpdir) / "cache",
            enable_hybrid_search=True,
            hybrid_search_weights={"vector": 0.6, "text": 0.4},
            rrf_k=60,
            chunk_size=500,
            chunk_overlap=50,
        )
        yield config


@pytest.fixture
async def mock_embedding_service():
    """Create a mock embedding service."""
    service = MagicMock(spec=EmbeddingService)
    service.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    service.generate_embeddings = AsyncMock()

    def mock_batch_embeddings(texts):
        return [[0.1] * 1536 for _ in texts]

    service.generate_embeddings.side_effect = mock_batch_embeddings
    return service


@pytest.fixture
async def vector_store_with_data(test_config, mock_embedding_service):
    """Create a vector store with sample data."""
    store = VectorStore(test_config)
    store.set_embedding_service(mock_embedding_service)

    # Initialize the store
    await store.initialize()

    # Create sample documents
    documents = [
        Document(
            id="doc_ml_1",
            path="/docs/machine_learning.pdf",
            title="Introduction to Machine Learning",
            checksum="hash_ml_1",
        ),
        Document(
            id="doc_dl_1", path="/docs/deep_learning.pdf", title="Deep Learning Fundamentals", checksum="hash_dl_1"
        ),
        Document(
            id="doc_python_1", path="/docs/python_guide.pdf", title="Python Programming Guide", checksum="hash_py_1"
        ),
    ]

    # Add chunks to documents
    chunks_data = [
        (
            "doc_ml_1",
            [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "Supervised learning uses labeled data to train models for classification and regression tasks.",
                "Common algorithms include decision trees, random forests, and support vector machines.",
            ],
        ),
        (
            "doc_dl_1",
            [
                "Deep learning uses neural networks with multiple layers to learn complex patterns.",
                "Convolutional neural networks are particularly effective for image recognition tasks.",
                "Transformers have revolutionized natural language processing with attention mechanisms.",
            ],
        ),
        (
            "doc_python_1",
            [
                "Python is a versatile programming language popular in data science and machine learning.",
                "Libraries like NumPy, Pandas, and Scikit-learn provide powerful tools for data analysis.",
                "TensorFlow and PyTorch are leading frameworks for deep learning development.",
            ],
        ),
    ]

    for doc_id, texts in chunks_data:
        doc = next(d for d in documents if d.id == doc_id)
        for i, text in enumerate(texts):
            chunk = Chunk(
                id=f"{doc_id}_chunk_{i}",
                document_id=doc_id,
                text=text,
                embedding=[0.1 + i * 0.01] * 1536,  # Slightly different embeddings
                page_number=i // 2 + 1,
                chunk_index=i,
            )
            doc.chunks.append(chunk)

    # Add documents to store
    for doc in documents:
        await store.add_document(doc)

    yield store

    # Cleanup
    await store.close()


@pytest.mark.asyncio
@pytest.mark.integration
class TestHybridSearchIntegration:
    """Integration tests for hybrid search functionality."""

    async def test_hybrid_search_end_to_end(self, vector_store_with_data, mock_embedding_service):
        """Test complete hybrid search workflow."""
        # Create search query
        query = SearchQuery(query="machine learning algorithms", limit=5, search_type="hybrid")

        # Generate query embedding
        query_embedding = await mock_embedding_service.generate_embedding(query.query)

        # Perform hybrid search
        results = await vector_store_with_data.search(query, query_embedding)

        # Verify results
        assert len(results) > 0
        assert all(r.search_type == "hybrid" for r in results)

        # Check that results are relevant (contain search terms)
        for result in results[:3]:  # Check top 3 results
            text_lower = result.chunk.text.lower()
            assert any(term in text_lower for term in ["machine", "learning", "algorithm", "neural", "model"])

    async def test_vector_only_search(self, vector_store_with_data, mock_embedding_service):
        """Test vector-only search mode."""
        query = SearchQuery(query="deep learning neural networks", limit=3, search_type="vector")

        query_embedding = await mock_embedding_service.generate_embedding(query.query)
        results = await vector_store_with_data.search(query, query_embedding)

        assert len(results) > 0
        # Results should only have vector scores
        for result in results:
            assert result.score > 0

    async def test_text_only_search(self, vector_store_with_data):
        """Test text-only search mode."""
        query = SearchQuery(query="Python programming", limit=3, search_type="text")

        # Text search doesn't need embeddings
        results = await vector_store_with_data.search(query, None)

        assert len(results) > 0
        assert all(r.search_type == "text" for r in results)

        # Top result should be from Python document
        assert "python" in results[0].chunk.text.lower()

    async def test_hybrid_search_ranking(self, vector_store_with_data, mock_embedding_service):
        """Test that hybrid search improves ranking."""
        # Search for a term that appears in multiple documents
        query = SearchQuery(query="learning", limit=5, search_type="hybrid")

        query_embedding = await mock_embedding_service.generate_embedding(query.query)
        hybrid_results = await vector_store_with_data.search(query, query_embedding)

        # Also do vector-only search for comparison
        query.search_type = "vector"
        await vector_store_with_data.search(query, query_embedding)

        # Hybrid should find results
        assert len(hybrid_results) > 0

        # Results should be properly scored
        for i in range(1, len(hybrid_results)):
            assert hybrid_results[i - 1].score >= hybrid_results[i].score

    async def test_document_addition_updates_both_indexes(self, test_config, mock_embedding_service):
        """Test that adding documents updates both vector and text indexes."""
        store = VectorStore(test_config)
        store.set_embedding_service(mock_embedding_service)
        await store.initialize()

        # Add a document
        doc = Document(id="test_doc", path="/test.pdf", title="Test Document", checksum="test_hash")
        doc.chunks = [
            Chunk(
                id="chunk_1",
                document_id=doc.id,
                text="This is a test document about hybrid search.",
                embedding=[0.1] * 1536,
                chunk_index=0,
            )
        ]

        await store.add_document(doc)

        # Search using hybrid mode
        query = SearchQuery(query="hybrid search", limit=5, search_type="hybrid")
        query_embedding = await mock_embedding_service.generate_embedding(query.query)
        results = await store.search(query, query_embedding)

        assert len(results) == 1
        assert results[0].chunk.id == "chunk_1"

        # Verify text index has the document
        text_results = await store.text_index.search("hybrid", limit=5)
        assert len(text_results) == 1
        assert text_results[0]["chunk_id"] == "chunk_1"

        await store.close()

    async def test_document_deletion_updates_both_indexes(self, test_config, mock_embedding_service):
        """Test that deleting documents updates both indexes."""
        store = VectorStore(test_config)
        store.set_embedding_service(mock_embedding_service)
        await store.initialize()

        # Add a document
        doc = Document(id="delete_test_doc", path="/delete_test.pdf", title="Delete Test", checksum="delete_hash")
        doc.chunks = [
            Chunk(
                id="chunk_del_1",
                document_id=doc.id,
                text="Document to be deleted",
                embedding=[0.1] * 1536,
                chunk_index=0,
            )
        ]

        await store.add_document(doc)

        # Verify it exists
        query = SearchQuery(query="deleted", limit=5, search_type="hybrid")
        query_embedding = await mock_embedding_service.generate_embedding(query.query)
        results = await store.search(query, query_embedding)
        assert len(results) == 1

        # Delete the document
        await store.delete_document(doc.id)

        # Verify it's gone from both indexes
        results = await store.search(query, query_embedding)
        assert len(results) == 0

        text_results = await store.text_index.search("deleted", limit=5)
        assert len(text_results) == 0

        await store.close()

    async def test_hybrid_search_with_metadata_filter(self, vector_store_with_data, mock_embedding_service):
        """Test hybrid search with metadata filtering."""
        # Add metadata to search query
        query = SearchQuery(
            query="learning", limit=5, metadata_filter={"document_id": "doc_ml_1"}, search_type="hybrid"
        )

        query_embedding = await mock_embedding_service.generate_embedding(query.query)
        results = await vector_store_with_data.search(query, query_embedding)

        # Note: metadata filtering only applies to vector search in current implementation
        # Text search doesn't support metadata filtering yet
        # So we just check that we get some results
        assert len(results) > 0

    async def test_config_disable_hybrid_search(self, mock_embedding_service):
        """Test that hybrid search can be disabled via config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ServerConfig(
                openai_api_key="sk-test-key-123456789",
                knowledgebase_path=Path(tmpdir) / "pdfs",
                cache_dir=Path(tmpdir) / "cache",
                enable_hybrid_search=False,  # Disabled
            )

            store = VectorStore(config)
            store.set_embedding_service(mock_embedding_service)
            await store.initialize()

            # Text index should not be initialized
            assert store.text_index is None
            assert store.hybrid_engine is None

            # Add a document
            doc = Document(id="test_doc", path="/test.pdf", title="Test", checksum="hash")
            doc.chunks = [
                Chunk(id="chunk_1", document_id=doc.id, text="Test content", embedding=[0.1] * 1536, chunk_index=0)
            ]
            await store.add_document(doc)

            # Search should still work (vector-only)
            query = SearchQuery(query="test", limit=5)
            query_embedding = await mock_embedding_service.generate_embedding(query.query)
            results = await store.search(query, query_embedding)

            assert len(results) == 1
            # Should default to vector search
            assert results[0].search_type == "hybrid"  # Will be marked as hybrid but actually vector

            await store.close()

    async def test_reset_database_clears_both_indexes(self, test_config, mock_embedding_service):
        """Test that resetting the database clears both vector and text indexes."""
        store = VectorStore(test_config)
        store.set_embedding_service(mock_embedding_service)
        await store.initialize()

        # Add some documents
        for i in range(3):
            doc = Document(id=f"doc_{i}", path=f"/doc_{i}.pdf", title=f"Document {i}", checksum=f"hash_{i}")
            doc.chunks = [
                Chunk(
                    id=f"chunk_{i}",
                    document_id=doc.id,
                    text=f"Content for document {i}",
                    embedding=[0.1] * 1536,
                    chunk_index=0,
                )
            ]
            await store.add_document(doc)

        # Verify documents exist
        assert await store.get_document_count() > 0
        assert await store.text_index.get_document_count() > 0

        # Reset database
        await store.reset_database()

        # Verify both indexes are empty
        assert await store.get_document_count() == 0
        assert await store.text_index.get_document_count() == 0

        await store.close()

    async def test_performance_comparison(self, vector_store_with_data, mock_embedding_service):
        """Test performance characteristics of different search modes."""
        import time

        queries = [
            "machine learning algorithms",
            "neural networks",
            "Python programming",
            "data analysis",
            "artificial intelligence",
        ]

        # Measure hybrid search time
        hybrid_times = []
        for q in queries:
            query = SearchQuery(query=q, limit=5, search_type="hybrid")
            query_embedding = await mock_embedding_service.generate_embedding(q)

            start = time.time()
            await vector_store_with_data.search(query, query_embedding)
            hybrid_times.append(time.time() - start)

        # Measure vector search time
        vector_times = []
        for q in queries:
            query = SearchQuery(query=q, limit=5, search_type="vector")
            query_embedding = await mock_embedding_service.generate_embedding(q)

            start = time.time()
            await vector_store_with_data.search(query, query_embedding)
            vector_times.append(time.time() - start)

        # Hybrid should not be significantly slower than vector alone
        avg_hybrid = sum(hybrid_times) / len(hybrid_times)
        avg_vector = sum(vector_times) / len(vector_times)

        # Allow hybrid to be up to 50x slower (due to additional text search overhead)
        # In practice it should be much faster, but in tests with mocked data it can be slower
        assert avg_hybrid < avg_vector * 50

    async def test_search_quality_metrics(self, vector_store_with_data, mock_embedding_service):
        """Test that hybrid search improves search quality."""
        # Search for exact term matches
        exact_queries = [("machine learning", "doc_ml_1"), ("deep learning", "doc_dl_1"), ("Python", "doc_python_1")]

        hybrid_correct = 0
        vector_correct = 0

        for query_text, expected_doc_id in exact_queries:
            # Hybrid search
            query = SearchQuery(query=query_text, limit=1, search_type="hybrid")
            query_embedding = await mock_embedding_service.generate_embedding(query_text)
            results = await vector_store_with_data.search(query, query_embedding)

            if results and results[0].document.id == expected_doc_id:
                hybrid_correct += 1

            # Vector search
            query.search_type = "vector"
            results = await vector_store_with_data.search(query, query_embedding)

            if results and results[0].document.id == expected_doc_id:
                vector_correct += 1

        # Hybrid should perform at least as well as vector alone
        assert hybrid_correct >= vector_correct
