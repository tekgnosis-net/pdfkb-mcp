"""Unit tests for HybridSearchEngine class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pdfkb.config import ServerConfig
from pdfkb.hybrid_search import HybridSearchEngine
from pdfkb.models import Chunk, Document, SearchQuery, SearchResult
from pdfkb.text_index import TextIndex


@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = MagicMock(spec=ServerConfig)
    config.enable_hybrid_search = True
    config.hybrid_search_weights = {"vector": 0.6, "text": 0.4}
    config.rrf_k = 60
    config.whoosh_min_score = 0.0
    config.hybrid_expansion_factor = 2.0
    config.hybrid_max_expanded_limit = 30
    return config


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store._vector_search = AsyncMock()
    return store


@pytest.fixture
def mock_text_index():
    """Create a mock text index."""
    index = MagicMock(spec=TextIndex)
    index.search = AsyncMock()
    return index


@pytest.fixture
def hybrid_engine(mock_vector_store, mock_text_index, test_config):
    """Create a HybridSearchEngine instance."""
    return HybridSearchEngine(mock_vector_store, mock_text_index, test_config)


@pytest.fixture
def sample_vector_results():
    """Create sample vector search results."""
    results = []
    for i in range(3):
        chunk = Chunk(
            id=f"chunk_vec_{i}", document_id=f"doc_{i}", text=f"Vector result {i}", page_number=i + 1, chunk_index=i
        )
        doc = Document(id=f"doc_{i}", path=f"/path/doc_{i}.pdf", title=f"Document {i}")
        result = SearchResult(
            chunk=chunk, score=0.9 - (i * 0.1), document=doc, search_type="vector"  # Decreasing scores
        )
        results.append(result)
    return results


@pytest.fixture
def sample_text_results():
    """Create sample text search results."""
    results = []
    for i in range(3):
        results.append(
            {
                "chunk_id": f"chunk_text_{i}",
                "document_id": f"doc_{i}",
                "text": f"Text result {i}",
                "score": 10.0 - i,  # BM25 scores
                "page_number": i + 1,
                "chunk_index": i,
                "metadata": {},
            }
        )
    return results


@pytest.fixture
def overlapping_results():
    """Create overlapping results between vector and text search."""
    vector_results = []
    text_results = []

    # Create some overlapping chunks
    for i in range(3):
        chunk = Chunk(
            id=f"chunk_common_{i}", document_id=f"doc_{i}", text=f"Common result {i}", page_number=i + 1, chunk_index=i
        )
        doc = Document(id=f"doc_{i}", path=f"/path/doc_{i}.pdf", title=f"Document {i}")
        vector_results.append(SearchResult(chunk=chunk, score=0.9 - (i * 0.1), document=doc, search_type="vector"))

        text_results.append(
            {
                "chunk_id": f"chunk_common_{i}",
                "document_id": f"doc_{i}",
                "text": f"Common result {i}",
                "score": 10.0 - i,
                "page_number": i + 1,
                "chunk_index": i,
                "metadata": {},
            }
        )

    # Add unique results to each
    unique_chunk = Chunk(
        id="chunk_vec_unique", document_id="doc_unique_v", text="Unique vector result", page_number=1, chunk_index=0
    )
    unique_doc = Document(id="doc_unique_v", path="/path/unique_v.pdf", title="Unique Vector Doc")
    vector_results.append(SearchResult(chunk=unique_chunk, score=0.85, document=unique_doc, search_type="vector"))

    text_results.append(
        {
            "chunk_id": "chunk_text_unique",
            "document_id": "doc_unique_t",
            "text": "Unique text result",
            "score": 8.5,
            "page_number": 1,
            "chunk_index": 0,
            "metadata": {},
        }
    )

    return vector_results, text_results


@pytest.mark.asyncio
@pytest.mark.unit
class TestHybridSearchEngine:
    """Test suite for HybridSearchEngine class."""

    async def test_initialization(self, hybrid_engine, mock_vector_store, mock_text_index, test_config):
        """Test HybridSearchEngine initialization."""
        assert hybrid_engine.vector_store == mock_vector_store
        assert hybrid_engine.text_index == mock_text_index
        assert hybrid_engine.config == test_config

    async def test_search_basic(
        self, hybrid_engine, mock_vector_store, mock_text_index, sample_vector_results, sample_text_results
    ):
        """Test basic hybrid search functionality."""
        # Setup mocks
        mock_vector_store._vector_search.return_value = sample_vector_results
        mock_text_index.search.return_value = sample_text_results

        # Create query
        query = SearchQuery(query="test query", limit=5)
        query_embedding = [0.1] * 768

        # Perform search
        results = await hybrid_engine.search(query, query_embedding)

        # Verify both searches were called
        mock_vector_store._vector_search.assert_called_once()
        mock_text_index.search.assert_called_once()

        # Check results
        assert len(results) <= query.limit
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.search_type == "hybrid" for r in results)

    async def test_search_parallel_execution(self, hybrid_engine, mock_vector_store, mock_text_index):
        """Test that vector and text searches execute in parallel."""

        # Add delays to simulate real searches
        async def delayed_vector_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return []

        async def delayed_text_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return []

        mock_vector_store._vector_search = delayed_vector_search
        mock_text_index.search = delayed_text_search

        query = SearchQuery(query="test", limit=5)
        query_embedding = [0.1] * 768

        # Time the search - should be ~0.1s (parallel) not ~0.2s (sequential)
        import time

        start = time.time()
        await hybrid_engine.search(query, query_embedding)
        elapsed = time.time() - start

        assert elapsed < 0.15  # Allow some overhead

    async def test_rrf_fusion_basic(self, hybrid_engine, overlapping_results):
        """Test RRF fusion with overlapping results."""
        vector_results, text_results = overlapping_results

        # Apply RRF fusion
        fused = hybrid_engine._apply_rrf(vector_results, text_results, limit=5)

        # Check basic properties
        assert len(fused) <= 5
        assert all(isinstance(r, SearchResult) for r in fused)

        # Check scores are in descending order
        for i in range(1, len(fused)):
            assert fused[i - 1].score >= fused[i].score

        # Check that overlapping chunks have higher scores (appear first)
        chunk_ids = [r.chunk.id for r in fused]
        # Common chunks should rank higher due to appearing in both result sets
        assert "chunk_common_0" in chunk_ids[:3]

    async def test_rrf_formula(self, test_config):
        """Test RRF score calculation formula."""
        # Create engine with known config
        test_config.rrf_k = 60
        test_config.hybrid_search_weights = {"vector": 0.5, "text": 0.5}

        engine = HybridSearchEngine(MagicMock(), MagicMock(), test_config)

        # Create simple results
        vector_results = [
            SearchResult(
                chunk=Chunk(id="chunk_1", document_id="doc_1", text="Text 1"),
                score=0.9,
                document=Document(id="doc_1", path="/doc1.pdf"),
            )
        ]

        text_results = [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "text": "Text 1",
                "score": 10.0,
                "page_number": 1,
                "chunk_index": 0,
                "metadata": {},
            }
        ]

        fused = engine._apply_rrf(vector_results, text_results, limit=1)

        # Calculate expected RRF score
        # chunk_1 is rank 1 in both lists
        expected_score = 0.5 * (1 / (60 + 1)) + 0.5 * (1 / (60 + 1))

        assert len(fused) == 1
        assert abs(fused[0].score - expected_score) < 0.0001

    async def test_rrf_weights(self, test_config):
        """Test RRF with different weight configurations."""
        # Test vector-heavy weighting
        test_config.hybrid_search_weights = {"vector": 0.8, "text": 0.2}
        engine = HybridSearchEngine(MagicMock(), MagicMock(), test_config)

        vector_results = [
            SearchResult(
                chunk=Chunk(id="chunk_vec", document_id="doc_1", text="Vector"),
                score=0.9,
                document=Document(id="doc_1", path="/doc1.pdf"),
            )
        ]

        text_results = [
            {
                "chunk_id": "chunk_text",
                "document_id": "doc_2",
                "text": "Text",
                "score": 10.0,
                "page_number": 1,
                "chunk_index": 0,
                "metadata": {},
            }
        ]

        fused = engine._apply_rrf(vector_results, text_results, limit=2)

        # Vector result should rank higher due to weight
        assert fused[0].chunk.id == "chunk_vec"

    async def test_rrf_deduplication(self, hybrid_engine):
        """Test that RRF properly deduplicates results."""
        # Create duplicate chunks with different scores
        vector_results = [
            SearchResult(
                chunk=Chunk(id="chunk_dup", document_id="doc_1", text="Duplicate"),
                score=0.9,
                document=Document(id="doc_1", path="/doc1.pdf"),
            )
        ]

        text_results = [
            {
                "chunk_id": "chunk_dup",
                "document_id": "doc_1",
                "text": "Duplicate",
                "score": 8.0,
                "page_number": 1,
                "chunk_index": 0,
                "metadata": {},
            }
        ]

        fused = hybrid_engine._apply_rrf(vector_results, text_results, limit=10)

        # Should only have one instance of the chunk
        assert len(fused) == 1
        assert fused[0].chunk.id == "chunk_dup"
        # Should have both scores recorded
        assert fused[0].vector_score == 0.9
        assert fused[0].text_score == 8.0

    async def test_search_fallback_on_error(
        self, hybrid_engine, mock_vector_store, mock_text_index, sample_vector_results
    ):
        """Test fallback to vector search on error."""
        # Make text search fail
        mock_text_index.search.side_effect = Exception("Text index error")
        mock_vector_store._vector_search.return_value = sample_vector_results

        query = SearchQuery(query="test", limit=5)
        query_embedding = [0.1] * 768

        # Should fall back to vector search
        results = await hybrid_engine.search(query, query_embedding)

        assert len(results) == len(sample_vector_results)
        # Results should be from vector search
        assert all(r.chunk.id.startswith("chunk_vec") for r in results)

    async def test_empty_results(self, hybrid_engine, mock_vector_store, mock_text_index):
        """Test handling of empty result sets."""
        mock_vector_store._vector_search.return_value = []
        mock_text_index.search.return_value = []

        query = SearchQuery(query="no results", limit=5)
        query_embedding = [0.1] * 768

        results = await hybrid_engine.search(query, query_embedding)

        assert results == []

    async def test_vector_only_results(self, hybrid_engine, mock_vector_store, mock_text_index, sample_vector_results):
        """Test when only vector search returns results."""
        mock_vector_store._vector_search.return_value = sample_vector_results
        mock_text_index.search.return_value = []

        query = SearchQuery(query="test", limit=5)
        query_embedding = [0.1] * 768

        results = await hybrid_engine.search(query, query_embedding)

        assert len(results) > 0
        assert all(r.vector_score is not None for r in results)
        assert all(r.text_score is None for r in results)

    async def test_text_only_results(self, hybrid_engine, mock_vector_store, mock_text_index, sample_text_results):
        """Test when only text search returns results."""
        mock_vector_store._vector_search.return_value = []
        mock_text_index.search.return_value = sample_text_results

        query = SearchQuery(query="test", limit=5)
        query_embedding = [0.1] * 768

        results = await hybrid_engine.search(query, query_embedding)

        assert len(results) > 0
        assert all(r.text_score is not None for r in results)
        assert all(r.vector_score is None for r in results)

    async def test_expanded_limit(self, hybrid_engine, mock_vector_store, mock_text_index, test_config):
        """Test that searches use expanded limit for better fusion."""
        query = SearchQuery(query="test", limit=5)
        query_embedding = [0.1] * 768

        mock_vector_store._vector_search.return_value = []
        mock_text_index.search.return_value = []

        await hybrid_engine.search(query, query_embedding)

        # Check that expanded limit was used (configured expansion factor)
        expansion_factor = test_config.hybrid_expansion_factor
        max_limit = test_config.hybrid_max_expanded_limit
        expected_limit = min(int(5 * expansion_factor), max_limit)

        # Get the actual calls
        vector_call = mock_vector_store._vector_search.call_args
        text_call = mock_text_index.search.call_args

        # Vector search: first arg is SearchQuery, check its limit
        assert vector_call[0][0].limit == expected_limit

        # Text search: second argument is limit
        assert text_call[0][1] == expected_limit

    async def test_result_metadata_preservation(self, hybrid_engine):
        """Test that metadata is preserved through RRF fusion."""
        vector_results = [
            SearchResult(
                chunk=Chunk(id="chunk_1", document_id="doc_1", text="Text", metadata={"source": "vector"}),
                score=0.9,
                document=Document(id="doc_1", path="/doc1.pdf", metadata={"type": "research"}),
            )
        ]

        text_results = []

        fused = hybrid_engine._apply_rrf(vector_results, text_results, limit=1)

        assert len(fused) == 1
        assert fused[0].chunk.metadata["source"] == "vector"
        assert fused[0].document.metadata["type"] == "research"
