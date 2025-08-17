"""Tests for the semantic chunker implementation."""

from unittest.mock import MagicMock, patch

import pytest

from pdfkb.chunker.chunker_semantic import SemanticChunker
from pdfkb.chunker.langchain_embeddings_wrapper import LangChainEmbeddingsWrapper
from pdfkb.config import ServerConfig
from pdfkb.embeddings_base import EmbeddingService
from pdfkb.models import Chunk


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing."""

    def __init__(self):
        self.initialized = False
        self.embeddings_generated = 0

    async def initialize(self) -> None:
        self.initialized = True

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate fake embeddings based on text length."""
        self.embeddings_generated += len(texts)
        # Create different embeddings based on text content to simulate semantic differences
        embeddings = []
        for text in texts:
            # Simple heuristic: use text length and first char code as embedding features
            length_feature = len(text) / 100.0
            char_feature = ord(text[0]) / 256.0 if text else 0.0
            # Create a 3-dimensional embedding
            embeddings.append([length_feature, char_feature, (length_feature + char_feature) / 2])
        return embeddings

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate fake embedding for a single text."""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        return 3

    async def test_connection(self) -> bool:
        return True

    def get_model_info(self) -> dict:
        return {"model": "mock", "dimension": 3}


@pytest.mark.unit
class TestLangChainEmbeddingsWrapper:
    """Test the LangChain embeddings wrapper."""

    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        mock_service = MockEmbeddingService()
        wrapper = LangChainEmbeddingsWrapper(mock_service)

        assert wrapper.embedding_service == mock_service
        assert not wrapper._initialized

    def test_embed_documents_sync(self):
        """Test synchronous document embedding."""
        mock_service = MockEmbeddingService()
        wrapper = LangChainEmbeddingsWrapper(mock_service)

        texts = ["Hello world", "Test document", "Another text"]
        embeddings = wrapper.embed_documents(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 3 for emb in embeddings)
        assert mock_service.initialized
        assert mock_service.embeddings_generated == 3

    def test_embed_query_sync(self):
        """Test synchronous query embedding."""
        mock_service = MockEmbeddingService()
        wrapper = LangChainEmbeddingsWrapper(mock_service)

        text = "Query text"
        embedding = wrapper.embed_query(text)

        assert len(embedding) == 3
        assert mock_service.initialized
        assert mock_service.embeddings_generated == 1

    @pytest.mark.asyncio
    async def test_embed_documents_async(self):
        """Test asynchronous document embedding."""
        mock_service = MockEmbeddingService()
        wrapper = LangChainEmbeddingsWrapper(mock_service)

        texts = ["Async text 1", "Async text 2"]
        embeddings = await wrapper.aembed_documents(texts)

        assert len(embeddings) == 2
        assert all(len(emb) == 3 for emb in embeddings)
        assert mock_service.initialized

    @pytest.mark.asyncio
    async def test_embed_query_async(self):
        """Test asynchronous query embedding."""
        mock_service = MockEmbeddingService()
        wrapper = LangChainEmbeddingsWrapper(mock_service)

        text = "Async query"
        embedding = await wrapper.aembed_query(text)

        assert len(embedding) == 3
        assert mock_service.initialized

    @pytest.mark.asyncio
    async def test_embed_documents_sync_from_async_context(self):
        """Test synchronous embedding from within an async context (simulates the error scenario)."""
        mock_service = MockEmbeddingService()
        wrapper = LangChainEmbeddingsWrapper(mock_service)

        # This simulates calling sync method from async context (like in background queue)
        texts = ["Text from async context 1", "Text from async context 2"]

        # This should NOT raise an error anymore with our fix
        embeddings = wrapper.embed_documents(texts)

        assert len(embeddings) == 2
        assert all(len(emb) == 3 for emb in embeddings)
        assert mock_service.initialized


@pytest.mark.unit
class TestSemanticChunker:
    """Test the semantic chunker implementation."""

    def test_chunker_initialization_defaults(self):
        """Test chunker initialization with default parameters."""
        mock_service = MockEmbeddingService()
        chunker = SemanticChunker(embedding_service=mock_service)

        assert chunker.embedding_service == mock_service
        assert chunker.breakpoint_threshold_type == "percentile"
        assert chunker.breakpoint_threshold_amount == 95.0
        assert chunker.buffer_size == 1
        assert chunker.number_of_chunks is None
        assert chunker.sentence_split_regex == r"(?<=[.?!])\s+"
        assert chunker.min_chunk_size == 0  # Global filtering, defaults to 0 (disabled)
        assert chunker.langchain_min_chunk_size is None  # LangChain-specific parameter
        assert chunker.min_chunk_chars is None

    def test_chunker_initialization_custom(self):
        """Test chunker initialization with custom parameters."""
        mock_service = MockEmbeddingService()
        chunker = SemanticChunker(
            embedding_service=mock_service,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=3.0,
            buffer_size=2,
            number_of_chunks=5,
            min_chunk_chars=100,
        )

        assert chunker.breakpoint_threshold_type == "standard_deviation"
        assert chunker.breakpoint_threshold_amount == 3.0
        assert chunker.buffer_size == 2
        assert chunker.number_of_chunks == 5
        assert chunker.min_chunk_chars == 100

    def test_chunk_simple_text(self):
        """Test chunking simple text."""
        with patch.object(SemanticChunker, "_get_splitter") as mock_get_splitter:
            # Setup mock
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = [
                "This is the first chunk.",
                "This is the second chunk.",
                "This is the third chunk.",
            ]
            mock_get_splitter.return_value = mock_splitter

            # Create chunker
            mock_service = MockEmbeddingService()
            chunker = SemanticChunker(embedding_service=mock_service)

            # Chunk text
            text = "This is the first chunk. This is the second chunk. This is the third chunk."
            metadata = {"source": "test"}
            chunks = chunker.chunk(text, metadata)

            # Verify results
            assert len(chunks) == 3
            assert all(isinstance(chunk, Chunk) for chunk in chunks)
            assert chunks[0].text == "This is the first chunk."
            assert chunks[1].text == "This is the second chunk."
            assert chunks[2].text == "This is the third chunk."

            # Check metadata
            for chunk in chunks:
                assert chunk.metadata["chunk_strategy"] == "semantic"
                assert chunk.metadata["source"] == "test"
                assert chunk.metadata["breakpoint_threshold_type"] == "percentile"
                assert chunk.metadata["breakpoint_threshold_amount"] == 95.0

    def test_chunk_with_min_size_filter(self):
        """Test chunking with minimum chunk size filter."""
        with patch.object(SemanticChunker, "_get_splitter") as mock_get_splitter:
            # Setup mock
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = [
                "Short",  # Will be filtered out
                "This is a longer chunk that meets the minimum size requirement.",
                "Another long chunk with sufficient characters to pass the filter.",
            ]
            mock_get_splitter.return_value = mock_splitter

            # Create chunker with min_chunk_chars
            mock_service = MockEmbeddingService()
            chunker = SemanticChunker(embedding_service=mock_service, min_chunk_chars=50)

            # Chunk text
            text = (
                "Short. This is a longer chunk that meets the minimum size requirement. "
                "Another long chunk with sufficient characters to pass the filter."
            )
            chunks = chunker.chunk(text, {})

            # Verify only chunks meeting minimum size are returned
            assert len(chunks) == 2
            assert all(len(chunk.text) >= 50 for chunk in chunks)

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        mock_service = MockEmbeddingService()
        chunker = SemanticChunker(embedding_service=mock_service)

        chunks = chunker.chunk("", {})
        assert chunks == []

        chunks = chunker.chunk("   ", {})
        assert chunks == []

    def test_chunk_import_error(self):
        """Test handling of missing langchain-experimental dependency."""
        mock_service = MockEmbeddingService()
        chunker = SemanticChunker(embedding_service=mock_service)

        # Mock the import to raise ImportError
        with patch.object(
            chunker,
            "_get_splitter",
            side_effect=ImportError(
                "LangChain experimental not available. Install with: pip install 'pdfkb-mcp[semantic]'"
            ),
        ):
            with pytest.raises(ImportError) as exc_info:
                chunker.chunk("Test text", {})

            assert "LangChain experimental not available" in str(exc_info.value)
            assert "pip install 'pdfkb-mcp[semantic]'" in str(exc_info.value)


@pytest.mark.unit
class TestServerConfigSemanticChunking:
    """Test ServerConfig validation for semantic chunking."""

    def test_config_semantic_chunker_defaults(self):
        """Test config with semantic chunker and default parameters."""
        config = ServerConfig(openai_api_key="sk-test", document_chunker="semantic")

        assert config.document_chunker == "semantic"
        assert config.semantic_chunker_threshold_type == "percentile"
        assert config.semantic_chunker_threshold_amount == 95.0
        assert config.semantic_chunker_buffer_size == 1
        assert config.semantic_chunker_number_of_chunks is None
        assert config.semantic_chunker_sentence_split_regex == r"(?<=[.?!])\s+"
        assert config.semantic_chunker_min_chunk_size is None
        assert config.semantic_chunker_min_chunk_chars == 100

    def test_config_semantic_chunker_custom(self):
        """Test config with semantic chunker and custom parameters."""
        config = ServerConfig(
            openai_api_key="sk-test",
            document_chunker="semantic",
            semantic_chunker_threshold_type="gradient",
            semantic_chunker_threshold_amount=90.0,
            semantic_chunker_buffer_size=3,
            semantic_chunker_min_chunk_chars=200,
        )

        assert config.semantic_chunker_threshold_type == "gradient"
        assert config.semantic_chunker_threshold_amount == 90.0
        assert config.semantic_chunker_buffer_size == 3
        assert config.semantic_chunker_min_chunk_chars == 200

    def test_config_invalid_threshold_type(self):
        """Test config validation for invalid threshold type."""
        with pytest.raises(Exception) as exc_info:
            ServerConfig(
                openai_api_key="sk-test", document_chunker="semantic", semantic_chunker_threshold_type="invalid"
            )

        assert "semantic_chunker_threshold_type" in str(exc_info.value)

    def test_config_invalid_percentile_amount(self):
        """Test config validation for invalid percentile amount."""
        with pytest.raises(Exception) as exc_info:
            ServerConfig(
                openai_api_key="sk-test",
                document_chunker="semantic",
                semantic_chunker_threshold_type="percentile",
                semantic_chunker_threshold_amount=150.0,  # > 100
            )

        assert "between 0 and 100" in str(exc_info.value)

    def test_config_invalid_std_dev_amount(self):
        """Test config validation for invalid standard deviation amount."""
        with pytest.raises(Exception) as exc_info:
            ServerConfig(
                openai_api_key="sk-test",
                document_chunker="semantic",
                semantic_chunker_threshold_type="standard_deviation",
                semantic_chunker_threshold_amount=-1.0,  # negative
            )

        assert "must be positive" in str(exc_info.value)

    def test_config_invalid_buffer_size(self):
        """Test config validation for invalid buffer size."""
        with pytest.raises(Exception) as exc_info:
            ServerConfig(openai_api_key="sk-test", document_chunker="semantic", semantic_chunker_buffer_size=-1)

        assert "cannot be negative" in str(exc_info.value)

    def test_config_invalid_min_chunk_chars(self):
        """Test config validation for invalid min_chunk_chars."""
        with pytest.raises(Exception) as exc_info:
            ServerConfig(openai_api_key="sk-test", document_chunker="semantic", semantic_chunker_min_chunk_chars=0)

        assert "must be positive" in str(exc_info.value)
