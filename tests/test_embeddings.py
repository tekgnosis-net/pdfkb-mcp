"""Tests for the embeddings module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            embedding_model="text-embedding-3-small",
            embedding_batch_size=10,
        )

    @pytest.fixture
    def embedding_service(self, config):
        """Create an EmbeddingService instance."""
        return EmbeddingService(config)

    @pytest.mark.asyncio
    async def test_initialize_embedding_service(self, embedding_service):
        """Test initializing the embedding service."""
        # Avoid real OpenAI client initialization by patching initialize internals
        # Patch importlib to fake presence of openai and AsyncOpenAI client
        import importlib
        import types

        fake_openai = types.SimpleNamespace(AsyncOpenAI=Mock())
        with patch.object(
            importlib,
            "import_module",
            side_effect=lambda name, package=None: (
                fake_openai if name == "openai" else importlib.__dict__["import_module"](name, package)
            ),
        ):
            await embedding_service.initialize()
            assert embedding_service.model == "text-embedding-3-small"
            assert embedding_service.batch_size == 10

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self, embedding_service):
        """Test generating embeddings for empty list."""
        embeddings = await embedding_service.generate_embeddings([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_single_text(self, embedding_service):
        """Test generating embeddings for single text."""
        texts = ["This is a test text."]
        # Patch batch generation to return fixed vectors to avoid network and dependency
        with patch.object(
            embedding_service,
            "_generate_batch_embeddings",
            new=AsyncMock(return_value=[[0.0] * 1536]),
        ):
            embeddings = await embedding_service.generate_embeddings(texts)
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1536  # text-embedding-3-small dimension

    @pytest.mark.asyncio
    async def test_generate_embedding_single(self, embedding_service):
        """Test generating embedding for single text."""
        text = "Single test text"
        with patch.object(
            embedding_service,
            "_generate_batch_embeddings",
            new=AsyncMock(return_value=[[0.0] * 1536]),
        ):
            embedding = await embedding_service.generate_embedding(text)
            assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_generate_embeddings_multiple_batches(self, embedding_service):
        """Test generating embeddings with multiple batches."""
        # Create more texts than batch size
        texts = [f"Test text {i}" for i in range(25)]
        with patch.object(
            embedding_service,
            "_generate_batch_embeddings",
            new=AsyncMock(side_effect=lambda batch: [[0.0] * 1536 for _ in batch]),
        ):
            embeddings = await embedding_service.generate_embeddings(texts)
            assert len(embeddings) == 25
            assert all(len(emb) == 1536 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_test_connection_success(self, embedding_service):
        """Test successful connection test."""
        # Ensure generate_embedding returns a non-empty vector by mocking batch generation
        with patch.object(
            embedding_service,
            "_generate_batch_embeddings",
            new=AsyncMock(return_value=[[0.0] * 1536]),
        ):
            result = await embedding_service.test_connection()
            assert result is True  # Implementation returns True on success

    def test_get_embedding_dimension(self, embedding_service):
        """Test getting embedding dimension."""
        dimension = embedding_service.get_embedding_dimension()
        assert dimension == 1536

    def test_get_embedding_dimension_different_model(self):
        """Test getting embedding dimension for different model."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            embedding_model="text-embedding-3-large",
        )
        service = EmbeddingService(config)
        dimension = service.get_embedding_dimension()
        assert dimension == 3072

    @pytest.mark.asyncio
    async def test_estimate_cost(self, embedding_service):
        """Test cost estimation for embeddings."""
        texts = ["Short text", "This is a longer text with more words"]
        # Patch embedding dimension lookup to avoid relying on external state
        with patch.object(embedding_service, "get_embedding_dimension", return_value=1536):
            cost = await embedding_service.estimate_cost(texts)
            assert isinstance(cost, float)
            assert cost >= 0

    def test_get_model_info(self, embedding_service):
        """Test getting model information."""
        info = embedding_service.get_model_info()

        assert info["model"] == "text-embedding-3-small"
        assert info["dimension"] == 1536
        assert info["batch_size"] == 10
        assert "rate_limit_interval" in info

    def test_extract_retry_after_with_match(self, embedding_service):
        """Test extracting retry-after from error message."""
        error_msg = "Rate limit exceeded. Please retry after 60 seconds."
        retry_after = embedding_service._extract_retry_after(error_msg)
        assert retry_after == 60

    def test_extract_retry_after_no_match(self, embedding_service):
        """Test extracting retry-after when no match found."""
        error_msg = "Some other error message"
        retry_after = embedding_service._extract_retry_after(error_msg)
        assert retry_after is None

    # TODO: Add more comprehensive tests when real implementation is added
    # - Test actual OpenAI API calls
    # - Test rate limiting behavior
    # - Test error handling for various API errors
    # - Test batch processing with real API responses
