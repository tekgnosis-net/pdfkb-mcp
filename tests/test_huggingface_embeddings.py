"""Tests for HuggingFace embedding service."""

import os
from unittest.mock import Mock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.embeddings_huggingface import HuggingFaceEmbeddingService
from pdfkb.exceptions import ConfigurationError, EmbeddingError


class TestHuggingFaceEmbeddingService:
    """Test cases for HuggingFace embedding service."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="huggingface",
            huggingface_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            huggingface_provider=None,
            huggingface_api_key="hf_test_token",
            embedding_batch_size=2,
        )

    @pytest.fixture
    def config_with_provider(self):
        """Create a test configuration with specific provider."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="huggingface",
            huggingface_embedding_model="Qwen/Qwen3-Embedding-8B",
            huggingface_provider="nebius",
            huggingface_api_key="hf_test_token",
            embedding_batch_size=2,
        )

    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        # Ensure HF_TOKEN is not set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="HF_TOKEN"):
                ServerConfig(
                    openai_api_key="sk-test-key",
                    embedding_provider="huggingface",
                    huggingface_embedding_model="test-model",
                    huggingface_api_key=None,
                )

    def test_initialization_with_env_token(self):
        """Test that initialization works with HF_TOKEN environment variable."""
        # Set HF_TOKEN in environment
        with patch.dict(os.environ, {"HF_TOKEN": "hf_env_token"}):
            config = ServerConfig(
                openai_api_key="sk-test-key",
                embedding_provider="huggingface",
                huggingface_embedding_model="test-model",
                huggingface_api_key=None,
            )
            service = HuggingFaceEmbeddingService(config)
            assert service.api_key == "hf_env_token"

    def test_initialization_without_model(self):
        """Test that initialization fails without model."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="huggingface",
            huggingface_embedding_model="",
            huggingface_api_key="hf_test_token",
        )

        with pytest.raises(ConfigurationError, match="huggingface_embedding_model"):
            HuggingFaceEmbeddingService(config)

    @pytest.mark.asyncio
    async def test_initialize_client(self, config):
        """Test client initialization."""
        service = HuggingFaceEmbeddingService(config)

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            mock_client.feature_extraction = Mock(return_value=[0.1, 0.2, 0.3])
            mock_client_class.return_value = mock_client

            await service.initialize()

            assert service.client is not None
            mock_client_class.assert_called_once_with(api_key="hf_test_token")
            assert service._embedding_dimension == 3

    @pytest.mark.asyncio
    async def test_initialize_client_with_provider(self, config_with_provider):
        """Test client initialization with specific provider."""
        service = HuggingFaceEmbeddingService(config_with_provider)

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            mock_client.feature_extraction = Mock(return_value=[0.1] * 1024)
            mock_client_class.return_value = mock_client

            await service.initialize()

            assert service.client is not None
            mock_client_class.assert_called_once_with(api_key="hf_test_token", provider="nebius")
            assert service._embedding_dimension == 1024

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, config):
        """Test generating embeddings for multiple texts."""
        service = HuggingFaceEmbeddingService(config)

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            # Mock feature_extraction to return different embeddings for each text
            # Include one for initialization test and 3 for actual texts
            mock_client.feature_extraction = Mock(
                side_effect=[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            )
            mock_client_class.return_value = mock_client

            texts = ["text1", "text2", "text3"]
            embeddings = await service.generate_embeddings(texts)

            assert len(embeddings) == 3
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
            assert embeddings[2] == [0.7, 0.8, 0.9]
            assert mock_client.feature_extraction.call_count == 4  # 1 for init + 3 for texts

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_processing(self, config):
        """Test that embeddings are processed in batches."""
        service = HuggingFaceEmbeddingService(config)
        service.batch_size = 2  # Small batch size for testing

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            mock_client.feature_extraction = Mock(side_effect=[[0.1] * 3] * 6)  # 1 for init + 5 for texts
            mock_client_class.return_value = mock_client

            texts = ["text1", "text2", "text3", "text4", "text5"]
            embeddings = await service.generate_embeddings(texts)

            assert len(embeddings) == 5
            # Should have been called: 1 for init + 5 for individual texts (processed in batches)
            assert mock_client.feature_extraction.call_count == 6

    @pytest.mark.asyncio
    async def test_generate_embedding_single(self, config):
        """Test generating embedding for a single text."""
        service = HuggingFaceEmbeddingService(config)

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            mock_client.feature_extraction = Mock(side_effect=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            mock_client_class.return_value = mock_client

            embedding = await service.generate_embedding("test text")

            assert embedding == [0.4, 0.5, 0.6]
            assert mock_client.feature_extraction.call_count == 2  # 1 for init + 1 for text

    @pytest.mark.asyncio
    async def test_generate_embeddings_nested_result(self, config):
        """Test handling nested embedding results."""
        service = HuggingFaceEmbeddingService(config)

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            # Return nested list (batch result format)
            mock_client.feature_extraction = Mock(return_value=[[0.1, 0.2, 0.3]])
            mock_client_class.return_value = mock_client

            await service.initialize()

            # Should extract the inner list
            assert service._embedding_dimension == 3

    def test_get_embedding_dimension(self, config):
        """Test getting embedding dimension."""
        service = HuggingFaceEmbeddingService(config)

        # Should return default for known model
        assert service.get_embedding_dimension() == 384  # all-MiniLM-L6-v2 dimension

        # Test with initialized dimension
        service._embedding_dimension = 768
        assert service.get_embedding_dimension() == 768

    def test_get_embedding_dimension_unknown_model(self):
        """Test getting embedding dimension for unknown model."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="huggingface",
            huggingface_embedding_model="unknown/model",
            huggingface_api_key="hf_test_token",
        )

        service = HuggingFaceEmbeddingService(config)

        # Should return default 768 for unknown model
        assert service.get_embedding_dimension() == 768

    @pytest.mark.asyncio
    async def test_test_connection_success(self, config):
        """Test successful connection test."""
        service = HuggingFaceEmbeddingService(config)

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            mock_client.feature_extraction = Mock(return_value=[0.1, 0.2, 0.3])
            mock_client_class.return_value = mock_client

            result = await service.test_connection()

            assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, config):
        """Test failed connection test."""
        service = HuggingFaceEmbeddingService(config)

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            mock_client.feature_extraction = Mock(side_effect=Exception("Connection failed"))
            mock_client_class.return_value = mock_client

            result = await service.test_connection()

            assert result is False

    def test_get_model_info(self, config):
        """Test getting model information."""
        service = HuggingFaceEmbeddingService(config)
        service._embedding_dimension = 384

        info = service.get_model_info()

        assert info["provider"] == "huggingface"
        assert info["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert info["dimension"] == 384
        assert info["batch_size"] == 2
        assert info["huggingface_provider"] == "default"

    def test_get_model_info_with_provider(self, config_with_provider):
        """Test getting model information with specific provider."""
        service = HuggingFaceEmbeddingService(config_with_provider)

        info = service.get_model_info()

        assert info["huggingface_provider"] == "nebius"

    @pytest.mark.asyncio
    async def test_estimate_cost_nebius(self, config_with_provider):
        """Test cost estimation for Nebius provider."""
        service = HuggingFaceEmbeddingService(config_with_provider)

        texts = ["text1" * 100, "text2" * 100, "text3" * 100]  # Long texts
        cost = await service.estimate_cost(texts)

        assert cost > 0  # Should have some cost for Nebius

    @pytest.mark.asyncio
    async def test_estimate_cost_default(self, config):
        """Test cost estimation for default provider."""
        service = HuggingFaceEmbeddingService(config)

        texts = ["text1", "text2", "text3"]
        cost = await service.estimate_cost(texts)

        assert cost == 0.0  # Should be free for default HuggingFace

    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test error handling in embedding generation."""
        service = HuggingFaceEmbeddingService(config)

        with patch("huggingface_hub.InferenceClient") as mock_client_class:
            mock_client = Mock()
            mock_client.feature_extraction = Mock(
                side_effect=[[0.1, 0.2, 0.3], Exception("API error")]  # Success for init  # Error for actual call
            )
            mock_client_class.return_value = mock_client

            with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
                await service.generate_embedding("test text")
