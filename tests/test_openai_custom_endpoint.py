"""Tests for OpenAI custom endpoint support."""

import os
from unittest.mock import Mock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.embeddings_openai import OpenAIEmbeddingService
from pdfkb.exceptions import EmbeddingError


class TestOpenAICustomEndpoint:
    """Test cases for OpenAI custom endpoint functionality."""

    @pytest.fixture
    def config_default(self):
        """Create a test configuration without custom base URL."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_batch_size=2,
        )

    @pytest.fixture
    def config_custom_base(self):
        """Create a test configuration with custom base URL."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_batch_size=2,
            openai_api_base="https://api.studio.nebius.com/v1/",
        )

    @pytest.mark.asyncio
    async def test_default_openai_endpoint(self, config_default):
        """Test that OpenAI service works with default endpoint."""
        service = OpenAIEmbeddingService(config_default)

        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            await service.initialize()

            # Should be called with only api_key (no base_url)
            mock_client_class.assert_called_once_with(api_key="sk-test-key")
            assert service.client is not None

    @pytest.mark.asyncio
    async def test_custom_openai_endpoint(self, config_custom_base):
        """Test that OpenAI service works with custom endpoint."""
        service = OpenAIEmbeddingService(config_custom_base)

        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            await service.initialize()

            # Should be called with both api_key and base_url
            mock_client_class.assert_called_once_with(
                api_key="sk-test-key", base_url="https://api.studio.nebius.com/v1/"
            )
            assert service.client is not None

    @pytest.mark.asyncio
    async def test_custom_endpoint_logging(self, config_custom_base, caplog):
        """Test that custom endpoint usage is logged."""
        # Set the logger level to INFO to capture the log message
        import logging

        caplog.set_level(logging.INFO, logger="pdfkb.embeddings_openai")

        service = OpenAIEmbeddingService(config_custom_base)

        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            await service.initialize()

            # Check that the custom base URL is logged
            assert "Using custom OpenAI API base URL: https://api.studio.nebius.com/v1/" in caplog.text

    @pytest.mark.asyncio
    async def test_embeddings_with_custom_endpoint(self, config_custom_base):
        """Test generating embeddings with custom endpoint."""
        # Use a non-test API key to avoid mock behavior
        config_custom_base.openai_api_key = "sk-real-api-key"
        service = OpenAIEmbeddingService(config_custom_base)

        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = Mock()
            mock_embedding_response = Mock()
            mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

            # Make embeddings.create properly awaitable
            async def mock_create(*args, **kwargs):
                return mock_embedding_response

            mock_client.embeddings.create = mock_create
            mock_client_class.return_value = mock_client

            embedding = await service.generate_embedding("test text")

            assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_connection_test_with_custom_endpoint(self, config_custom_base):
        """Test connection test with custom endpoint."""
        service = OpenAIEmbeddingService(config_custom_base)

        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = Mock()
            mock_embedding_response = Mock()
            mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_embedding_response
            mock_client_class.return_value = mock_client

            result = await service.test_connection()

            assert result is True

    @pytest.mark.asyncio
    async def test_custom_endpoint_with_test_key(self, config_custom_base):
        """Test that custom endpoint works with test keys."""
        # Use test key that triggers mock behavior
        config_custom_base.openai_api_key = "sk-test-mock-key"
        service = OpenAIEmbeddingService(config_custom_base)

        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = Mock()
            mock_client.embeddings.create.side_effect = Exception("Test exception")
            mock_client_class.return_value = mock_client

            # Should return mock embeddings when using test key
            embedding = await service.generate_embedding("test text")

            assert len(embedding) == service.get_embedding_dimension()
            assert all(isinstance(x, float) for x in embedding)

    def test_get_model_info_with_custom_endpoint(self, config_custom_base):
        """Test that model info includes custom endpoint information."""
        service = OpenAIEmbeddingService(config_custom_base)

        info = service.get_model_info()

        assert info["provider"] == "openai"
        assert info["model"] == "text-embedding-3-small"
        assert info["dimension"] == 1536
        assert info["batch_size"] == 2

    @pytest.mark.asyncio
    async def test_env_var_support(self):
        """Test that PDFKB_OPENAI_API_BASE environment variable is supported."""
        from pdfkb.config import ServerConfig

        # Test with environment variable
        with patch.dict(
            os.environ,
            {
                "PDFKB_OPENAI_API_BASE": "https://custom.endpoint.com/v1",
                "PDFKB_OPENAI_API_KEY": "sk-test-key",
                "PDFKB_EMBEDDING_PROVIDER": "openai",
            },
        ):
            config = ServerConfig.from_env()

            assert config.openai_api_base == "https://custom.endpoint.com/v1"

    @pytest.mark.asyncio
    async def test_no_custom_endpoint_no_logging(self, config_default, caplog):
        """Test that no custom endpoint logging occurs when using default."""
        service = OpenAIEmbeddingService(config_default)

        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            await service.initialize()

            # Check that no custom base URL logging occurs
            assert "Using custom OpenAI API base URL" not in caplog.text

    @pytest.mark.asyncio
    async def test_error_handling_with_custom_endpoint(self, config_custom_base):
        """Test error handling when custom endpoint fails."""
        service = OpenAIEmbeddingService(config_custom_base)

        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            with pytest.raises(EmbeddingError, match="Failed to initialize OpenAI embedding service"):
                await service.initialize()

    @pytest.mark.asyncio
    async def test_various_custom_endpoints(self):
        """Test various custom endpoint formats."""
        test_endpoints = [
            "https://api.studio.nebius.com/v1/",
            "http://localhost:8080/v1",
            "https://api.openrouter.ai/api/v1",
            "https://api.together.xyz/v1",
        ]

        for endpoint in test_endpoints:
            config = ServerConfig(
                openai_api_key="sk-test-key",
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
                openai_api_base=endpoint,
            )
            service = OpenAIEmbeddingService(config)

            with patch("openai.AsyncOpenAI") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                await service.initialize()

                mock_client_class.assert_called_once_with(api_key="sk-test-key", base_url=endpoint)

                # Reset for next iteration
                mock_client_class.reset_mock()
