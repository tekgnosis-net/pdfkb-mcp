"""Tests for DeepInfra reranker service."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.exceptions import EmbeddingError
from pdfkb.reranker_deepinfra import DeepInfraRerankerService


@pytest.fixture
def test_config():
    """Create test configuration for DeepInfra reranker."""
    config = ServerConfig(
        knowledgebase_path=Path("/tmp/test"),
        openai_api_key="sk-local-embeddings-dummy-key",
        enable_reranker=True,
        reranker_provider="deepinfra",
        deepinfra_api_key="test-deepinfra-key",
    )
    return config


@pytest.fixture
def deepinfra_reranker(test_config):
    """Create DeepInfra reranker service instance."""
    return DeepInfraRerankerService(test_config)


class TestDeepInfraRerankerService:
    """Test DeepInfra reranker service."""

    async def test_initialize(self, deepinfra_reranker):
        """Test service initialization."""
        await deepinfra_reranker.initialize()
        assert deepinfra_reranker._initialized
        assert deepinfra_reranker.api_key == "test-deepinfra-key"

    async def test_initialize_missing_api_key(self):
        """Test initialization fails without API key."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            deepinfra_api_key="",  # Empty key
            enable_reranker=True,
            reranker_provider="deepinfra",
        )
        reranker = DeepInfraRerankerService(config)

        with pytest.raises(EmbeddingError) as exc_info:
            await reranker.initialize()
        assert "DeepInfra API key required" in str(exc_info.value)

    async def test_initialize_with_dummy_key(self):
        """Test initialization fails with dummy key."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            deepinfra_api_key="sk-local-embeddings-dummy-key",  # Dummy key
            enable_reranker=True,
            reranker_provider="deepinfra",
        )
        reranker = DeepInfraRerankerService(config)

        with pytest.raises(EmbeddingError) as exc_info:
            await reranker.initialize()
        assert "DeepInfra API key required" in str(exc_info.value)

    @patch("aiohttp.ClientSession")
    async def test_rerank(self, mock_session_class, deepinfra_reranker):
        """Test document reranking."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "scores": [0.9, 0.3, 0.7],
                "input_tokens": 42,
                "inference_status": {
                    "status": "success",
                    "runtime_ms": 100,
                    "cost": 0.001,
                },
            }
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        # Test reranking
        documents = [
            "Machine learning is a subset of AI",
            "Database optimization techniques",
            "Neural networks and deep learning",
        ]

        results = await deepinfra_reranker.rerank("machine learning", documents)

        # Check results are sorted by score
        assert len(results) == 3
        assert results[0] == (0, 0.9)  # Highest score
        assert results[1] == (2, 0.7)
        assert results[2] == (1, 0.3)  # Lowest score

        # Verify API was called correctly
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == deepinfra_reranker.model_endpoint
        # Check that query was duplicated for each document
        assert call_args[1]["json"]["queries"] == ["machine learning"] * 3
        assert call_args[1]["json"]["documents"] == documents

    @patch("aiohttp.ClientSession")
    async def test_rerank_api_error(self, mock_session_class, deepinfra_reranker):
        """Test fallback behavior on API error."""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        results = await deepinfra_reranker.rerank("test query", documents)

        # Should return fallback equal scores
        assert len(results) == 3
        assert all(score == 1.0 for _, score in results)

    @patch("aiohttp.ClientSession")
    async def test_rerank_missing_scores(self, mock_session_class, deepinfra_reranker):
        """Test error handling when response missing scores."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "input_tokens": 42,
                # Missing "scores" field
            }
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        results = await deepinfra_reranker.rerank("test query", documents)

        # Should return fallback equal scores due to missing scores field
        assert len(results) == 3
        assert all(score == 1.0 for _, score in results)

    @patch("aiohttp.ClientSession")
    async def test_rerank_with_token_logging(self, mock_session_class, deepinfra_reranker, caplog):
        """Test that token usage is logged when available."""
        import logging

        caplog.set_level(logging.DEBUG)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "scores": [0.5, 0.5],
                "input_tokens": 100,
            }
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        await deepinfra_reranker.rerank("test", ["doc1", "doc2"])

        # Check that token usage was logged
        assert "DeepInfra token usage: 100 input tokens" in caplog.text

    async def test_get_model_info(self, deepinfra_reranker):
        """Test getting model information."""
        info = deepinfra_reranker.get_model_info()

        assert info["provider"] == "deepinfra"
        assert info["model"] == "Qwen/Qwen3-Reranker-8B"
        assert info["endpoint"] == deepinfra_reranker.model_endpoint
        assert "description" in info
        assert "capabilities" in info
        assert "available_models" in info
        assert len(info["available_models"]) == 3

    @patch("aiohttp.ClientSession")
    async def test_test_connection_success(self, mock_session_class, deepinfra_reranker):
        """Test connection testing success."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "scores": [0.5],
                "input_tokens": 10,
            }
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await deepinfra_reranker.test_connection()
        assert result is True

    @patch("pdfkb.reranker_deepinfra.DeepInfraRerankerService.rerank")
    async def test_test_connection_failure(self, mock_rerank, deepinfra_reranker):
        """Test connection testing failure."""
        # Make rerank raise an exception
        mock_rerank.side_effect = Exception("Connection failed")

        result = await deepinfra_reranker.test_connection()
        assert result is False

    async def test_rerank_empty_documents(self, deepinfra_reranker):
        """Test reranking with empty document list."""
        results = await deepinfra_reranker.rerank("test query", [])
        assert results == []

    @patch("aiohttp.ClientSession")
    async def test_score_normalization(self, mock_session_class, deepinfra_reranker):
        """Test that scores are normalized to [0, 1] range."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "scores": [1.5, -0.3, 0.7],  # Out of range scores
                "input_tokens": 42,
            }
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        results = await deepinfra_reranker.rerank("test query", documents)

        # Check that scores are clamped to [0, 1]
        assert len(results) == 3
        assert results[0] == (0, 1.0)  # 1.5 clamped to 1.0
        assert results[1] == (2, 0.7)
        assert results[2] == (1, 0.0)  # -0.3 clamped to 0.0

    async def test_model_0_6b(self):
        """Test using the 0.6B model."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="deepinfra",
            deepinfra_api_key="test-deepinfra-key",
            deepinfra_reranker_model="Qwen/Qwen3-Reranker-0.6B",
        )
        reranker = DeepInfraRerankerService(config)

        await reranker.initialize()
        assert reranker.model_name == "Qwen/Qwen3-Reranker-0.6B"
        assert reranker.model_endpoint == DeepInfraRerankerService.MODEL_ENDPOINTS["Qwen/Qwen3-Reranker-0.6B"]

        info = reranker.get_model_info()
        assert info["model"] == "Qwen/Qwen3-Reranker-0.6B"
        assert "0.6B" in info["description"]

    async def test_model_4b(self):
        """Test using the 4B model."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="deepinfra",
            deepinfra_api_key="test-deepinfra-key",
            deepinfra_reranker_model="Qwen/Qwen3-Reranker-4B",
        )
        reranker = DeepInfraRerankerService(config)

        await reranker.initialize()
        assert reranker.model_name == "Qwen/Qwen3-Reranker-4B"
        assert reranker.model_endpoint == DeepInfraRerankerService.MODEL_ENDPOINTS["Qwen/Qwen3-Reranker-4B"]

        info = reranker.get_model_info()
        assert info["model"] == "Qwen/Qwen3-Reranker-4B"
        assert "4B" in info["description"]

    async def test_invalid_model_fallback(self):
        """Test that invalid model falls back to default."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="deepinfra",
            deepinfra_api_key="test-deepinfra-key",
            deepinfra_reranker_model="Qwen/Invalid-Model",
        )
        reranker = DeepInfraRerankerService(config)

        # Should fall back to default model
        assert reranker.model_name == DeepInfraRerankerService.DEFAULT_MODEL
        assert (
            reranker.model_endpoint == DeepInfraRerankerService.MODEL_ENDPOINTS[DeepInfraRerankerService.DEFAULT_MODEL]
        )
