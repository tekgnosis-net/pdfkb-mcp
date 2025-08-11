"""Tests for local embedding service."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.embeddings_factory import create_embedding_service
from pdfkb.embeddings_local import LocalEmbeddingService, LRUCache
from pdfkb.exceptions import ConfigurationError


class TestLRUCache:
    """Test cases for LRU cache."""

    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        cache = LRUCache(maxsize=3)

        # Add items
        cache.put("key1", [1.0, 2.0])
        cache.put("key2", [3.0, 4.0])

        # Get items
        assert cache.get("key1") == [1.0, 2.0]
        assert cache.get("key2") == [3.0, 4.0]
        assert cache.get("key3") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(maxsize=2)

        # Add items
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        cache.put("key3", [3.0])  # Should evict key1

        # key1 should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == [2.0]
        assert cache.get("key3") == [3.0]

    def test_cache_move_to_end(self):
        """Test that accessing an item moves it to end (most recently used)."""
        cache = LRUCache(maxsize=3)

        # Add items
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        cache.put("key3", [3.0])

        # Access key1 to move it to end
        _ = cache.get("key1")

        # Add new item - should evict key2 (least recently used)
        cache.put("key4", [4.0])

        assert cache.get("key1") == [1.0]  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == [3.0]
        assert cache.get("key4") == [4.0]

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(maxsize=3)
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestLocalEmbeddingService:
    """Test cases for local embedding service."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="local",
            local_embedding_model="Qwen/Qwen3-Embedding-0.6B",
            local_embedding_batch_size=2,
            embedding_device="cpu",
            embedding_cache_size=100,
            max_sequence_length=512,
            use_model_optimization=False,
            model_cache_dir="/tmp/test-models",
        )

    def test_service_creation(self, config):
        """Test service creation."""
        service = LocalEmbeddingService(config)
        assert service.model_name == "Qwen/Qwen3-Embedding-0.6B"
        assert service.batch_size == 2
        assert not service._initialized

    def test_device_selection(self, config):
        """Test device selection logic."""
        service = LocalEmbeddingService(config)
        # Device selection happens during initialization
        assert service.device is None  # Not yet initialized

        # Test CPU preference is respected
        assert service._select_device("cpu") == "cpu"

        # Test that auto-detect returns at least CPU
        result = service._select_device(None)
        assert result in ["cpu", "mps", "cuda"]

    @pytest.mark.asyncio
    async def test_initialize(self, config):
        """Test service initialization."""
        service = LocalEmbeddingService(config)

        # Mock the transformers imports that happen inside initialize
        with patch("transformers.AutoModel") as mock_model_cls:
            with patch("transformers.AutoTokenizer") as mock_tokenizer_cls:
                # Mock model and tokenizer
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_model_cls.from_pretrained.return_value = mock_model
                mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = mock_model

                await service.initialize()

                assert service._initialized
                assert service.model is not None
                assert service.tokenizer is not None

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_cache(self, config):
        """Test embedding generation with caching."""
        service = LocalEmbeddingService(config)

        # Mock the synchronous batch generation
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        service._generate_batch_sync = Mock(return_value=mock_embeddings)
        service._initialized = True

        # Generate embeddings
        texts = ["text1", "text2"]
        embeddings = await service.generate_embeddings(texts)

        assert embeddings == mock_embeddings
        service._generate_batch_sync.assert_called_once()

        # Second call should use cache
        embeddings2 = await service.generate_embeddings(texts)
        assert embeddings2 == mock_embeddings

    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, config):
        """Test single embedding generation."""
        service = LocalEmbeddingService(config)

        # Mock batch generation
        mock_embedding = [0.1, 0.2, 0.3]
        service.generate_embeddings = AsyncMock(return_value=[mock_embedding])

        # Generate single embedding
        embedding = await service.generate_embedding("test text")

        assert embedding == mock_embedding
        service.generate_embeddings.assert_called_once_with(["test text"])

    def test_get_embedding_dimension(self, config):
        """Test getting embedding dimension."""
        service = LocalEmbeddingService(config)

        # Test known model
        service.model_name = "Qwen/Qwen3-Embedding-0.6B"
        assert service.get_embedding_dimension() == 1024

        # Test with model config - need to temporarily change model name to unknown
        service.model_name = "unknown-model"
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        service.model = mock_model
        assert service.get_embedding_dimension() == 768

        # Test fallback
        service.model = None
        service.model_name = "unknown-model"
        assert service.get_embedding_dimension() == 768  # Default

    @pytest.mark.asyncio
    async def test_test_connection(self, config):
        """Test connection testing."""
        service = LocalEmbeddingService(config)

        # Mock successful embedding generation
        service.generate_embedding = AsyncMock(return_value=[0.1, 0.2])

        result = await service.test_connection()
        assert result is True

        # Mock failed embedding generation
        service.generate_embedding = AsyncMock(side_effect=Exception("Test error"))
        service._initialized = True  # Skip initialization

        result = await service.test_connection()
        assert result is False

    def test_get_model_info(self, config):
        """Test getting model information."""
        service = LocalEmbeddingService(config)
        service.device = "mps"

        info = service.get_model_info()

        assert info["provider"] == "local"
        assert info["model"] == "Qwen/Qwen3-Embedding-0.6B"
        assert info["dimension"] == 1024
        assert info["max_sequence_length"] == 32000
        assert info["batch_size"] == 2
        assert info["device"] == "mps"
        assert info["cache_size"] == 100

    @pytest.mark.asyncio
    async def test_oom_handling(self, config):
        """Test out-of-memory error handling."""
        service = LocalEmbeddingService(config)
        service._initialized = True

        # Create a mock that fails on first large batch, then succeeds on smaller batches
        call_count = [0]
        batch_sizes_received = []

        def mock_batch_sync(texts):
            call_count[0] += 1
            batch_sizes_received.append(len(texts))
            if call_count[0] == 1 and len(texts) > 1:
                raise RuntimeError("CUDA out of memory")
            return [[0.1] * 384 for _ in texts]

        service._generate_batch_sync = mock_batch_sync

        # Should retry with smaller batch
        texts = ["text1", "text2", "text3", "text4"]
        embeddings = await service.generate_embeddings(texts)

        # Due to the batch reduction, we should get all embeddings but in smaller batches
        assert len(embeddings) >= 2  # At least some embeddings returned
        assert call_count[0] > 1  # Should have retried
        assert 1 in batch_sizes_received  # Should have reduced to batch size 1


class TestEmbeddingFactory:
    """Test cases for embedding factory."""

    @pytest.fixture
    def local_config(self):
        """Create local embedding configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="local",
            local_embedding_model="Qwen/Qwen3-Embedding-0.6B",
        )

    @pytest.fixture
    def openai_config(self):
        """Create OpenAI embedding configuration."""
        return ServerConfig(
            openai_api_key="sk-test-key",
            embedding_provider="openai",
        )

    def test_create_local_service(self, local_config):
        """Test creating local embedding service."""
        with patch("pdfkb.embeddings_factory.LocalEmbeddingService") as mock_local:
            mock_instance = Mock()
            mock_local.return_value = mock_instance

            service = create_embedding_service(local_config)

            assert service == mock_instance
            mock_local.assert_called_once_with(local_config)

    def test_create_openai_service(self, openai_config):
        """Test creating OpenAI embedding service."""
        from pdfkb.embeddings_openai import OpenAIEmbeddingService

        service = create_embedding_service(openai_config)
        assert isinstance(service, OpenAIEmbeddingService)

    def test_fallback_to_openai(self, local_config):
        """Test fallback to OpenAI when local fails."""
        local_config.fallback_to_openai = True

        with patch("pdfkb.embeddings_factory.LocalEmbeddingService") as mock_local:
            mock_local.side_effect = Exception("Local init failed")

            with patch("pdfkb.embeddings_factory.OpenAIEmbeddingService") as mock_openai:
                mock_instance = Mock()
                mock_openai.return_value = mock_instance

                service = create_embedding_service(local_config, fallback=True)

                assert service == mock_instance
                mock_openai.assert_called_once()

    def test_no_fallback_raises_error(self, local_config):
        """Test that error is raised when fallback is disabled."""
        local_config.fallback_to_openai = False

        with patch("pdfkb.embeddings_factory.LocalEmbeddingService") as mock_local:
            mock_local.side_effect = Exception("Local init failed")

            with pytest.raises(ConfigurationError):
                create_embedding_service(local_config, fallback=True)

    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        # ConfigurationError should be raised during config validation,
        # not during factory creation
        with pytest.raises(ConfigurationError) as exc_info:
            ServerConfig(
                openai_api_key="sk-test-key",
                embedding_provider="invalid",
            )

        assert "Invalid embedding_provider" in str(exc_info.value)
