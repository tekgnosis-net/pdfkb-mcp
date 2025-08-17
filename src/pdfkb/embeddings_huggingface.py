"""HuggingFace embedding service for generating text embeddings."""

import asyncio
import logging
import os
from typing import Dict, List

from .config import ServerConfig
from .embeddings_base import EmbeddingService
from .exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingService(EmbeddingService):
    """Handles HuggingFace embedding generation using InferenceClient."""

    def __init__(self, config: ServerConfig):
        """Initialize the HuggingFace embedding service.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.client = None
        self.model = config.huggingface_embedding_model
        self.provider = config.huggingface_provider
        self.api_key = config.huggingface_api_key or os.getenv("HF_TOKEN")
        self.batch_size = config.embedding_batch_size

        # Cache for embedding dimension
        self._embedding_dimension = None

        # Validate configuration
        if not self.model:
            raise ConfigurationError("huggingface_embedding_model is required for HuggingFace embeddings")

        if not self.api_key:
            raise ConfigurationError("HF_TOKEN environment variable or huggingface_api_key is required")

    async def initialize(self) -> None:
        """Initialize the HuggingFace client."""
        try:
            from huggingface_hub import InferenceClient

            # Create the client with provider and API key
            client_kwargs = {
                "api_key": self.api_key,
            }

            if self.provider:
                client_kwargs["provider"] = self.provider

            self.client = InferenceClient(**client_kwargs)

            # Test the connection and get embedding dimension
            test_text = "test"
            test_embedding = await self._generate_single_embedding(test_text)
            self._embedding_dimension = len(test_embedding)

            logger.info(
                f"HuggingFace embedding service initialized with model: {self.model}, "
                f"provider: {self.provider or 'default'}, dimension: {self._embedding_dimension}"
            )

        except ImportError:
            raise EmbeddingError(
                "huggingface_hub package not installed. Install with: pip install huggingface_hub", self.model
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize HuggingFace embedding service: {e}", self.model, e)

    async def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        loop = asyncio.get_running_loop()

        # Run the synchronous feature_extraction in a thread pool
        embedding = await loop.run_in_executor(None, lambda: self.client.feature_extraction(text, model=self.model))

        # The result might be nested, extract the actual embedding
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                # It's a batch result, take the first one
                return embedding[0]
            else:
                # It's already a single embedding
                return embedding
        else:
            raise EmbeddingError(f"Unexpected embedding format from HuggingFace: {type(embedding)}", self.model)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if self.client is None:
            await self.initialize()

        try:
            all_embeddings = []

            # Process texts in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # Generate embeddings for the batch
                batch_embeddings = []
                for text in batch:
                    embedding = await self._generate_single_embedding(text)
                    batch_embeddings.append(embedding)

                all_embeddings.extend(batch_embeddings)

                # Small delay between batches to avoid rate limits
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)

            logger.info(f"Generated embeddings for {len(texts)} texts using HuggingFace")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}", self.model, e)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        if self.client is None:
            await self.initialize()

        try:
            return await self._generate_single_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}", self.model, e)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model.

        Returns:
            Embedding dimension.
        """
        if self._embedding_dimension is None:
            # Common dimensions for popular models
            model_dimensions = {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "BAAI/bge-small-en-v1.5": 384,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-large-en-v1.5": 1024,
                "Qwen/Qwen3-Embedding-8B": 1024,  # Example from user
                "intfloat/e5-base-v2": 768,
                "intfloat/e5-large-v2": 1024,
            }

            # Check if we know this model
            for known_model, dim in model_dimensions.items():
                if known_model in self.model:
                    self._embedding_dimension = dim
                    break

            if self._embedding_dimension is None:
                # Default to 768 which is common for BERT-based models
                logger.warning(
                    f"Unknown embedding dimension for model {self.model}, defaulting to 768. "
                    "Will be updated after first embedding."
                )
                self._embedding_dimension = 768

        return self._embedding_dimension

    async def test_connection(self) -> bool:
        """Test the connection to the HuggingFace service.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            if self.client is None:
                await self.initialize()

            # Try to generate a test embedding
            test_text = "connection test"
            embedding = await self.generate_embedding(test_text)

            return len(embedding) > 0

        except Exception as e:
            logger.error(f"HuggingFace connection test failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get information about the current embedding model.

        Returns:
            Dictionary with model information.
        """
        return {
            "provider": "huggingface",
            "model": self.model,
            "dimension": self.get_embedding_dimension(),
            "batch_size": self.batch_size,
            "huggingface_provider": self.provider or "default",
        }

    async def estimate_cost(self, texts: List[str]) -> float:
        """Estimate the cost of embedding a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            Estimated cost in USD.
        """
        # HuggingFace pricing varies by provider and model
        # This is a rough estimate - actual costs depend on the specific provider
        # Some providers may offer free tiers or different pricing models

        if self.provider == "nebius":
            # Example pricing for Nebius (adjust based on actual pricing)
            # Assuming $0.0001 per 1K tokens (rough estimate)
            total_chars = sum(len(text) for text in texts)
            estimated_tokens = total_chars / 4  # Rough char to token ratio
            estimated_cost = (estimated_tokens / 1000) * 0.0001
            return estimated_cost
        else:
            # For other providers or default HuggingFace inference API
            # Many models on HuggingFace are free for moderate usage
            return 0.0
