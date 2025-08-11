"""Compatibility wrapper for embedding services.

This module provides backward compatibility for existing code that imports from embeddings.py.
New code should use embeddings_factory.py directly.
"""

import logging
from typing import Dict, List

from .config import ServerConfig
from .embeddings_base import EmbeddingService as BaseEmbeddingService
from .embeddings_factory import create_embedding_service
from .embeddings_openai import OpenAIEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingService(BaseEmbeddingService):
    """Compatibility wrapper for embedding services.

    This class maintains backward compatibility by automatically selecting
    the appropriate embedding service based on configuration.
    """

    def __init__(self, config: ServerConfig):
        """Initialize the embedding service based on configuration.

        Args:
            config: Server configuration.
        """
        self.config = config
        self._service = create_embedding_service(config)

    async def initialize(self) -> None:
        """Initialize the underlying embedding service."""
        await self._service.initialize()

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return await self._service.generate_embeddings(texts)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        return await self._service.generate_embedding(text)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model.

        Returns:
            Embedding dimension.
        """
        return self._service.get_embedding_dimension()

    async def test_connection(self) -> bool:
        """Test the connection to the embedding service.

        Returns:
            True if connection is successful, False otherwise.
        """
        return await self._service.test_connection()

    def get_model_info(self) -> Dict:
        """Get information about the current embedding model.

        Returns:
            Dictionary with model information.
        """
        return self._service.get_model_info()

    async def estimate_cost(self, texts: List[str]) -> float:
        """Estimate the cost of embedding a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            Estimated cost in USD (0 for local models).
        """
        return await self._service.estimate_cost(texts)


# Re-export for backward compatibility
__all__ = ["EmbeddingService", "OpenAIEmbeddingService", "create_embedding_service"]
