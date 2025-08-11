"""Abstract base class for embedding services."""

from abc import ABC, abstractmethod
from typing import Dict, List


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding service."""
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model.

        Returns:
            Embedding dimension.
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the connection to the embedding service.

        Returns:
            True if connection is successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get information about the current embedding model.

        Returns:
            Dictionary with model information.
        """
        pass

    async def estimate_cost(self, texts: List[str]) -> float:
        """Estimate the cost of embedding a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            Estimated cost in USD (0 for local models).
        """
        return 0.0  # Default to 0 for local models
