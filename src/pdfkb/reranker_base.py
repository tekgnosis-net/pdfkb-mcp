"""Base interface for reranking services."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class RerankerService(ABC):
    """Abstract base class for reranking services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the reranker service."""
        pass

    @abstractmethod
    async def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.

        Returns:
            List of tuples containing (original_index, relevance_score) sorted by relevance.
            Higher scores indicate higher relevance.
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the reranker service is working.

        Returns:
            True if service is working, False otherwise.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get information about the current reranker model.

        Returns:
            Dictionary with model information.
        """
        pass
