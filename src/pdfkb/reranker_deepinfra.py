"""DeepInfra-based reranking service using Qwen3-Reranker-8B model."""

import logging
from typing import Dict, List, Tuple

import aiohttp

from .config import ServerConfig
from .exceptions import EmbeddingError
from .reranker_base import RerankerService

logger = logging.getLogger(__name__)


class DeepInfraRerankerService(RerankerService):
    """DeepInfra-based reranking service using Qwen3-Reranker models.

    Note: DeepInfra's API requires queries and documents arrays to have the same length.
    As a workaround for search reranking (1 query vs N documents), we duplicate the query
    for each document.
    """

    # DeepInfra model endpoints
    MODEL_ENDPOINTS = {
        "Qwen/Qwen3-Reranker-0.6B": "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-0.6B",
        "Qwen/Qwen3-Reranker-4B": "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B",
        "Qwen/Qwen3-Reranker-8B": "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B",
    }

    # Default model
    DEFAULT_MODEL = "Qwen/Qwen3-Reranker-8B"

    def __init__(self, config: ServerConfig):
        """Initialize the DeepInfra reranker service.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.api_key = config.deepinfra_api_key

        # Select model based on configuration
        self.model_name = getattr(config, "deepinfra_reranker_model", self.DEFAULT_MODEL)
        if self.model_name not in self.MODEL_ENDPOINTS:
            logger.warning(f"Unknown DeepInfra model: {self.model_name}. Using default: {self.DEFAULT_MODEL}")
            self.model_name = self.DEFAULT_MODEL

        self.model_endpoint = self.MODEL_ENDPOINTS[self.model_name]
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the DeepInfra reranker service."""
        if self._initialized:
            return

        if not self.api_key or self.api_key == "sk-local-embeddings-dummy-key":
            raise EmbeddingError(
                "DeepInfra API key required for DeepInfra reranker. " "Set PDFKB_DEEPINFRA_API_KEY",
                self.model_name,
            )

        logger.info(f"DeepInfra reranker service initialized with model: {self.model_name}")
        self._initialized = True

    async def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Rerank documents based on relevance to the query using DeepInfra.

        Args:
            query: The search query.
            documents: List of document texts to rerank.

        Returns:
            List of tuples containing (original_index, relevance_score) sorted by relevance.
        """
        if not documents:
            return []

        if not self._initialized:
            await self.initialize()

        try:
            # DeepInfra expects queries and documents arrays to be the same length
            # We duplicate the query for each document as a workaround
            queries = [query] * len(documents)
            logger.debug(f"Duplicating query {len(documents)} times for DeepInfra API compatibility")
            scores = await self._call_deepinfra(queries, documents)

            # Create list of (index, score) tuples
            results = [(i, score) for i, score in enumerate(scores)]

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Reranked {len(documents)} documents. Top score: {results[0][1]:.4f}")
            return results

        except Exception as e:
            logger.error(f"Failed to rerank with DeepInfra: {e}")
            # Return original order with equal scores as fallback
            return [(i, 1.0) for i in range(len(documents))]

    async def _call_deepinfra(self, queries: List[str], documents: List[str]) -> List[float]:
        """Call DeepInfra API for scoring.

        Args:
            queries: List of queries (must be same length as documents).
            documents: List of documents to score.

        Returns:
            List of scores from the model.
        """
        headers = {
            "Authorization": f"bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "queries": queries,
            "documents": documents,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.model_endpoint, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise EmbeddingError(
                        f"DeepInfra API error: {response.status} - {error_text}",
                        self.model_name,
                    )

                result = await response.json()

                # Extract scores from response
                if "scores" not in result:
                    raise EmbeddingError(
                        f"Unexpected response format from DeepInfra: {result}",
                        self.model_name,
                    )

                scores = result["scores"]

                # Log token usage if available
                if "input_tokens" in result:
                    logger.debug(f"DeepInfra token usage: {result['input_tokens']} input tokens")

                # Ensure scores are floats and in valid range
                scores = [max(0.0, min(1.0, float(s))) for s in scores]

                return scores

    async def test_connection(self) -> bool:
        """Test the DeepInfra reranker service.

        Returns:
            True if service is working, False otherwise.
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Test with a simple query
            test_results = await self.rerank("test query", ["test document"])
            return len(test_results) > 0

        except Exception as e:
            logger.error(f"DeepInfra reranker service test failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get information about the current reranker model.

        Returns:
            Dictionary with model information.
        """
        model_descriptions = {
            "Qwen/Qwen3-Reranker-0.6B": "Lightweight Qwen3-Reranker-0.6B via DeepInfra API",
            "Qwen/Qwen3-Reranker-4B": "High-quality Qwen3-Reranker-4B via DeepInfra API",
            "Qwen/Qwen3-Reranker-8B": "Maximum-quality Qwen3-Reranker-8B via DeepInfra API",
        }

        return {
            "provider": "deepinfra",
            "model": self.model_name,
            "endpoint": self.model_endpoint,
            "description": model_descriptions.get(self.model_name, f"{self.model_name} via DeepInfra API"),
            "capabilities": "Cross-encoder reranking model optimized for relevance scoring",
            "available_models": list(self.MODEL_ENDPOINTS.keys()),
        }
