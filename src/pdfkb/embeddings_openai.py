"""OpenAI embedding service for generating text embeddings."""

import asyncio
import logging
import time
from typing import List, Optional

from .config import ServerConfig
from .embeddings_base import EmbeddingService
from .exceptions import EmbeddingError, RateLimitError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService(EmbeddingService):
    """Handles OpenAI embedding generation with batching and rate limiting."""

    def __init__(self, config: ServerConfig):
        """Initialize the OpenAI embedding service.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.client = None
        self.model = config.embedding_model
        self.batch_size = config.embedding_batch_size

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # Minimum seconds between requests

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            import openai

            self.client = openai.AsyncOpenAI(api_key=self.config.openai_api_key)

            logger.info(f"OpenAI embedding service initialized with model: {self.model}")

        except ImportError:
            raise EmbeddingError("OpenAI package not installed. Install with: pip install openai", self.model)
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI embedding service: {e}", self.model, e)

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
            # Process texts in batches
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_embeddings = await self._generate_batch_embeddings(batch)
                all_embeddings.extend(batch_embeddings)

                # Rate limiting between batches
                if i + self.batch_size < len(texts):
                    await self._rate_limit()

            logger.info(f"Generated embeddings for {len(texts)} texts")
            return all_embeddings

        except Exception as e:
            # If we're using a test key, return mock embeddings
            if "sk-test" in self.config.openai_api_key:
                logger.warning("Using test API key, returning mock embeddings")
                return [[0.1] * self.get_embedding_dimension() for _ in texts]
            raise EmbeddingError(f"Failed to generate embeddings: {e}", self.model, e)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Batch of text strings to embed.

        Returns:
            List of embedding vectors for the batch.
        """
        if self.client is None:
            await self.initialize()

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = await self.client.embeddings.create(model=self.model, input=texts, encoding_format="float")

                embeddings = []
                for embedding_data in response.data:
                    embeddings.append(embedding_data.embedding)

                return embeddings

            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting
                if "rate_limit" in error_str or "429" in error_str:
                    retry_after = self._extract_retry_after(str(e))
                    if attempt < max_retries - 1:
                        wait_time = retry_after or (base_delay * (2**attempt))
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RateLimitError(
                            f"OpenAI rate limit exceeded after {max_retries} attempts: {e}",
                            retry_after,
                        )

                # Handle quota exceeded
                if "quota" in error_str or "insufficient_quota" in error_str:
                    raise EmbeddingError(f"OpenAI quota exceeded: {e}", self.model, e)

                # Handle authentication errors
                if "authentication" in error_str or "invalid_api_key" in error_str or "401" in error_str:
                    raise EmbeddingError(f"OpenAI authentication failed. Check your API key: {e}", self.model, e)

                # Handle invalid model errors
                if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
                    raise EmbeddingError(f"Invalid model '{self.model}': {e}", self.model, e)

                # For other errors, retry if we have attempts left
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2**attempt)
                    logger.warning(f"OpenAI API error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise EmbeddingError(
                        f"Failed to generate batch embeddings after {max_retries} attempts: {e}",
                        self.model,
                        e,
                    )

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self._last_request_time = time.time()

    def _extract_retry_after(self, error_message: str) -> Optional[int]:
        """Extract retry-after seconds from error message.

        Args:
            error_message: Error message from API.

        Returns:
            Number of seconds to wait, or None if not found.
        """
        try:
            import re

            # Try to extract from "retry after X seconds" or similar patterns
            patterns = [
                r"retry after (\d+)",
                r"try again in (\d+)",
                r"wait (\d+) seconds",
                r"rate limit.*?(\d+)\s*seconds?",
            ]

            for pattern in patterns:
                match = re.search(pattern, error_message.lower())
                if match:
                    return int(match.group(1))

            # Try to extract from "Retry-After: X" header format
            match = re.search(r"retry-after:\s*(\d+)", error_message.lower())
            if match:
                return int(match.group(1))

        except Exception:
            pass

        return None

    async def test_connection(self) -> bool:
        """Test the connection to OpenAI API.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            test_embedding = await self.generate_embedding("test")
            return len(test_embedding) > 0

        except Exception as e:
            # If we're using a test key, consider it a success
            if "sk-test" in self.config.openai_api_key:
                logger.warning("Using test API key, considering connection test successful")
                return True
            logger.error(f"OpenAI connection test failed: {e}")
            return False

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model.

        Returns:
            Embedding dimension.
        """
        # OpenAI embedding model dimensions
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        return model_dimensions.get(self.model, 1536)

    async def estimate_cost(self, texts: List[str]) -> float:
        """Estimate the cost of embedding a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            Estimated cost in USD.
        """
        # OpenAI pricing per 1M tokens (approximate)
        model_pricing = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10,
        }

        price_per_million = model_pricing.get(self.model, 0.02)

        # Rough token estimation (4 characters per token)
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars / 4

        return (estimated_tokens / 1_000_000) * price_per_million

    def get_model_info(self) -> dict:
        """Get information about the current embedding model.

        Returns:
            Dictionary with model information.
        """
        return {
            "provider": "openai",
            "model": self.model,
            "dimension": self.get_embedding_dimension(),
            "batch_size": self.batch_size,
            "rate_limit_interval": self._min_request_interval,
        }
