"""Factory for creating embedding services."""

import logging

from .config import ServerConfig
from .embeddings_base import EmbeddingService
from .embeddings_huggingface import HuggingFaceEmbeddingService
from .embeddings_local import LocalEmbeddingService
from .embeddings_openai import OpenAIEmbeddingService
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def create_embedding_service(config: ServerConfig, fallback: bool = True) -> EmbeddingService:
    """Create an embedding service based on configuration.

    Args:
        config: Server configuration.
        fallback: Whether to allow fallback to OpenAI if local fails.

    Returns:
        EmbeddingService instance.

    Raises:
        ConfigurationError: If unable to create embedding service.
    """
    provider = config.embedding_provider.lower()

    if provider == "local":
        try:
            logger.info("Creating local embedding service")
            return LocalEmbeddingService(config)
        except Exception as e:
            logger.error(f"Failed to create local embedding service: {e}")
            if fallback and config.fallback_to_openai:
                logger.warning("Falling back to OpenAI embedding service")
                return OpenAIEmbeddingService(config)
            else:
                raise ConfigurationError(f"Failed to create local embedding service: {e}")

    elif provider == "openai":
        logger.info("Creating OpenAI embedding service")
        return OpenAIEmbeddingService(config)

    elif provider == "huggingface":
        logger.info("Creating HuggingFace embedding service")
        return HuggingFaceEmbeddingService(config)

    else:
        raise ConfigurationError(f"Unknown embedding provider: {provider}")


def get_embedding_service(config: ServerConfig) -> EmbeddingService:
    """Get or create the embedding service singleton.

    Args:
        config: Server configuration.

    Returns:
        EmbeddingService instance.
    """
    # For now, always create a new instance
    # In the future, we might want to cache this
    return create_embedding_service(config)
