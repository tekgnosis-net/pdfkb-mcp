"""Factory for creating reranker services."""

import logging
from typing import Optional

from .config import ServerConfig
from .exceptions import ConfigurationError
from .reranker_base import RerankerService
from .reranker_deepinfra import DeepInfraRerankerService
from .reranker_local import LocalRerankerService

logger = logging.getLogger(__name__)


def create_reranker_service(config: ServerConfig) -> Optional[RerankerService]:
    """Create a reranker service based on configuration.

    Args:
        config: Server configuration.

    Returns:
        RerankerService instance or None if reranker is disabled.

    Raises:
        ConfigurationError: If unable to create reranker service when enabled.
    """
    if not config.enable_reranker:
        logger.info("Reranker is disabled")
        return None

    provider = config.reranker_provider.lower()

    if provider == "deepinfra":
        try:
            logger.info("Creating DeepInfra reranker service")
            return DeepInfraRerankerService(config)
        except Exception as e:
            logger.error(f"Failed to create DeepInfra reranker service: {e}")
            raise ConfigurationError(f"Failed to create DeepInfra reranker service: {e}")
    elif provider == "local":
        try:
            logger.info("Creating local reranker service")
            return LocalRerankerService(config)
        except Exception as e:
            logger.error(f"Failed to create local reranker service: {e}")
            raise ConfigurationError(f"Failed to create local reranker service: {e}")
    else:
        raise ConfigurationError(f"Unknown reranker provider: {provider}. Supported: 'local', 'deepinfra'")


def get_reranker_service(config: ServerConfig) -> Optional[RerankerService]:
    """Get or create the reranker service singleton.

    Args:
        config: Server configuration.

    Returns:
        RerankerService instance or None if disabled.
    """
    # For now, always create a new instance
    # In the future, we might want to cache this
    return create_reranker_service(config)
