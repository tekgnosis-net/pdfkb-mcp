"""Factory for creating summarization service instances."""

import logging
from typing import Optional

from .config import ServerConfig
from .exceptions import ConfigurationError
from .summarizer_base import SummarizerService

logger = logging.getLogger(__name__)


def create_summarizer_service(config: ServerConfig) -> Optional[SummarizerService]:
    """Create a summarization service based on configuration.

    Args:
        config: Server configuration.

    Returns:
        SummarizerService instance or None if summarization is disabled.

    Raises:
        ConfigurationError: If the provider is unknown or service creation fails.
    """
    if not config.enable_summarizer:
        logger.info("Summarization is disabled")
        return None

    provider = config.summarizer_provider.lower()
    logger.info(f"Creating summarizer service with provider: {provider}")

    try:
        if provider == "local":
            from .summarizer_local import LocalSummarizerService

            return LocalSummarizerService(config)

        elif provider == "remote" or provider == "openai":
            from .summarizer_remote import RemoteSummarizerService

            return RemoteSummarizerService(config)

        else:
            supported_providers = ["local", "remote", "openai"]
            raise ConfigurationError(
                f"Unknown summarizer provider: {provider}. "
                f"Supported: {', '.join(repr(p) for p in supported_providers)}"
            )

    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise

        provider_type = "local" if provider == "local" else "remote"
        raise ConfigurationError(f"Failed to create {provider_type} summarizer service: {e}")
