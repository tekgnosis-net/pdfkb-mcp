"""Tests for reranker factory."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.exceptions import ConfigurationError
from pdfkb.reranker_deepinfra import DeepInfraRerankerService
from pdfkb.reranker_factory import create_reranker_service
from pdfkb.reranker_local import LocalRerankerService


class TestRerankerFactory:
    """Test reranker factory functionality."""

    def test_create_disabled_reranker(self):
        """Test that disabled reranker returns None."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=False,
        )

        service = create_reranker_service(config)
        assert service is None

    def test_create_local_reranker(self):
        """Test creating a local reranker service."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="local",
        )

        service = create_reranker_service(config)
        assert service is not None
        assert isinstance(service, LocalRerankerService)

    def test_create_deepinfra_reranker(self):
        """Test creating a DeepInfra reranker service."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="deepinfra",
            deepinfra_api_key="test-deepinfra-key",
        )

        service = create_reranker_service(config)
        assert service is not None
        assert isinstance(service, DeepInfraRerankerService)

    def test_create_deepinfra_reranker_case_insensitive(self):
        """Test that provider name is case insensitive."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="DeepInfra",  # Mixed case
            deepinfra_api_key="test-deepinfra-key",
        )

        service = create_reranker_service(config)
        assert service is not None
        assert isinstance(service, DeepInfraRerankerService)

    def test_create_unknown_provider(self):
        """Test that unknown provider raises error."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="unknown",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            create_reranker_service(config)

        assert "Unknown reranker provider: unknown" in str(exc_info.value)
        assert "Supported: 'local', 'deepinfra'" in str(exc_info.value)

    @patch("pdfkb.reranker_factory.LocalRerankerService")
    def test_local_reranker_creation_error(self, mock_local_service):
        """Test error handling when local reranker fails to create."""
        mock_local_service.side_effect = Exception("Failed to load model")

        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="local",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            create_reranker_service(config)

        assert "Failed to create local reranker service" in str(exc_info.value)

    @patch("pdfkb.reranker_factory.DeepInfraRerankerService")
    def test_deepinfra_reranker_creation_error(self, mock_deepinfra_service):
        """Test error handling when DeepInfra reranker fails to create."""
        mock_deepinfra_service.side_effect = Exception("Invalid API configuration")

        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_reranker=True,
            reranker_provider="deepinfra",
            deepinfra_api_key="test-key",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            create_reranker_service(config)

        assert "Failed to create DeepInfra reranker service" in str(exc_info.value)
