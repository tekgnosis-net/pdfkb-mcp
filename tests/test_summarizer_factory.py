"""Tests for summarizer factory."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.exceptions import ConfigurationError
from pdfkb.summarizer_factory import create_summarizer_service
from pdfkb.summarizer_local import LocalSummarizerService
from pdfkb.summarizer_remote import RemoteSummarizerService


class TestSummarizerFactory:
    """Test summarizer factory functionality."""

    def test_create_disabled_summarizer(self):
        """Test that disabled summarizer returns None."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_summarizer=False,
        )

        service = create_summarizer_service(config)
        assert service is None

    def test_create_local_summarizer(self):
        """Test creating a local summarizer service."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_summarizer=True,
            summarizer_provider="local",
        )

        service = create_summarizer_service(config)
        assert service is not None
        assert isinstance(service, LocalSummarizerService)

    def test_create_remote_summarizer(self):
        """Test creating a remote summarizer service."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-test-key",
            enable_summarizer=True,
            summarizer_provider="remote",
        )

        service = create_summarizer_service(config)
        assert service is not None
        assert isinstance(service, RemoteSummarizerService)

    def test_create_openai_summarizer(self):
        """Test creating an OpenAI summarizer service (alias for remote)."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-test-key",
            enable_summarizer=True,
            summarizer_provider="openai",
        )

        service = create_summarizer_service(config)
        assert service is not None
        assert isinstance(service, RemoteSummarizerService)

    def test_create_summarizer_case_insensitive(self):
        """Test that provider name is case insensitive."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-test-key",
            enable_summarizer=True,
            summarizer_provider="REMOTE",  # Mixed case
        )

        service = create_summarizer_service(config)
        assert service is not None
        assert isinstance(service, RemoteSummarizerService)

    def test_create_unknown_provider(self):
        """Test that unknown provider raises error."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-test-key",
            enable_summarizer=True,
            summarizer_provider="unknown",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            create_summarizer_service(config)

        assert "Unknown summarizer provider: unknown" in str(exc_info.value)
        assert "Supported: 'local', 'remote', 'openai'" in str(exc_info.value)

    @patch("pdfkb.summarizer_local.LocalSummarizerService")
    def test_local_summarizer_creation_error(self, mock_local_service):
        """Test error handling when local summarizer fails to create."""
        mock_local_service.side_effect = Exception("Failed to load model")

        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",
            enable_summarizer=True,
            summarizer_provider="local",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            create_summarizer_service(config)

        assert "Failed to create local summarizer service" in str(exc_info.value)

    @patch("pdfkb.summarizer_remote.RemoteSummarizerService")
    def test_remote_summarizer_creation_error(self, mock_remote_service):
        """Test error handling when remote summarizer fails to create."""
        mock_remote_service.side_effect = Exception("Invalid API configuration")

        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-test-key",
            enable_summarizer=True,
            summarizer_provider="remote",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            create_summarizer_service(config)

        assert "Failed to create remote summarizer service" in str(exc_info.value)
