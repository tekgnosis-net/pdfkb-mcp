"""Tests for remote summarizer service."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.exceptions import EmbeddingError
from pdfkb.summarizer_remote import RemoteSummarizerService


@pytest.fixture
def test_config():
    """Create test configuration for remote summarizer."""
    config = ServerConfig(
        knowledgebase_path=Path("/tmp/test"),
        openai_api_key="sk-test-openai-key",
        enable_summarizer=True,
        summarizer_provider="remote",
        summarizer_model="gpt-3.5-turbo",
    )
    return config


@pytest.fixture
def remote_summarizer(test_config):
    """Create remote summarizer service instance."""
    return RemoteSummarizerService(test_config)


class TestRemoteSummarizerService:
    """Test remote summarizer service."""

    async def test_initialize(self, remote_summarizer):
        """Test service initialization."""
        with patch("pdfkb.summarizer_remote.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            await remote_summarizer.initialize()

            assert remote_summarizer._initialized
            assert remote_summarizer.api_key == "sk-test-openai-key"
            assert remote_summarizer.client is not None
            mock_openai.assert_called_once_with(api_key="sk-test-openai-key")

    async def test_initialize_with_custom_api_base(self):
        """Test initialization with custom API base URL."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-test-key",
            enable_summarizer=True,
            summarizer_provider="remote",
            summarizer_api_base="https://custom-api.example.com/v1",
        )
        summarizer = RemoteSummarizerService(config)

        with patch("pdfkb.summarizer_remote.AsyncOpenAI") as mock_openai:
            await summarizer.initialize()

            mock_openai.assert_called_once_with(api_key="sk-test-key", base_url="https://custom-api.example.com/v1")

    async def test_initialize_missing_api_key(self):
        """Test initialization fails without API key."""
        config = ServerConfig(
            knowledgebase_path=Path("/tmp/test"),
            openai_api_key="sk-local-embeddings-dummy-key",  # Dummy key
            enable_summarizer=True,
            summarizer_provider="remote",
        )
        summarizer = RemoteSummarizerService(config)

        with pytest.raises(EmbeddingError) as exc_info:
            await summarizer.initialize()
        assert "OpenAI API key required" in str(exc_info.value)

    @patch("pdfkb.summarizer_remote.AsyncOpenAI")
    async def test_summarize_document(self, mock_openai_class, remote_summarizer):
        """Test document summarization."""
        # Mock the OpenAI client and response
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = """
        {
            "title": "Machine Learning Fundamentals",
            "short_description": "An introduction to machine learning concepts and algorithms",
            "long_description": "This document provides a comprehensive overview of machine learning "
            "fundamentals, covering supervised and unsupervised learning, neural networks, "
            "and practical applications in data science."
        }
        """

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Initialize and test
        await remote_summarizer.initialize()

        content = (
            "This is a document about machine learning. "
            "It covers various algorithms and techniques used in data science."
        )
        result = await remote_summarizer.summarize_document(content, "ml_guide.pdf")

        assert result.title == "Machine Learning Fundamentals"
        assert result.short_description == "An introduction to machine learning concepts and algorithms"
        assert "comprehensive overview" in result.long_description

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"  # Default model from test config
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["max_tokens"] == 1024

    @patch("pdfkb.summarizer_remote.AsyncOpenAI")
    async def test_summarize_with_api_error(self, mock_openai_class, remote_summarizer):
        """Test fallback behavior on API error."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        await remote_summarizer.initialize()

        content = "Test document content"
        result = await remote_summarizer.summarize_document(content, "test.pdf")

        # Should return fallback summary
        assert result.title == "test.pdf"
        assert "API error" in result.short_description

    @patch("pdfkb.summarizer_remote.AsyncOpenAI")
    async def test_summarize_invalid_json_response(self, mock_openai_class, remote_summarizer):
        """Test handling of invalid JSON response."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid JSON response from API"

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await remote_summarizer.initialize()

        content = "Test document content"
        result = await remote_summarizer.summarize_document(content, "test.pdf")

        # Should return fallback summary due to parsing error
        assert result.title == "Test"  # Cleaned filename
        assert "Test document content" not in result.short_description  # Fallback summary

    async def test_get_model_info(self, remote_summarizer):
        """Test getting model information."""
        info = remote_summarizer.get_model_info()

        assert info["provider"] == "remote"
        assert info["model"] == "gpt-3.5-turbo"  # Default model from test config
        assert info["api_base"] == "https://api.openai.com/v1"
        assert "description" in info
        assert info["max_pages"] == 10  # Default

    @patch("pdfkb.summarizer_remote.AsyncOpenAI")
    async def test_test_connection_success(self, mock_openai_class, remote_summarizer):
        """Test connection testing success."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = """
        {
            "title": "Test Document",
            "short_description": "A test document",
            "long_description": "This is a test document for connection validation."
        }
        """

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await remote_summarizer.test_connection()
        assert result is True

    @patch("pdfkb.summarizer_remote.RemoteSummarizerService.summarize_document")
    async def test_test_connection_failure(self, mock_summarize, remote_summarizer):
        """Test connection testing failure."""
        mock_summarize.side_effect = Exception("Connection failed")

        result = await remote_summarizer.test_connection()
        assert result is False

    async def test_empty_content_validation(self, remote_summarizer):
        """Test validation of empty content."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            await remote_summarizer.summarize_document("", "test.pdf")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            await remote_summarizer.summarize_document("   ", "test.pdf")
