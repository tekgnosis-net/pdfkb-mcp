"""Remote LLM-based document summarization service using OpenAI-compatible APIs."""

import json
import logging
import re
from typing import Dict

from .config import ServerConfig
from .exceptions import EmbeddingError
from .summarizer_base import DocumentSummary, SummarizerService

logger = logging.getLogger(__name__)


class RemoteSummarizerService(SummarizerService):
    """Remote summarization service using OpenAI-compatible APIs."""

    def __init__(self, config: ServerConfig):
        """Initialize the remote summarizer service.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.api_key = config.summarizer_api_key or config.openai_api_key
        self.api_base = config.summarizer_api_base or config.openai_api_base
        self.model_name = config.summarizer_model
        self.max_pages = config.summarizer_max_pages
        self._initialized = False

        # OpenAI client will be initialized in initialize()
        self.client = None

    async def initialize(self) -> None:
        """Initialize the remote summarizer service."""
        if self._initialized:
            return

        if not self.api_key or self.api_key == "sk-local-embeddings-dummy-key":
            raise EmbeddingError(
                "OpenAI API key required for remote summarizer. Set PDFKB_OPENAI_API_KEY or PDFKB_SUMMARIZER_API_KEY",
                self.model_name,
            )

        try:
            from openai import AsyncOpenAI

            # Initialize OpenAI client with custom base URL if provided
            client_kwargs = {"api_key": self.api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
                logger.info(f"Using custom API base: {self.api_base}")

            self.client = AsyncOpenAI(**client_kwargs)

            logger.info(f"Remote summarizer service initialized with model: {self.model_name}")
            self._initialized = True

        except ImportError as e:
            raise EmbeddingError(
                f"OpenAI package not installed. Install with: pip install openai: {e}",
                self.model_name,
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize remote summarizer service: {e}", self.model_name, e)

    def _create_summarization_prompt(self, content: str, filename: str = "") -> str:
        """Create a comprehensive prompt for document summarization.

        Args:
            content: Document content to summarize.
            filename: Optional filename for context.

        Returns:
            Formatted prompt for the LLM.
        """
        filename_context = f" The document filename is: {filename}." if filename else ""

        return (
            f"You are an expert document analyst. Your task is to analyze the provided document "
            f"and create a comprehensive summary with three components: a title, a short description, "
            f"and a long description.{filename_context}\n\n"
            f"Please analyze the following document content and provide:\n\n"
            f"1. **Title**: A clear, descriptive title that captures the main subject/purpose (max 80 characters)\n"
            f"2. **Short Description**: A concise 1-2 sentence summary highlighting "
            f"the key topic and purpose (max 200 characters)\n"
            f"3. **Long Description**: A detailed paragraph explaining the document's content, "
            f"key points, methodology, findings, or conclusions (max 500 characters)\n\n"
            f"**Important**: Return your response as a valid JSON object with exactly these keys: "
            f'"title", "short_description", "long_description". Do not include any other text outside the JSON.\n\n'
            f"Document content:\n{content}"
        )

    def _truncate_content(self, content: str, max_tokens: int = 30000) -> str:
        """Truncate content to fit within the API's context window.

        Args:
            content: Original document content.
            max_tokens: Maximum tokens to allow for content.

        Returns:
            Truncated content that fits within the model's context.
        """
        # Rough estimation: 4 characters per token
        max_content_chars = max_tokens * 4

        if len(content) <= max_content_chars:
            return content

        # Truncate and add indicator
        truncated = content[:max_content_chars]
        # Try to cut at a sentence boundary
        last_period = truncated.rfind(".")
        if last_period > max_content_chars * 0.8:  # If we can find a period in the last 20%
            truncated = truncated[: last_period + 1]

        return truncated + "\n\n[Content truncated due to length...]"

    async def summarize_document(self, content: str, filename: str = "") -> DocumentSummary:
        """Summarize a document using the remote LLM.

        Args:
            content: The document content to summarize.
            filename: Optional filename for context.

        Returns:
            DocumentSummary with title, short description, and long description.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        if not self._initialized:
            await self.initialize()

        try:
            # Truncate content if necessary
            truncated_content = self._truncate_content(content)

            # Create prompt
            prompt = self._create_summarization_prompt(truncated_content, filename)

            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert document analyst. Always respond with valid JSON "
                            "containing title, short_description, and long_description keys."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            # Extract and parse response
            response_content = response.choices[0].message.content

            try:
                return self._parse_summary_response(response_content, filename)
            except Exception as e:
                logger.warning(f"Failed to parse API response, using fallback: {e}")
                return self._create_fallback_summary(content, filename)

        except Exception as e:
            logger.error(f"Failed to summarize document with remote API: {e}")
            # Return a basic fallback summary
            title = filename if filename else "Document"
            return DocumentSummary(
                title=title,
                short_description="Document summary unavailable due to API error",
                long_description=(
                    "This document could not be automatically summarized due to an API error. "
                    f"Original content length: {len(content)} characters."
                ),
            )

    def _parse_summary_response(self, response: str, filename: str = "") -> DocumentSummary:
        """Parse the API's JSON response into a DocumentSummary.

        Args:
            response: The API's response string.
            filename: Optional filename for fallback.

        Returns:
            Parsed DocumentSummary.
        """
        # Clean the response - extract JSON if wrapped in other text
        response = response.strip()

        # Try to find JSON object
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response

        try:
            parsed = json.loads(json_str)

            # Validate required fields
            title = parsed.get("title", "").strip()
            short_desc = parsed.get("short_description", "").strip()
            long_desc = parsed.get("long_description", "").strip()

            if not title or not short_desc or not long_desc:
                raise ValueError("Missing required fields in response")

            # Truncate if necessary
            title = title[:80] if len(title) > 80 else title
            short_desc = short_desc[:200] if len(short_desc) > 200 else short_desc
            long_desc = long_desc[:500] if len(long_desc) > 500 else long_desc

            return DocumentSummary(
                title=title,
                short_description=short_desc,
                long_description=long_desc,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            raise

    def _create_fallback_summary(self, content: str, filename: str = "") -> DocumentSummary:
        """Create a basic fallback summary when API parsing fails.

        Args:
            content: Original document content.
            filename: Optional filename.

        Returns:
            Basic DocumentSummary.
        """
        # Use filename as title if available, otherwise generic
        title = filename.replace(".pdf", "").replace("_", " ").title() if filename else "Document"

        # Create basic descriptions
        word_count = len(content.split())
        char_count = len(content)

        short_desc = f"Document with {word_count} words"
        long_desc = (
            f"This document contains {word_count} words and {char_count} characters. "
            f"Automatic summarization was not available, but the document appears to contain "
            f"structured content suitable for analysis."
        )

        return DocumentSummary(
            title=title,
            short_description=short_desc,
            long_description=long_desc,
        )

    async def test_connection(self) -> bool:
        """Test the remote summarizer service.

        Returns:
            True if service is working, False otherwise.
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Test with simple content
            test_summary = await self.summarize_document(
                "This is a test document for validating the summarization service.", "test.pdf"
            )
            return test_summary.title and test_summary.short_description and test_summary.long_description

        except Exception as e:
            logger.error(f"Remote summarizer service test failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get information about the current summarizer model.

        Returns:
            Dictionary with model information.
        """
        return {
            "provider": "remote",
            "model": self.model_name,
            "api_base": self.api_base or "https://api.openai.com/v1",
            "description": "Remote LLM via OpenAI-compatible API",
            "max_pages": self.max_pages,
        }
