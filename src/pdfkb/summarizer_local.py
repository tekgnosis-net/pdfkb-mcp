"""Local LLM-based document summarization service using transformers."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict

from .config import ServerConfig
from .exceptions import EmbeddingError
from .summarizer_base import DocumentSummary, SummarizerService

logger = logging.getLogger(__name__)


class LocalSummarizerService(SummarizerService):
    """Local summarization service using transformers LLMs."""

    # Supported models with their specifications
    MODEL_SPECS = {
        "Qwen/Qwen3-4B-Instruct-2507-FP8": {
            "size_gb": 8.0,
            "description": "Qwen3 4B Instruct model with FP8 quantization (default)",
            "context_length": 32768,
        },
        "Qwen/Qwen3-8B-Instruct": {
            "size_gb": 16.0,
            "description": "Qwen3 8B Instruct model",
            "context_length": 32768,
        },
        "Qwen/Qwen3-1.5B-Instruct": {
            "size_gb": 3.0,
            "description": "Qwen3 1.5B Instruct model (lightweight)",
            "context_length": 32768,
        },
        "microsoft/Phi-3-mini-4k-instruct": {
            "size_gb": 7.0,
            "description": "Microsoft Phi-3 Mini 4K Instruct",
            "context_length": 4096,
        },
    }

    def __init__(self, config: ServerConfig):
        """Initialize the local summarizer service.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.model_name = config.summarizer_model
        self.device = None
        self.model = None
        self.tokenizer = None
        self._model_cache_dir = Path(config.summarizer_model_cache_dir).expanduser()
        self._initialized = False
        self.max_pages = config.summarizer_max_pages

        # Generation parameters
        self.max_new_tokens = 1024
        self.temperature = 0.3
        self.do_sample = True
        self.top_p = 0.8

    async def initialize(self) -> None:
        """Initialize the model and tokenizer."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Select device
            self.device = self._select_device(self.config.summarizer_device)
            logger.info(f"Using device for summarizer: {self.device}")

            # Create cache directory
            self._model_cache_dir.mkdir(parents=True, exist_ok=True)

            # Load model and tokenizer
            logger.info(f"Loading summarizer model: {self.model_name}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=str(self._model_cache_dir),
                    trust_remote_code=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=str(self._model_cache_dir),
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    trust_remote_code=True,
                    device_map="auto" if self.device != "cpu" else None,
                )

                if self.device == "cpu":
                    self.model = self.model.to(self.device)

                self.model.eval()  # Set to evaluation mode

            except Exception as e:
                # Fallback to a smaller model if the requested one fails
                logger.warning(f"Failed to load {self.model_name}: {e}. Falling back to Qwen3-1.5B-Instruct")
                self.model_name = "Qwen/Qwen3-1.5B-Instruct"

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=str(self._model_cache_dir),
                    trust_remote_code=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=str(self._model_cache_dir),
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    trust_remote_code=True,
                    device_map="auto" if self.device != "cpu" else None,
                )

                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                self.model.eval()

            self._initialized = True
            logger.info(f"Local summarizer service initialized with model: {self.model_name}")

        except ImportError as e:
            raise EmbeddingError(
                f"Required packages not installed. Install with: pip install torch transformers: {e}",
                self.model_name,
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize local summarizer service: {e}", self.model_name, e)

    def _select_device(self, preferred_device: str) -> str:
        """Select the best available device.

        Args:
            preferred_device: User-specified device preference.

        Returns:
            Selected device string.
        """
        try:
            import torch

            # Check user preference first
            if preferred_device:
                if preferred_device == "mps" and torch.backends.mps.is_available():
                    return "mps"
                elif preferred_device == "cuda" and torch.cuda.is_available():
                    return "cuda"
                elif preferred_device == "cpu":
                    return "cpu"
                else:
                    logger.warning(f"Requested device '{preferred_device}' not available, auto-detecting")

            # Auto-detect best available device
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"

        except ImportError:
            return "cpu"

    def _create_summarization_prompt(self, content: str, filename: str = "") -> str:
        """Create a comprehensive prompt for document summarization.

        Args:
            content: Document content to summarize.
            filename: Optional filename for context.

        Returns:
            Formatted prompt for the LLM.
        """
        filename_context = f" The document filename is: {filename}." if filename else ""

        prompt = (
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
            f"Document content:\n{content}\n\nJSON Response:"
        )

        return prompt

    def _truncate_content(self, content: str) -> str:
        """Truncate content to fit within model context window.

        Args:
            content: Original document content.

        Returns:
            Truncated content that fits within the model's context.
        """
        # Get model context length (default to 4096 if not specified)
        model_spec = self.MODEL_SPECS.get(self.model_name, {})
        context_length = model_spec.get("context_length", 4096)

        # Reserve space for prompt and response
        max_content_tokens = context_length - 1024  # Reserve 1024 tokens for prompt + response

        # Rough estimation: 4 characters per token
        max_content_chars = max_content_tokens * 4

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
        """Summarize a document using the local LLM.

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
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._summarize_sync, content, filename)

        except Exception as e:
            logger.error(f"Failed to summarize document: {e}")
            # Return a basic fallback summary
            title = filename if filename else "Document"
            return DocumentSummary(
                title=title,
                short_description="Document summary unavailable due to processing error",
                long_description=(
                    "This document could not be automatically summarized. "
                    f"Original content length: {len(content)} characters."
                ),
            )

    def _summarize_sync(self, content: str, filename: str = "") -> DocumentSummary:
        """Synchronous summarization implementation.

        Args:
            content: The document content to summarize.
            filename: Optional filename for context.

        Returns:
            DocumentSummary with generated title and descriptions.
        """
        import torch

        # Truncate content if necessary
        truncated_content = self._truncate_content(content)

        # Create prompt
        prompt = self._create_summarization_prompt(truncated_content, filename)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Parse JSON response
        try:
            return self._parse_summary_response(response, filename)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response, using fallback: {e}")
            return self._create_fallback_summary(content, filename)

    def _parse_summary_response(self, response: str, filename: str = "") -> DocumentSummary:
        """Parse the LLM's JSON response into a DocumentSummary.

        Args:
            response: The LLM's response string.
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
        """Create a basic fallback summary when LLM parsing fails.

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
        """Test the summarizer service.

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
            logger.error(f"Local summarizer service test failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get information about the current summarizer model.

        Returns:
            Dictionary with model information.
        """
        model_spec = self.MODEL_SPECS.get(self.model_name, {})
        return {
            "provider": "local",
            "model": self.model_name,
            "model_size_gb": model_spec.get("size_gb", "unknown"),
            "description": model_spec.get("description", ""),
            "context_length": model_spec.get("context_length", "unknown"),
            "device": self.device or "not initialized",
            "max_pages": self.max_pages,
            "available_models": list(self.MODEL_SPECS.keys()),
        }
