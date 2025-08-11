"""Semantic chunker using LangChain's SemanticChunker."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..embeddings_base import EmbeddingService
from ..models import Chunk
from .chunker import Chunker
from .langchain_embeddings_wrapper import LangChainEmbeddingsWrapper

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    """Semantic chunker using LangChain's SemanticChunker.

    This chunker uses embedding similarity to identify natural breakpoints
    in the text, creating more coherent and contextually complete chunks.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        buffer_size: int = 1,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        min_chunk_size: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
    ):
        """Initialize semantic chunker with configurable parameters.

        Args:
            embedding_service: The embedding service to use for similarity computation.
            breakpoint_threshold_type: Method for breakpoint detection:
                - "percentile": Split at distances exceeding Nth percentile
                - "standard_deviation": Split at mean + N*std_dev
                - "interquartile": Split at mean + N*IQR
                - "gradient": Use gradient-based detection
            breakpoint_threshold_amount: Threshold value (interpretation depends on type).
            buffer_size: Number of sentences to include as context around breakpoints.
            number_of_chunks: Target number of chunks (overrides threshold if set).
            sentence_split_regex: Regex pattern for sentence splitting.
            min_chunk_size: Minimum chunk size in tokens (deprecated, use min_chunk_chars).
            min_chunk_chars: Minimum chunk size in characters.
        """
        super().__init__()
        self.embedding_service = embedding_service
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.buffer_size = buffer_size
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        self.min_chunk_chars = min_chunk_chars

        # Create LangChain-compatible embeddings wrapper
        self.embeddings_wrapper = LangChainEmbeddingsWrapper(embedding_service)

        # Initialize the LangChain SemanticChunker (lazy loading)
        self._splitter = None

    def _get_splitter(self):
        """Lazy initialization of the LangChain SemanticChunker."""
        if self._splitter is None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
            except ImportError:
                raise ImportError(
                    "LangChain experimental not available. " "Install with: pip install 'pdfkb-mcp[semantic]'"
                )

            # Create the semantic chunker with our embeddings
            kwargs = {
                "embeddings": self.embeddings_wrapper,
                "breakpoint_threshold_type": self.breakpoint_threshold_type,
                "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
                "buffer_size": self.buffer_size,
                "sentence_split_regex": self.sentence_split_regex,
            }

            # Add optional parameters if specified
            if self.number_of_chunks is not None:
                kwargs["number_of_chunks"] = self.number_of_chunks
            if self.min_chunk_size is not None:
                kwargs["min_chunk_size"] = self.min_chunk_size

            self._splitter = LangChainSemanticChunker(**kwargs)

        return self._splitter

    def chunk(self, markdown_content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk content using semantic similarity.

        Args:
            markdown_content: Markdown text to chunk.
            metadata: Document metadata.

        Returns:
            List of Chunk objects.
        """
        try:
            if not markdown_content or not markdown_content.strip():
                logger.warning("Empty markdown content provided to semantic chunker")
                return []

            # Get the semantic chunker
            splitter = self._get_splitter()

            # Split the text using semantic similarity
            text_chunks = splitter.split_text(markdown_content)

            # Apply minimum chunk size filter if specified
            if self.min_chunk_chars:
                text_chunks = [chunk for chunk in text_chunks if len(chunk) >= self.min_chunk_chars]

            # Convert to Chunk objects with metadata
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:  # Skip empty chunks
                    continue

                chunk_metadata = {
                    "chunk_strategy": "semantic",
                    "breakpoint_threshold_type": self.breakpoint_threshold_type,
                    "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
                    "buffer_size": self.buffer_size,
                    "sentence_split_regex": self.sentence_split_regex,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                # Add optional parameters to metadata
                if self.number_of_chunks is not None:
                    chunk_metadata["number_of_chunks"] = self.number_of_chunks
                if self.min_chunk_chars is not None:
                    chunk_metadata["min_chunk_chars"] = self.min_chunk_chars

                # Add any provided metadata
                chunk_metadata.update(metadata)

                chunk = Chunk(text=chunk_text, chunk_index=i, metadata=chunk_metadata)
                chunks.append(chunk)

            logger.info(
                f"Created {len(chunks)} chunks using semantic chunking "
                f"(type={self.breakpoint_threshold_type}, amount={self.breakpoint_threshold_amount})"
            )
            return chunks

        except ImportError as e:
            logger.error(f"Failed to import required dependencies: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to chunk content with semantic chunker: {e}")
            # Could implement fallback here if desired
            raise RuntimeError(f"Semantic chunking failed: {e}") from e
