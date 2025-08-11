"""LangChain-compatible wrapper for the EmbeddingService."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_core.embeddings import Embeddings

from ..embeddings_base import EmbeddingService

logger = logging.getLogger(__name__)


class LangChainEmbeddingsWrapper(Embeddings):
    """Wrapper to make our EmbeddingService compatible with LangChain.

    This wrapper allows LangChain components (like SemanticChunker) to use
    our existing embedding services (local or OpenAI) without modification.
    """

    def __init__(self, embedding_service: EmbeddingService):
        """Initialize the wrapper with an embedding service.

        Args:
            embedding_service: The underlying embedding service to wrap.
        """
        self.embedding_service = embedding_service
        self._initialized = False
        self._thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embeddings")
        self._thread_loop = None
        self._thread = None

    def _ensure_thread_loop(self):
        """Ensure we have a dedicated thread with an event loop for sync operations."""
        if self._thread_loop is None or not self._thread or not self._thread.is_alive():
            self._thread_loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._thread_loop.run_forever, daemon=True, name="embeddings-loop")
            self._thread.start()

    def _run_async_in_thread(self, coro):
        """Run an async coroutine in our dedicated thread's event loop."""
        self._ensure_thread_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._thread_loop)
        return future.result()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding for multiple documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """

        async def _embed():
            if not self._initialized:
                await self.embedding_service.initialize()
                self._initialized = True
            return await self.embedding_service.generate_embeddings(texts)

        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, use the dedicated thread
                return self._run_async_in_thread(_embed())
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                return asyncio.run(_embed())
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Synchronous embedding for a single query.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """

        async def _embed():
            if not self._initialized:
                await self.embedding_service.initialize()
                self._initialized = True
            return await self.embedding_service.generate_embedding(text)

        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, use the dedicated thread
                return self._run_async_in_thread(_embed())
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                return asyncio.run(_embed())
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embedding for multiple documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not self._initialized:
            await self.embedding_service.initialize()
            self._initialized = True
        return await self.embedding_service.generate_embeddings(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embedding for a single query.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        if not self._initialized:
            await self.embedding_service.initialize()
            self._initialized = True
        return await self.embedding_service.generate_embedding(text)

    def __del__(self):
        """Clean up resources."""
        try:
            if self._thread_loop and self._thread_loop.is_running():
                self._thread_loop.call_soon_threadsafe(self._thread_loop.stop)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1)
            self._thread_pool.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors
