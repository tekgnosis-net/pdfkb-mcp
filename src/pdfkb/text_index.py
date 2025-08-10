"""Full-text search index using Whoosh."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from whoosh import index
from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer
from whoosh.fields import ID, NUMERIC, STORED, TEXT, Schema
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F

from .config import ServerConfig
from .exceptions import VectorStoreError
from .models import Document

logger = logging.getLogger(__name__)


class TextIndex:
    """Full-text search index using Whoosh."""

    def __init__(self, config: ServerConfig):
        """Initialize the text index.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.index_dir = Path(config.whoosh_index_dir)
        self.schema = self._create_schema()
        self.index = None
        self._writer_lock = None  # Will be created lazily per event loop
        self._lock_loop = None  # Track which loop the lock belongs to

    @property
    def writer_lock(self):
        """Get or create a writer lock for the current event loop.

        This ensures the lock is always compatible with the current event loop,
        even when running in background threads.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new lock that will bind later
            return asyncio.Lock()

        # Create a new lock if we don't have one or if it's for a different loop
        if self._writer_lock is None or self._lock_loop != current_loop:
            self._writer_lock = asyncio.Lock()
            self._lock_loop = current_loop

        return self._writer_lock

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Serialize metadata to JSON string, filtering out non-serializable objects.

        Args:
            metadata: Metadata dictionary to serialize.

        Returns:
            JSON string of serializable metadata.
        """
        if not metadata:
            return "{}"

        # Filter out non-serializable values
        serializable = {}
        for key, value in metadata.items():
            try:
                # Try to JSON serialize the value
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                logger.debug(f"Skipping non-serializable metadata field: {key}")
                continue

        return json.dumps(serializable)

    def _create_schema(self) -> Schema:
        """Create Whoosh schema matching document chunks.

        Returns:
            Whoosh schema for indexing chunks.
        """
        # Select analyzer based on config
        if self.config.whoosh_analyzer == "stemming":
            analyzer = StemmingAnalyzer()
        else:
            analyzer = StandardAnalyzer()

        return Schema(
            chunk_id=ID(unique=True, stored=True),
            document_id=ID(stored=True),
            text=TEXT(analyzer=analyzer, stored=True),
            metadata=STORED(),
            page_number=NUMERIC(stored=True),
            chunk_index=NUMERIC(stored=True),
        )

    async def initialize(self) -> None:
        """Initialize or open Whoosh index."""
        try:
            # Ensure index directory exists
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Check if index exists
            if index.exists_in(str(self.index_dir)):
                logger.info(f"Opening existing Whoosh index at {self.index_dir}")
                self.index = index.open_dir(str(self.index_dir))
            else:
                logger.info(f"Creating new Whoosh index at {self.index_dir}")
                self.index = index.create_in(str(self.index_dir), self.schema)

            logger.info("Text index initialized successfully")

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize text index: {e}", "initialize", e)

    async def add_document(self, document: Document) -> None:
        """Add document chunks to text index.

        Args:
            document: Document to add to the text index.
        """
        try:
            if not document.chunks:
                logger.warning(f"Document {document.id} has no chunks to index")
                return

            if self.index is None:
                await self.initialize()

            async with self.writer_lock:
                writer = self.index.writer()
                try:
                    chunks_added = 0

                    for chunk in document.chunks:
                        # Serialize metadata to avoid pickle errors
                        serialized_metadata = self._serialize_metadata(chunk.metadata)

                        # Use update_document which handles both insert and update
                        writer.update_document(
                            chunk_id=chunk.id,
                            document_id=document.id,
                            text=chunk.text,
                            metadata=serialized_metadata,
                            page_number=chunk.page_number,
                            chunk_index=chunk.chunk_index,
                        )
                        chunks_added += 1

                    writer.commit()
                    logger.info(f"Added/Updated {chunks_added} chunks from document {document.id} to text index")

                except Exception as e:
                    writer.cancel()
                    raise e

        except Exception as e:
            raise VectorStoreError(f"Failed to add document to text index: {e}", "add", e)

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform BM25 search and return scored results.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of search results with scores.
        """
        try:
            if self.index is None:
                await self.initialize()

            results = []

            with self.index.searcher(weighting=BM25F()) as searcher:
                # Parse the query
                query_parser = QueryParser("text", self.index.schema)
                parsed_query = query_parser.parse(query)

                # Perform search
                search_results = searcher.search(parsed_query, limit=limit)

                # Extract results
                for hit in search_results:
                    # Deserialize metadata from JSON string
                    metadata_str = hit.get("metadata", "{}")
                    try:
                        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                    result = {
                        "chunk_id": hit["chunk_id"],
                        "document_id": hit["document_id"],
                        "text": hit["text"],
                        "score": hit.score,
                        "page_number": hit.get("page_number"),
                        "chunk_index": hit.get("chunk_index", 0),
                        "metadata": metadata,
                    }

                    # Apply minimum score threshold
                    if result["score"] >= self.config.whoosh_min_score:
                        results.append(result)

            logger.info(f"Text search found {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Failed to search text index: {e}")
            return []

    async def delete_document(self, document_id: str) -> None:
        """Remove all chunks for a document from text index.

        Args:
            document_id: ID of the document to remove.
        """
        try:
            if self.index is None:
                await self.initialize()

            async with self.writer_lock:
                writer = self.index.writer()
                try:
                    # Delete all chunks for this document
                    # Note: delete_by_term returns None, not count
                    writer.delete_by_term("document_id", document_id)
                    writer.commit()
                    deleted_count = 0  # Whoosh doesn't return count

                    logger.info(f"Deleted {deleted_count} chunks for document {document_id} from text index")

                except Exception as e:
                    writer.cancel()
                    raise e

        except Exception as e:
            raise VectorStoreError(f"Failed to delete document from text index: {e}", "delete", e)

    async def get_document_count(self) -> int:
        """Get the total number of unique documents in the text index.

        Returns:
            Number of unique documents.
        """
        try:
            if self.index is None:
                await self.initialize()

            document_ids = set()

            with self.index.searcher() as searcher:
                for doc in searcher.documents():
                    doc_id = doc.get("document_id")
                    if doc_id:
                        document_ids.add(doc_id)

            return len(document_ids)

        except Exception as e:
            logger.error(f"Failed to count documents in text index: {e}")
            return 0

    async def get_chunk_count(self) -> int:
        """Get the total number of chunks in the text index.

        Returns:
            Number of chunks.
        """
        try:
            if self.index is None:
                await self.initialize()

            with self.index.searcher() as searcher:
                return searcher.doc_count_all()

        except Exception as e:
            logger.error(f"Failed to count chunks in text index: {e}")
            return 0

    async def reset_index(self) -> None:
        """Reset the entire text index by deleting and recreating it."""
        try:
            if self.index is not None:
                self.index.close()
                self.index = None

            # Delete existing index
            if self.index_dir.exists():
                import shutil

                shutil.rmtree(self.index_dir)
                logger.info(f"Deleted existing text index at {self.index_dir}")

            # Recreate index
            await self.initialize()
            logger.info("Text index reset successfully")

        except Exception as e:
            raise VectorStoreError(f"Failed to reset text index: {e}", "reset", e)

    async def close(self) -> None:
        """Close the text index."""
        try:
            if self.index is not None:
                self.index.close()
                self.index = None
                logger.info("Text index closed")

        except Exception as e:
            logger.error(f"Error closing text index: {e}")
