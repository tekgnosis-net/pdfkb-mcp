"""Web search service that wraps VectorStore and EmbeddingService functionality."""

import logging
import time
from typing import Any, Dict

from ...embeddings import EmbeddingService
from ...models import SearchQuery
from ...vector_store import VectorStore
from ...context_shift import ContextShiftManager
from ..models.web_models import SearchRequest, SearchResponse, SearchResultItem, SearchSuggestionsResponse

logger = logging.getLogger(__name__)


class WebSearchService:
    """Service for search operations via web interface."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        document_cache: Dict[str, Any],
    ):
        """Initialize the web search service.

        Args:
            vector_store: Vector storage service
            embedding_service: Embedding generation service
            document_cache: Document metadata cache
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.document_cache = document_cache
        # create a ContextShiftManager for web searches (use vector_store.config if available)
        try:
            cfg = getattr(self.vector_store, "config", None)
            self.context_manager = ContextShiftManager(self.vector_store, self.embedding_service, self.vector_store.text_index, cfg)
        except Exception:
            self.context_manager = None

    async def search(self, search_request: SearchRequest) -> SearchResponse:
        """Perform vector similarity search.

        Args:
            search_request: Search request parameters

        Returns:
            SearchResponse with search results
        """
        try:
            start_time = time.time()

            logger.info(f"Performing search: {search_request.query} (limit: {search_request.limit})")

            # Create internal search query object
            search_query = SearchQuery(
                query=search_request.query,
                limit=search_request.limit,
                metadata_filter=search_request.metadata_filter,
                min_score=search_request.min_score,
            )

            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(search_request.query)
            if not query_embedding:
                raise ValueError("Failed to generate query embedding")

            # Perform search (use ContextShiftManager when available)
            if getattr(self, "context_manager", None):
                # Attempt to use a session identifier if SearchRequest carries one
                session_id = getattr(search_request, "session_id", None)
                search_results = await self.context_manager.scoped_search(search_request.query, session_id=session_id, limit=search_request.limit)
            else:
                search_results = await self.vector_store.search(search_query, query_embedding)

            # Convert to web response format
            result_items = []
            for result in search_results:
                result_item = SearchResultItem(
                    document_id=result.document.id,
                    document_title=result.document.title or result.document.filename,
                    document_path=result.document.path,
                    chunk_id=result.chunk.id,
                    chunk_text=result.chunk.text,
                    page_number=result.chunk.page_number,
                    chunk_index=result.chunk.chunk_index,
                    score=result.score,
                    metadata={
                        **result.document.metadata,
                        **result.chunk.metadata,
                    },
                )
                result_items.append(result_item)

            search_time = time.time() - start_time

            logger.info(f"Search completed: {len(result_items)} results in {search_time:.2f}s")

            return SearchResponse(
                results=result_items,
                total_results=len(result_items),
                query=search_request.query,
                search_time=search_time,
                metadata={
                    "limit": search_request.limit,
                    "min_score": search_request.min_score,
                    "metadata_filter": search_request.metadata_filter,
                    "include_chunks": search_request.include_chunks,
                },
            )

        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise

    async def get_search_suggestions(self, query_fragment: str) -> SearchSuggestionsResponse:
        """Get search query suggestions based on existing content.

        Args:
            query_fragment: Partial query to generate suggestions for

        Returns:
            SearchSuggestionsResponse with suggested queries
        """
        try:
            # For now, return basic suggestions based on document titles and common terms
            # In a more advanced implementation, this could use:
            # - Document titles and metadata
            # - Frequently searched terms
            # - NLP-based query expansion
            # - Vector similarity to find related content

            suggestions = []
            query_lower = query_fragment.lower()

            # Basic suggestions from document titles
            for document in self.document_cache.values():
                if document.title:
                    title_words = document.title.lower().split()
                    for word in title_words:
                        if len(word) > 3 and query_lower in word and word not in suggestions:
                            suggestions.append(word.capitalize())

            # Add some common query patterns
            common_patterns = [
                f"{query_fragment} overview",
                f"{query_fragment} introduction",
                f"{query_fragment} summary",
                f"{query_fragment} definition",
                f"{query_fragment} examples",
            ]

            for pattern in common_patterns:
                if pattern not in suggestions and len(pattern) > len(query_fragment):
                    suggestions.append(pattern)

            # Limit suggestions and remove duplicates
            suggestions = list(set(suggestions))[:10]

            return SearchSuggestionsResponse(
                suggestions=suggestions,
                query=query_fragment,
            )

        except Exception as e:
            logger.error(f"Error generating search suggestions: {e}")
            return SearchSuggestionsResponse(
                suggestions=[],
                query=query_fragment,
            )

    async def get_similar_documents(self, document_id: str, limit: int = 5) -> SearchResponse:
        """Find documents similar to the given document.

        Args:
            document_id: Reference document ID
            limit: Maximum number of similar documents to return

        Returns:
            SearchResponse with similar documents
        """
        try:
            if document_id not in self.document_cache:
                raise ValueError(f"Document not found: {document_id}")

            document = self.document_cache[document_id]

            # Use the document title or first chunk as query
            query_text = document.title or ""
            if not query_text and document.chunks:
                query_text = document.chunks[0].text[:200]  # First 200 chars

            if not query_text:
                raise ValueError("No content available for similarity search")

            # Create search request
            search_request = SearchRequest(
                query=query_text,
                limit=limit + 1,  # +1 because we'll filter out the original document
                min_score=0.1,
                metadata_filter={"document_id": {"$ne": document_id}},  # Exclude the source document
            )

            # Perform search
            results = await self.search(search_request)

            # Filter out the original document if it appears in results
            filtered_results = [result for result in results.results if result.document_id != document_id][:limit]

            return SearchResponse(
                results=filtered_results,
                total_results=len(filtered_results),
                query=f"Similar to: {document.title or document.filename}",
                search_time=results.search_time,
                metadata={
                    "similarity_search": True,
                    "reference_document_id": document_id,
                    "limit": limit,
                },
            )

        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            raise

    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get search-related statistics.

        Returns:
            Dictionary with search statistics
        """
        try:
            # Get vector store statistics
            document_count = await self.vector_store.get_document_count()
            chunk_count = await self.vector_store.get_chunk_count()

            # Calculate average chunks per document
            avg_chunks = chunk_count / document_count if document_count > 0 else 0

            # Get document type distribution
            doc_types = {}
            for document in self.document_cache.values():
                doc_type = document.metadata.get("document_type", "pdf")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            return {
                "document_count": document_count,
                "chunk_count": chunk_count,
                "average_chunks_per_document": round(avg_chunks, 2),
                "document_types": doc_types,
                "embedding_model": self.embedding_service.config.embedding_model,
                "vector_dimensions": (
                    len(await self.embedding_service.generate_embedding("test")) if chunk_count > 0 else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {
                "document_count": 0,
                "chunk_count": 0,
                "average_chunks_per_document": 0,
                "document_types": {},
                "error": str(e),
            }
