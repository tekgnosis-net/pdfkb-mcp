"""Hybrid search engine combining vector and text search with RRF."""

import asyncio
import logging
from typing import Any, Dict, List

from .config import ServerConfig
from .models import Chunk, Document, SearchQuery, SearchResult
from .text_index import TextIndex

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Combines vector and text search with RRF."""

    def __init__(self, vector_store, text_index: TextIndex, config: ServerConfig):
        """Initialize the hybrid search engine.

        Args:
            vector_store: VectorStore instance for semantic search.
            text_index: TextIndex instance for BM25 search.
            config: Server configuration.
        """
        self.vector_store = vector_store
        self.text_index = text_index
        self.config = config

    async def search(self, query: SearchQuery, query_embedding: List[float]) -> List[SearchResult]:
        """Execute hybrid search with RRF fusion.

        Args:
            query: Search query parameters.
            query_embedding: Query embedding vector.

        Returns:
            List of search results fused using RRF.
        """
        try:
            # Execute searches in parallel
            # Get more results than requested for better fusion
            expanded_limit = min(query.limit * 3, 50)

            # Create tasks for parallel execution
            vector_task = asyncio.create_task(self._vector_search(query, query_embedding, expanded_limit))
            text_task = asyncio.create_task(self.text_index.search(query.query, expanded_limit))

            # Wait for both searches to complete
            vector_results, text_results = await asyncio.gather(vector_task, text_task)

            # Apply RRF fusion
            fused_results = self._apply_rrf(vector_results, text_results, query.limit)

            logger.info(
                f"Hybrid search: {len(vector_results)} vector results, "
                f"{len(text_results)} text results, {len(fused_results)} fused results"
            )

            return fused_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fall back to vector search on error
            return await self._vector_search(query, query_embedding, query.limit)

    async def _vector_search(self, query: SearchQuery, query_embedding: List[float], limit: int) -> List[SearchResult]:
        """Perform vector search.

        Args:
            query: Search query parameters.
            query_embedding: Query embedding vector.
            limit: Maximum number of results.

        Returns:
            List of vector search results.
        """
        # Create a modified query with the expanded limit
        vector_query = SearchQuery(
            query=query.query,
            limit=limit,
            metadata_filter=query.metadata_filter,
            min_score=query.min_score,
            search_type="vector",
        )

        # Use the existing vector store search
        return await self.vector_store._vector_search(vector_query, query_embedding)

    def _apply_rrf(
        self, vector_results: List[SearchResult], text_results: List[Dict[str, Any]], limit: int
    ) -> List[SearchResult]:
        """Apply Reciprocal Rank Fusion to merge results.

        RRF formula: score = Σ(1 / (k + rank_i))
        where k is the RRF constant, rank_i is the rank in result set i

        Args:
            vector_results: Results from vector search.
            text_results: Results from text search.
            limit: Maximum number of results to return.

        Returns:
            Fused and ranked search results.
        """
        k = self.config.rrf_k
        vector_weight = self.config.hybrid_search_weights["vector"]
        text_weight = self.config.hybrid_search_weights["text"]

        # Store RRF scores by chunk ID
        rrf_scores = {}

        # Store original results by chunk ID for reconstruction
        chunk_data = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk.id

            # Calculate RRF score for this result
            rrf_score = vector_weight * (1 / (k + rank))

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0
                chunk_data[chunk_id] = {
                    "result": result,
                    "vector_score": result.score,
                    "vector_rank": rank,
                    "text_score": None,
                    "text_rank": None,
                }

            rrf_scores[chunk_id] += rrf_score
            chunk_data[chunk_id]["vector_score"] = result.score
            chunk_data[chunk_id]["vector_rank"] = rank

        # Process text results
        for rank, text_result in enumerate(text_results, start=1):
            chunk_id = text_result["chunk_id"]

            # Calculate RRF score for this result
            rrf_score = text_weight * (1 / (k + rank))

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0

                # Need to create a SearchResult from text result
                chunk = Chunk(
                    id=chunk_id,
                    document_id=text_result["document_id"],
                    text=text_result["text"],
                    page_number=text_result.get("page_number"),
                    chunk_index=text_result.get("chunk_index", 0),
                    metadata=text_result.get("metadata", {}),
                )

                # Create minimal document
                document = Document(
                    id=text_result["document_id"], path="", chunks=[]  # Will be filled from vector store if needed
                )

                search_result = SearchResult(
                    chunk=chunk,
                    score=0,  # Will be updated with RRF score
                    document=document,
                    search_type="text",
                    text_score=text_result["score"],
                )

                chunk_data[chunk_id] = {
                    "result": search_result,
                    "vector_score": None,
                    "vector_rank": None,
                    "text_score": text_result["score"],
                    "text_rank": rank,
                }

            rrf_scores[chunk_id] += rrf_score
            chunk_data[chunk_id]["text_score"] = text_result["score"]
            chunk_data[chunk_id]["text_rank"] = rank

        # Sort by RRF score and create final results
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_results = []
        for chunk_id in sorted_chunk_ids[:limit]:
            data = chunk_data[chunk_id]
            result = data["result"]

            # Update result with hybrid information
            result.score = rrf_scores[chunk_id]
            result.search_type = "hybrid"
            result.vector_score = data["vector_score"]
            result.text_score = data["text_score"]

            final_results.append(result)

            logger.debug(
                f"RRF result: chunk_id={chunk_id[:8]}..., "
                f"rrf_score={result.score:.4f}, "
                f"vector_rank={data['vector_rank']}, "
                f"text_rank={data['text_rank']}"
            )

        return final_results
