"""Context-shifting utilities for large corpora.

This module implements a small pluggable ContextShiftManager that sits on top
of the existing VectorStore/TextIndex/EmbeddingService and decides a scoped
search strategy when the corpus is large. It's designed to be imported and
used by higher-level handlers without changing the core vector store.

Design goals:
- Minimal intrusion: new file only, no changes to existing modules.
- Lightweight: avoid heavy runtime deps; clustering is optional and only used
  when embeddings are available.
- Safe defaults: if no text index or embedding service is available, fall
  back to an unscoped search.

Usage (example):
    from pdfkb.context_shift import ContextShiftManager

    manager = ContextShiftManager(vector_store, embedding_service, text_index, config)
    results = await manager.scoped_search("my query", session_id="user-1", limit=5)

"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from .models import SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class ContextShiftManager:
    """Manage search scoping for large document collections.

    The manager provides a single public method `scoped_search` which will:
    - estimate corpus size
    - when the corpus is "large" (config or default threshold), perform a
      lightweight coarse retrieval to select a subset of document IDs
    - run the normal vector/hybrid search restricted to that subset

    The goal is to avoid touching internal vector store implementation and
    keep the integration surface small.
    """

    def __init__(
        self,
        vector_store,
        embedding_service: Optional[Any] = None,
        text_index: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.text_index = text_index
        self.config = config

        # In-memory session scopes: session_id -> {"doc_ids": [...], "ts": last_used}
        self._session_scopes: Dict[str, Dict[str, Any]] = {}

        # Threshold (number of documents) above which we trigger scoping.
        # Use config.large_corpus_threshold if present, otherwise default 1000.
        def _safe_int(val: Any, default: int) -> int:
            """Safely convert a config value to int.

            Avoid converting MagicMock or other complex test doubles which
            may implement __int__ and return misleading values (MagicMock -> 1).
            Only attempt conversion for plain ints or strings containing digits.
            """
            try:
                # Keep simple and deterministic: accept ints directly.
                if isinstance(val, int):
                    return val
                # Accept strings that represent integers.
                if isinstance(val, str):
                    return int(val)
            except Exception:
                pass
            return default

        # When tests pass a MagicMock for `config`, getattr will return another
        # MagicMock which is not int()-able. Use a safe cast to fall back to
        # sensible defaults instead of letting comparisons blow up during tests.
        self.large_corpus_threshold = (
            _safe_int(getattr(config, "large_corpus_threshold", None), 1000)
            if config
            else 1000
        )

        # How many docs to keep when scoping (tunable)
        self.scope_doc_limit = (
            _safe_int(getattr(config, "scope_doc_limit", None), 50) if config else 50
        )

        # Optional Redis-backed scope store
        self._redis = None
        # Only enable Redis when the config explicitly provides a bool True.
        raw_use_redis = getattr(config, "use_scope_redis", None) if config else None
        self._use_redis = True if isinstance(raw_use_redis, bool) and raw_use_redis else False
        self._scope_ttl = _safe_int(getattr(config, "scope_ttl_seconds", None), 3600) if config else 3600
        if self._use_redis:
            try:
                # Try modern redis asyncio client
                import redis.asyncio as redis_async

                self._redis = redis_async.from_url(
                    getattr(config, "scope_redis_url", "redis://localhost:6379/0")
                )
            except Exception as e:
                logger.warning(
                    f"Redis async client not available or failed to connect; falling back to in-memory scopes: {e}"
                )
                self._redis = None

    async def estimate_corpus_size(self) -> int:
        """Return number of indexed documents (best-effort)."""
        try:
            # prefer vector store method
            if hasattr(self.vector_store, "get_document_count"):
                return await self.vector_store.get_document_count()

            # fallback to chunk count / text index
            if hasattr(self.vector_store, "get_chunk_count"):
                return await self.vector_store.get_chunk_count()

            if self.text_index and hasattr(self.text_index, "get_document_count"):
                return await self.text_index.get_document_count()

        except Exception as e:
            logger.debug(f"Error estimating corpus size: {e}")

        return 0

    async def _coarse_retrieve_doc_ids(self, query: str, k: int) -> List[str]:
        """Perform a low-cost coarse retrieval to select candidate documents.

        Strategy:
        - If a text index is available, use BM25 to return top document chunk hits,
          extract document_ids and pick top-k unique documents.
        - Otherwise, if embeddings are available, compute query embedding and do
          a short vector search (limit=k*2) and extract document ids.
        - Fallback: return empty list meaning "don't scope".
        """
        try:
            doc_ids = []

            if self.text_index:
                # text_index.search returns chunk-level hits with document_id
                text_results = await self.text_index.search(query, limit=min(100, k * 10))
                for r in text_results:
                    did = r.get("document_id")
                    if did and did not in doc_ids:
                        doc_ids.append(did)
                        if len(doc_ids) >= k:
                            break

            elif self.embedding_service:
                # create query embedding then do a small vector search
                # EmbeddingService in this project exposes generate_embedding()
                emb = None
                if hasattr(self.embedding_service, "generate_embedding"):
                    emb = await self.embedding_service.generate_embedding(query)
                elif hasattr(self.embedding_service, "embed_text"):
                    emb = await self.embedding_service.embed_text(query)
                # Use vector_store.search but with a small limit; build SearchQuery
                q = SearchQuery(query=query, limit=min(k * 2, 20), metadata_filter=None, min_score=0.0)
                results = await self.vector_store.search(q, emb)
                for res in results:
                    did = getattr(res.document, "id", None)
                    if did and did not in doc_ids:
                        doc_ids.append(did)
                        if len(doc_ids) >= k:
                            break

            # Return up to k document ids
            return doc_ids[:k]

        except Exception as e:
            logger.debug(f"Coarse retrieval failed: {e}")
            return []

    def _make_metadata_filter_for_doc_ids(self, doc_ids: List[str]) -> Optional[Dict[str, Any]]:
        if not doc_ids:
            return None
        return {"document_id": {"$in": doc_ids}}

    async def _store_session_scope(self, session_id: str, doc_ids: List[str]) -> None:
        """Persist session scope either in Redis (if configured) or in-memory."""
        if not session_id:
            return
        try:
            if self._redis:
                # store as a JSON-like string using redis list
                await self._redis.delete(session_id)
                if doc_ids:
                    # use RPUSH and set TTL
                    await self._redis.rpush(session_id, *doc_ids)
                    await self._redis.expire(session_id, self._scope_ttl)
            else:
                # in-memory store
                self._session_scopes[session_id] = {"doc_ids": doc_ids, "ts": asyncio.get_event_loop().time()}
        except Exception as e:
            logger.debug(f"Failed to persist session scope: {e}")

    async def _load_session_scope(self, session_id: str) -> Optional[List[str]]:
        """Load session scope from Redis or in-memory store."""
        if not session_id:
            return None
        try:
            if self._redis:
                vals = await self._redis.lrange(session_id, 0, -1)
                # ensure str conversion
                if vals:
                    return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in vals]
                return None
            else:
                scope = self._session_scopes.get(session_id)
                return scope.get("doc_ids") if scope else None
        except Exception as e:
            logger.debug(f"Failed to load session scope: {e}")
            return None

    async def scoped_search(self, query: str, session_id: Optional[str] = None, limit: int = 5) -> List[SearchResult]:
        """Perform a context-shift aware search.

        Returns a list of SearchResult objects from the underlying vector store.
        """
        # quick validation
        if not query or not query.strip():
            return []

        try:
            corpus_size = await self.estimate_corpus_size()
            logger.debug(f"Corpus size estimate: {corpus_size}")

            # Default query object
            search_query = SearchQuery(query=query, limit=limit, metadata_filter=None, min_score=0.0)

            # If corpus is large, attempt to scope
            if corpus_size and corpus_size >= self.large_corpus_threshold:
                # If session already has a scope, reuse it
                if session_id:
                    scope = await self._load_session_scope(session_id)
                    if scope:
                        logger.debug(f"Using existing session scope with {len(scope)} docs")
                        search_query.metadata_filter = self._make_metadata_filter_for_doc_ids(scope)

                # Otherwise try to compute a new scope
                if not search_query.metadata_filter:
                    doc_ids = await self._coarse_retrieve_doc_ids(query, k=self.scope_doc_limit)
                    if doc_ids:
                        logger.info(f"Context-shift: scoping to {len(doc_ids)} documents")
                        search_query.metadata_filter = self._make_metadata_filter_for_doc_ids(doc_ids)
                        # store session scope (persist if configured)
                        if session_id:
                            await self._store_session_scope(session_id, doc_ids)

            # Compute embedding if vector/hybrid search will be used (vector_store handles routing)
            query_embedding = None
            if self.embedding_service and hasattr(self.embedding_service, "embed_text"):
                try:
                    query_embedding = await self.embedding_service.embed_text(query)
                except Exception:
                    query_embedding = None

            # Call into vector store (it will decide vector/text/hybrid internally)
            results = await self.vector_store.search(search_query, query_embedding)

            return results

        except Exception as e:
            logger.error(f"Scoped search failed: {e}")
            return []

    def paginate_results(
        self, results: List[SearchResult], page_size: int, page_token: Optional[str]
    ) -> Tuple[List[SearchResult], Optional[str]]:
        """Simple pagination helper returning next page token (opaque index).

        page_token is an optional string-encoded integer index. Returns (page_results, next_token).
        """
        try:
            start = int(page_token) if page_token else 0
        except Exception:
            start = 0

        page = results[start : start + page_size]
        next_index = start + page_size
        next_token = str(next_index) if next_index < len(results) else None
        return page, next_token

    def clear_session_scope(self, session_id: str) -> None:
        """Remove any stored scope for the session."""
        if session_id in self._session_scopes:
            del self._session_scopes[session_id]


__all__ = ["ContextShiftManager"]
