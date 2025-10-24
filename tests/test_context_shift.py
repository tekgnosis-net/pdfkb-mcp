import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pdfkb.context_shift import ContextShiftManager
from pdfkb.models import SearchQuery, SearchResult, Document, Chunk


class DummyEmbeddingService:
    def __init__(self):
        self.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])


class DummyVectorStore:
    def __init__(self):
        self.get_document_count = AsyncMock(return_value=0)
        self.get_chunk_count = AsyncMock(return_value=0)
        self.search = AsyncMock(return_value=[])


class DummyTextIndex:
    def __init__(self, results=None):
        # results should be a list of dicts with document_id keys
        self._results = results or []

    async def search(self, query: str, limit: int = 10):
        return self._results[:limit]


@pytest.mark.asyncio
async def test_small_corpus_no_scoping_calls_vector_search():
    vs = DummyVectorStore()
    vs.get_document_count.return_value = 10

    emb = DummyEmbeddingService()

    manager = ContextShiftManager(vs, embedding_service=emb, text_index=None, config=MagicMock())
    # small corpus threshold default is 1000, so no scoping expected
    results = await manager.scoped_search("test query", session_id="s1", limit=3)

    # vector_store.search should have been called once
    assert vs.search.call_count == 1


@pytest.mark.asyncio
async def test_large_corpus_scopes_using_text_index():
    vs = DummyVectorStore()
    vs.get_document_count.return_value = 5000

    # Return some fake BM25 chunk hits from text index
    hits = [
        {"chunk_id": "c1", "document_id": "doc_a", "text": "a"},
        {"chunk_id": "c2", "document_id": "doc_b", "text": "b"},
        {"chunk_id": "c3", "document_id": "doc_c", "text": "c"},
    ]
    ti = DummyTextIndex(results=hits)

    emb = DummyEmbeddingService()

    cfg = MagicMock()
    cfg.large_corpus_threshold = 1000
    cfg.scope_doc_limit = 2

    manager = ContextShiftManager(vs, embedding_service=emb, text_index=ti, config=cfg)

    results = await manager.scoped_search("find me something", session_id="sess1", limit=5)

    # vector_store.search should have been called once
    assert vs.search.call_count == 1

    # The search call should have been given a SearchQuery whose metadata_filter limits to 2 doc ids
    called_query = vs.search.call_args[0][0]
    assert isinstance(called_query, SearchQuery)
    mf = called_query.metadata_filter
    assert mf is not None and "document_id" in mf and "$in" in mf["document_id"]
    assert len(mf["document_id"]["$in"]) <= 2


@pytest.mark.asyncio
async def test_paginate_results():
    vs = DummyVectorStore()
    manager = ContextShiftManager(vs)

    # Create fake SearchResult list
    doc = Document(id="doc1", path="/tmp/a.pdf", chunks=[])
    chunk = Chunk(id="chunk1", document_id="doc1", text="hello")
    results = [SearchResult(chunk=chunk, score=1.0, document=doc) for _ in range(12)]

    page, token = manager.paginate_results(results, page_size=5, page_token=None)
    assert len(page) == 5
    assert token == "5"

    page2, token2 = manager.paginate_results(results, page_size=5, page_token=token)
    assert len(page2) == 5
    assert token2 == "10"
