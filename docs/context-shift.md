## Context Shift Manager (scoped search)

This project includes a pluggable `ContextShiftManager` that helps scale search and list responses
when your knowledgebase grows large. It performs a lightweight coarse retrieval and scopes
vector/hybrid searches to a small set of candidate documents, reducing runtime and improving
relevance in large corpora.

How it works

- Estimate corpus size (via `vector_store.get_document_count()` or fallbacks).
- If the corpus is larger than `config.large_corpus_threshold`, perform a coarse retrieval:
  - Use Whoosh BM25 (`TextIndex`) if available to find chunk hits and extract document IDs.
  - Otherwise use a small vector search using `EmbeddingService.generate_embedding()`.
- Build a metadata filter `{"document_id": {"$in": [...]}}` and forward that to `VectorStore.search()`.

Integration

1. The MCP server initializes `ContextShiftManager` automatically when `config.enable_context_shifting` is true.

2. Web handlers and MCP tools use `ContextShiftManager.scoped_search(query, session_id, limit)` instead of calling
   `vector_store.search()` directly.

Tuning

- `large_corpus_threshold` (default 1000): minimum number of documents to trigger scoping.
- `scope_doc_limit` (default 50): number of documents to keep for the scoped search.
- `use_scope_redis` (default false): optionally persist session scopes in Redis for multi-instance deployments.

Run directives and summarization

To reduce cost and enable better coarse retrieval for very long documents, you can enable the summarizer
(`config.enable_summarizer`) and use a run directive to request summarization before embedding. Summaries
should be cached (document metadata) to avoid repeated LLM calls.

Example

```py
from pdfkb.context_shift import ContextShiftManager

manager = ContextShiftManager(vector_store, embedding_service, text_index, config)
results = await manager.scoped_search("how to configure X", session_id="user-1", limit=5)
```

For more details and examples see the code in `src/pdfkb/context_shift.py`.
