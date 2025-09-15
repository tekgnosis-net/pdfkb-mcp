"""Main MCP server implementation using FastMCP."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastmcp import FastMCP

from .config import ServerConfig
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingService
from .exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
    EmbeddingError,
    PDFProcessingError,
    ValidationError,
    VectorStoreError,
)
from .file_monitor import FileMonitor
from .intelligent_cache import IntelligentCacheManager
from .models import Document, SearchQuery
from .vector_store import VectorStore

if TYPE_CHECKING:
    from .background_queue import BackgroundProcessingQueue

logger = logging.getLogger(__name__)


class PDFKnowledgebaseServer:
    """Main MCP server implementation for PDF knowledgebase management."""

    def __init__(
        self, config: Optional[ServerConfig] = None, background_queue: Optional["BackgroundProcessingQueue"] = None
    ):
        """Initialize the PDF knowledgebase server.

        Args:
            config: Server configuration. If None, loads from environment.
            background_queue: Optional background processing queue for non-blocking operations.
        """
        self.config = config or ServerConfig.from_env()
        self.app = FastMCP("PDF Knowledgebase")
        self.document_processor: Optional[DocumentProcessor] = None
        self.vector_store: Optional[VectorStore] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.reranker_service = None
        self.summarizer_service = None
        self.file_monitor: Optional[FileMonitor] = None
        self.cache_manager: Optional[IntelligentCacheManager] = None
        self.background_queue = background_queue
        self._web_document_service = None  # Optional reference to web document service

        # Document metadata cache
        self._document_cache: Dict[str, Document] = {}
        self._cache_file = self.config.metadata_path / "documents.json"

        # Semaphores for controlling parallel processing
        self._parsing_semaphore = asyncio.Semaphore(self.config.max_parallel_parsing)
        self._embedding_semaphore = asyncio.Semaphore(self.config.max_parallel_embedding)

        self._setup_tools()
        self._setup_resources()

    async def initialize_core(self) -> None:
        """Initialize core components (excluding FileMonitor) asynchronously."""
        try:
            logger.info("Initializing PDF Knowledgebase server core components...")

            # Initialize cache manager
            self.cache_manager = IntelligentCacheManager(self.config, self.config.cache_dir)

            # Check for configuration changes and handle selective cache invalidation
            await self._handle_intelligent_config_changes()

            # Initialize components in order
            self.embedding_service = EmbeddingService(self.config)
            await self.embedding_service.initialize()

            # Initialize reranker service if enabled
            if self.config.enable_reranker:
                from .reranker_factory import get_reranker_service

                try:
                    self.reranker_service = get_reranker_service(self.config)
                    if self.reranker_service:
                        await self.reranker_service.initialize()
                        logger.info("Reranker service initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize reranker service: {e}")
                    logger.warning("Continuing without reranker")
                    self.reranker_service = None

            # Initialize summarizer service if enabled
            self.summarizer_service = None
            if self.config.enable_summarizer:
                try:
                    from .summarizer_factory import create_summarizer_service

                    self.summarizer_service = create_summarizer_service(self.config)
                    if self.summarizer_service:
                        await self.summarizer_service.initialize()
                        logger.info("Summarizer service initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize summarizer service: {e}")
                    logger.warning("Continuing without summarizer")
                    self.summarizer_service = None

            self.vector_store = VectorStore(self.config)
            self.vector_store.set_embedding_service(self.embedding_service)
            self.vector_store.set_reranker_service(self.reranker_service)
            await self.vector_store.initialize()

            self.document_processor = DocumentProcessor(
                self.config,
                self.embedding_service,
                self.cache_manager,
                self._embedding_semaphore,
                self.summarizer_service,
            )

            # Log startup configuration summary for diagnostics
            try:
                parser_name = (
                    type(self.document_processor.parser).__name__
                    if self.document_processor
                    else str(self.config.pdf_parser)
                )
                chunker_name = (
                    type(self.document_processor.chunker).__name__
                    if self.document_processor
                    else str(self.config.document_chunker)
                )
            except Exception:
                parser_name = str(self.config.pdf_parser)
                chunker_name = str(self.config.pdf_chunker)
            reranker_info = f"Enabled ({self.config.reranker_model})" if self.config.enable_reranker else "Disabled"
            logger.info(
                "Startup configuration: Parser=%s, Chunker=%s, EmbeddingModel=%s, Reranker=%s, "
                "KnowledgebasePath=%s, CacheDir=%s",
                parser_name,
                chunker_name,
                self.config.embedding_model,
                reranker_info,
                self.config.knowledgebase_path,
                self.config.cache_dir,
            )
            # Load document metadata cache BEFORE re-processing (needed for re-summarization)
            await self._load_document_cache()

            # Handle re-processing of cached documents after components are initialized
            await self._handle_post_initialization_reprocessing()

            # Update intelligent cache fingerprints
            self.cache_manager.update_fingerprints()

            logger.info("PDF Knowledgebase server core components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise ConfigurationError(f"Core initialization failed: {e}")

    async def initialize_file_monitor(self, web_document_service=None) -> None:
        """Initialize file monitor with optional web document service.

        Args:
            web_document_service: Optional web document service for in-progress tracking
        """
        try:
            logger.info("Initializing file monitor...")

            # Store web document service reference
            self._web_document_service = web_document_service

            self.file_monitor = FileMonitor(
                self.config,
                self.document_processor,
                self.vector_store,
                self._update_document_cache,
                background_queue=self.background_queue,
                web_document_service=self._web_document_service,
            )
            await self.file_monitor.start_monitoring()

            logger.info("File monitor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize file monitor: {e}")
            raise ConfigurationError(f"File monitor initialization failed: {e}")

    async def initialize(self, web_document_service=None) -> None:
        """Initialize all components asynchronously.

        Args:
            web_document_service: Optional web document service for in-progress tracking
        """
        await self.initialize_core()
        await self.initialize_file_monitor(web_document_service)

    async def _handle_intelligent_config_changes(self) -> None:
        """Handle configuration changes using intelligent cache management for selective invalidation."""
        try:
            changes = self.cache_manager.detect_config_changes()

            # Check if any changes detected
            if not any(changes.values()):
                logger.info("No configuration changes detected, using existing caches")
                self._config_changes = None
                return

            logger.info(f"Configuration changes detected: {changes}")

            # Store changes for later re-processing after components are initialized
            self._config_changes = changes

            # Handle parsing changes - most destructive, affects everything downstream
            if changes["parsing"]:
                logger.warning("Parsing configuration changed - full database reset required")
                await self._reset_all_caches()
                return

            # Handle chunking changes - affects chunking and embedding stages
            if changes["chunking"]:
                logger.warning("Chunking configuration changed - clearing chunking and embedding caches")
                await self._reset_chunking_and_embedding_caches()

            # Handle embedding-only changes - least destructive
            elif changes["embedding"]:
                logger.info("Embedding configuration changed - clearing embedding caches only")
                await self._reset_embedding_caches()

            # Handle summarizer-only changes - no cache operations needed
            elif changes["summarizer"]:
                logger.info("Summarizer configuration changed - will re-summarize documents")
                # No cache clearing needed for summarizer changes

            logger.info("Selective cache invalidation complete")

        except Exception as e:
            logger.error(f"Error handling intelligent configuration changes: {e}")
            # Fallback to full reset on error
            logger.warning("Falling back to full cache reset due to error")
            await self._reset_all_caches()
            self._config_changes = None

    async def _reset_all_caches(self) -> None:
        """Reset all caches and vector database (full reset)."""
        try:
            logger.info("Performing full cache reset...")

            # Initialize vector store temporarily to reset it
            temp_vector_store = VectorStore(self.config)
            await temp_vector_store.initialize()
            await temp_vector_store.reset_database()
            await temp_vector_store.close()

            # Clear document cache
            if self._cache_file.exists():
                self._cache_file.unlink()
                logger.info("Cleared document cache")

            # Clear file monitor metadata
            file_index_path = self.config.metadata_path / "file_index.json"
            if file_index_path.exists():
                file_index_path.unlink()
                logger.info("Cleared file monitor index")

            # Clear processing caches
            if self.config.processing_path.exists():
                import shutil

                shutil.rmtree(self.config.processing_path)
                self.config.processing_path.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared processing caches")

            # Clear intelligent cache fingerprints
            self.cache_manager.clear_all_fingerprints()

            logger.info("Full cache reset complete. All PDFs will be re-processed.")

        except Exception as e:
            logger.error(f"Error during full cache reset: {e}")
            raise ConfigurationError(f"Failed to reset caches: {e}")

    async def _reset_chunking_and_embedding_caches(self) -> None:
        """Reset chunking and embedding caches, preserve parsing results."""
        try:
            logger.info("Resetting chunking and embedding caches...")

            # Reset vector database (contains embeddings)
            temp_vector_store = VectorStore(self.config)
            await temp_vector_store.initialize()
            await temp_vector_store.reset_database()
            await temp_vector_store.close()

            # Clear document cache (will be repopulated)
            if self._cache_file.exists():
                self._cache_file.unlink()
                logger.info("Cleared document cache")

            # Clear chunking and embedding stage fingerprints
            self.cache_manager.clear_stage_fingerprint("chunking")
            self.cache_manager.clear_stage_fingerprint("embedding")

            logger.info("Chunking and embedding caches reset complete")

        except Exception as e:
            logger.error(f"Error resetting chunking and embedding caches: {e}")
            raise ConfigurationError(f"Failed to reset chunking/embedding caches: {e}")

    async def _reprocess_existing_documents(self) -> None:
        """Re-process existing cached documents after configuration changes.

        This method is called after cache clearing to re-process cached markdown
        files with new chunking/embedding configurations, avoiding expensive
        PDF parsing while ensuring documents are processed with current settings.
        """
        try:
            logger.info("Re-processing existing cached documents with new configuration...")

            # Re-process cached markdown files (skips expensive parsing)
            documents = await self.document_processor.reprocess_cached_documents()

            if documents:
                # Add all documents to vector store
                for document in documents:
                    await self.vector_store.add_document(document)
                    self._document_cache[document.id] = document

                # Save updated document cache
                await self._save_document_cache()

                logger.info(f"✓ Successfully re-processed {len(documents)} cached documents")
            else:
                logger.info("No cached documents found to re-process")

        except Exception as e:
            logger.error(f"Error re-processing cached documents: {e}")
            # Don't raise - let system continue with empty state
            # This allows the system to recover gracefully if re-processing fails

    async def _reset_embedding_caches(self) -> None:
        """Reset only embedding caches, preserve parsing and chunking results."""
        try:
            logger.info("Resetting embedding caches only...")

            # Reset vector database (contains embeddings)
            temp_vector_store = VectorStore(self.config)
            await temp_vector_store.initialize()
            await temp_vector_store.reset_database()
            await temp_vector_store.close()

            # Clear document cache (will be repopulated with new embeddings)
            if self._cache_file.exists():
                self._cache_file.unlink()
                logger.info("Cleared document cache")

            # Clear only embedding stage fingerprint
            self.cache_manager.clear_stage_fingerprint("embedding")

            logger.info("Embedding caches reset complete")

        except Exception as e:
            logger.error(f"Error resetting embedding caches: {e}")
            raise ConfigurationError(f"Failed to reset embedding caches: {e}")

    async def _resummarize_existing_documents(self) -> None:
        """Re-summarize existing documents with new summarizer configuration.

        This method only generates new summaries using cached parsing results,
        without re-parsing, re-chunking, or re-embedding documents.
        """
        try:
            logger.info("Re-summarizing existing documents with new summarizer configuration...")

            # Skip if no summarizer service is available
            if not self.document_processor or not self.document_processor.summarizer_service:
                logger.info("No summarizer service available, skipping re-summarization")
                return

            # Ensure document cache is synchronized with vector store before re-summarization
            logger.info("Synchronizing document cache with vector store before re-summarization...")
            await self._synchronize_document_cache_with_vector_store()

            # Get all documents directly from the vector store (not from cache)
            documents_data = await self.vector_store.list_documents()
            if not documents_data:
                logger.info("No documents found to re-summarize")
                return

            logger.info(f"Found {len(documents_data)} documents to re-summarize")
            resummarized_count = 0

            for doc_data in documents_data:
                try:
                    doc_id = doc_data["id"]
                    doc_title = doc_data["title"]
                    doc_path = doc_data["path"]
                    logger.info(f"Processing document for re-summarization: {doc_id} ({doc_title}) at {doc_path}")

                    # Skip if we don't have a valid path
                    if not doc_path or doc_path == "":
                        logger.warning(f"Skipping document {doc_id} - no valid path")
                        continue

                    file_path = Path(doc_path)

                    # Check if file still exists
                    if not file_path.exists():
                        logger.warning(f"Skipping document {doc_id} - file no longer exists: {doc_path}")
                        continue

                    # Try to load cached parsing result instead of re-parsing
                    logger.info(f"Loading cached parsing result for: {file_path}")
                    cached_parse_result = await self._load_cached_parsing_result(file_path)

                    if not cached_parse_result:
                        logger.warning(f"No cached parsing result found for {doc_path}, skipping summarization")
                        continue

                    logger.info(f"Found cached parsing result, generating summary for: {file_path.name}")
                    # Generate new summary using cached content
                    summary_data = await self.document_processor._generate_document_summary(
                        cached_parse_result, file_path.name
                    )

                    if summary_data:
                        logger.info(f"Summary generated successfully for {doc_title}")

                        # Find the document in cache by path (since IDs might not match)
                        cached_document = None
                        cached_doc_id = None

                        # First try by document ID
                        if doc_id in self._document_cache:
                            cached_document = self._document_cache[doc_id]
                            cached_doc_id = doc_id
                            logger.debug(f"Found document by ID: {doc_id}")
                        else:
                            # Try to find by path if ID doesn't match
                            for cache_doc_id, cache_doc in self._document_cache.items():
                                if cache_doc.path == doc_path:
                                    cached_document = cache_doc
                                    cached_doc_id = cache_doc_id
                                    logger.info(f"Found document by path match: {doc_path} -> {cache_doc_id}")
                                    break

                        if cached_document and cached_doc_id:
                            old_title = cached_document.title
                            # Update summary metadata
                            if summary_data.title and len(summary_data.title) > 5:
                                cached_document.title = summary_data.title
                                logger.info(f"Updated title from '{old_title}' to '{cached_document.title}'")
                            cached_document.metadata.update(
                                {
                                    "short_description": summary_data.short_description,
                                    "long_description": summary_data.long_description,
                                    "summary_generated": True,
                                }
                            )
                            resummarized_count += 1
                            logger.info(f"Successfully re-summarized document: {doc_title} (cache ID: {cached_doc_id})")
                        else:
                            logger.warning(f"Document not found in cache by ID ({doc_id}) or path ({doc_path})")
                            logger.debug(f"Available cache IDs: {list(self._document_cache.keys())}")
                    else:
                        logger.warning(f"No summary generated for document: {doc_title}")

                except Exception as e:
                    logger.error(f"Failed to re-summarize document {doc_data.get('id', 'unknown')}: {e}")
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")

            # Save the updated document cache if we have any documents there
            if resummarized_count > 0 and self._document_cache:
                await self._save_document_cache()

            logger.info(f"Completed re-summarizing {resummarized_count} out of {len(documents_data)} documents")

        except Exception as e:
            logger.error(f"Error re-summarizing documents: {e}")
            # Don't raise - let system continue

    async def _load_cached_parsing_result(self, file_path: Path):
        """Load cached parsing result for a file without re-parsing.

        For PDF files, loads from cache. For Markdown files, re-parses on the fly
        since Markdown parsing is fast and not cached during normal processing.

        Args:
            file_path: Path to the document file.

        Returns:
            ParseResult if cached result exists or can be generated, None otherwise.
        """
        try:
            logger.debug(f"Attempting to load cached parsing result for: {file_path}")

            # Calculate file checksum
            checksum = await self.document_processor._calculate_checksum(file_path)
            logger.debug(f"Calculated checksum for {file_path}: {checksum}")

            # Handle different file types
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                # For PDF files, try to load from cache
                parse_result = await self.document_processor._load_parsing_result(file_path, checksum)

                if parse_result:
                    logger.info(f"Successfully loaded cached parsing result for PDF: {file_path}")
                    page_count = len(parse_result.pages) if hasattr(parse_result, "pages") else "unknown"
                    logger.debug(f"Parse result has {page_count} pages")
                    return parse_result
                else:
                    logger.warning(f"No cached parsing result found for PDF: {file_path}")
                    return None

            elif suffix in [".md", ".markdown"]:
                # For Markdown files, re-parse on the fly since parsing is fast
                logger.info(f"Re-parsing Markdown file for summarization: {file_path}")

                # Import MarkdownParser and create parse result
                from .parsers.parser_markdown import MarkdownParser

                markdown_config = {
                    "parse_frontmatter": getattr(self.config, "markdown_parse_frontmatter", True),
                    "extract_title": getattr(self.config, "markdown_extract_title", True),
                    "page_boundary_pattern": getattr(
                        self.config, "markdown_page_boundary_pattern", r"--\[PAGE:\s*(\d+)\]--"
                    ),
                    "split_on_page_boundaries": getattr(self.config, "markdown_split_on_page_boundaries", True),
                }

                markdown_parser = MarkdownParser(config=markdown_config)
                parse_result = await markdown_parser.parse(file_path)

                if parse_result:
                    logger.info(f"Successfully re-parsed Markdown file for summarization: {file_path}")
                    page_count = len(parse_result.pages) if hasattr(parse_result, "pages") else "unknown"
                    logger.debug(f"Parse result has {page_count} pages")
                    return parse_result
                else:
                    logger.warning(f"Failed to re-parse Markdown file: {file_path}")
                    return None

            else:
                logger.warning(f"Unsupported file type for re-summarization: {suffix}")
                return None

        except Exception as e:
            logger.error(f"Failed to load cached parsing result for {file_path}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def _handle_post_initialization_reprocessing(self) -> None:
        """Handle re-processing of cached documents after components are initialized."""
        try:
            # Check if we detected config changes that require re-processing
            if not hasattr(self, "_config_changes") or not self._config_changes:
                return

            changes = self._config_changes

            # Only re-process if chunking or embedding changed (parsing changes do full reset)
            if changes.get("chunking") or changes.get("embedding"):
                await self._reprocess_existing_documents()

            # Re-summarize documents if summarizer configuration changed
            if changes.get("summarizer"):
                await self._resummarize_existing_documents()

            # Clear the stored changes
            self._config_changes = None

        except Exception as e:
            logger.error(f"Error in post-initialization re-processing: {e}")
            # Don't raise - let system continue

    def _setup_tools(self) -> None:
        """Set up MCP tools."""

        @self.app.tool()
        async def add_document(path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Add a document (PDF or Markdown) to the knowledgebase for searching and analysis.

            Use this tool to ingest new documents. Supports both PDF and Markdown (.md, .markdown) files.
            Once added, the document content will be automatically processed, chunked, and made searchable
            via search_documents. You do not need to call any other tools after adding - the document
            becomes immediately available for searching.

            Args:
                path: Path to the document file (PDF or Markdown) to add to the knowledgebase.
                metadata: Optional metadata to associate with the document (e.g., tags, categories).

            Returns:
                Processing result with document information, success status, and processing time.
            """
            try:
                # Validate input
                if not path or not path.strip():
                    raise ValidationError("Path cannot be empty", "path")

                file_path = Path(path)
                if not file_path.exists():
                    raise ValidationError(f"File does not exist: {path}", "path")

                supported_extensions = [".pdf", ".md", ".markdown"]
                if not file_path.suffix.lower() in supported_extensions:
                    raise ValidationError(
                        f"File type not supported. Must be one of {supported_extensions}: {path}", "path"
                    )

                logger.info(f"Adding document: {path}")
                start_time = time.time()

                # Process the PDF with semaphore to limit parallelism
                async with self._parsing_semaphore:
                    result = await self.document_processor.process_document(file_path, metadata)

                if not result.success:
                    logger.error(f"Failed to process PDF {path}: {result.error}")
                    return {
                        "success": False,
                        "error": result.error,
                        "processing_time": time.time() - start_time,
                    }

                # Add document to vector store
                if result.document:
                    await self.vector_store.add_document(result.document)

                    # Cache document metadata
                    self._document_cache[result.document.id] = result.document
                    await self._save_document_cache()

                    # Note: Don't call file_monitor.process_new_file() here to avoid double processing
                    # The file monitor will pick up the file through its normal scanning process

                processing_time = time.time() - start_time
                result.processing_time = processing_time

                logger.info(f"Successfully added document {path} in {processing_time:.2f}s")

                return result.to_dict()

            except ValidationError as e:
                logger.error(f"Validation error adding document: {e}")
                return {"success": False, "error": str(e)}
            except PDFProcessingError as e:
                logger.error(f"PDF processing error: {e}")
                return {"success": False, "error": str(e)}
            except VectorStoreError as e:
                logger.error(f"Vector store error: {e}")
                return {"success": False, "error": str(e)}
            except Exception as e:
                logger.error(f"Unexpected error adding document {path}: {e}")
                return {"success": False, "error": f"Unexpected error: {e}"}

        @self.app.tool()
        async def search_documents(
            query: str,
            limit: int = 5,
            metadata_filter: Optional[Dict[str, Any]] = None,
            search_type: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Search for relevant content across the entire PDF knowledgebase.

            This is the primary tool for finding information. By default, it uses hybrid search
            combining semantic similarity (vector search) with keyword matching (BM25) for optimal
            results. You do NOT need to call list_documents first - simply provide your search
            query and this tool will find the most relevant content across all PDFs.

            Args:
                query: Search query text describing what you're looking for.
                limit: Maximum number of results to return (default: 5).
                metadata_filter: Optional metadata filters to apply to narrow results.
                search_type: Optional search type - "hybrid" (default), "vector", or "text".

            Returns:
                Search results with relevant document chunks, similarity scores, and metadata.
            """
            try:
                # Validate input
                if not query or not query.strip():
                    raise ValidationError("Query cannot be empty", "query")

                if limit <= 0:
                    raise ValidationError("Limit must be positive", "limit")

                # Validate search_type
                if search_type and search_type not in ["hybrid", "vector", "text"]:
                    raise ValidationError("search_type must be 'hybrid', 'vector', or 'text'", "search_type")

                # Use hybrid by default if enabled, otherwise fall back to vector
                if search_type is None:
                    search_type = "hybrid" if self.config.enable_hybrid_search else "vector"

                logger.info(f"Searching for: {query} (limit: {limit}, type: {search_type})")
                start_time = time.time()

                # Create search query object
                search_query = SearchQuery(
                    query=query.strip(), limit=limit, metadata_filter=metadata_filter, search_type=search_type
                )

                # Generate query embedding
                query_embedding = await self.embedding_service.generate_embedding(query)
                if not query_embedding:
                    raise EmbeddingError("Failed to generate query embedding")

                # Search vector store
                search_results = await self.vector_store.search(search_query, query_embedding)

                # Format results
                results_data = []
                for result in search_results:
                    results_data.append(result.to_dict())

                search_time = time.time() - start_time

                response = {
                    "success": True,
                    "results": results_data,
                    "total_results": len(results_data),
                    "query": query,
                    "search_time": search_time,
                    "metadata": {"limit": limit, "metadata_filter": metadata_filter},
                }

                logger.info(f"Search completed: {len(results_data)} results in {search_time:.2f}s")
                return response

            except ValidationError as e:
                logger.error(f"Validation error in search: {e}")
                return {"success": False, "error": str(e)}
            except EmbeddingError as e:
                logger.error(f"Embedding error in search: {e}")
                return {"success": False, "error": str(e)}
            except VectorStoreError as e:
                logger.error(f"Vector store error in search: {e}")
                return {"success": False, "error": str(e)}
            except Exception as e:
                logger.error(f"Unexpected error in search: {e}")
                return {"success": False, "error": f"Unexpected error: {e}"}

        @self.app.tool()
        async def list_documents(
            metadata_filter: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """List all documents in the knowledgebase for management and browsing purposes.

            Use this tool ONLY when you need to:
            - Browse available documents and their metadata
            - Get document management information (file sizes, page counts, etc.)
            - Remove or manage specific documents by ID

            DO NOT use this tool before searching - use search_documents directly instead,
            as it automatically searches across all documents without requiring a list first.

            Args:
                metadata_filter: Optional metadata filters to apply.

            Returns:
                List of document metadata and statistics (titles, paths, page counts, etc.).
            """
            try:
                logger.info("Listing documents")
                start_time = time.time()

                # Get all documents from cache
                documents = list(self._document_cache.values())

                # If cache is empty, try to populate from vector store
                if not documents:
                    await self._populate_document_cache_from_vector_store()
                    documents = list(self._document_cache.values())

                # Apply metadata filter if provided
                if metadata_filter:
                    filtered_docs = []
                    for doc in documents:
                        # Simple metadata filtering
                        matches = True
                        for key, value in metadata_filter.items():
                            if key not in doc.metadata or doc.metadata[key] != value:
                                matches = False
                                break
                        if matches:
                            filtered_docs.append(doc)
                    documents = filtered_docs

                # Format document list
                document_list = []
                for doc in documents:
                    document_list.append(doc.to_dict(include_chunks=False))

                # Get vector store statistics
                total_chunks = await self.vector_store.get_chunk_count()

                list_time = time.time() - start_time

                response = {
                    "success": True,
                    "documents": document_list,
                    "total_count": len(document_list),
                    "total_chunks": total_chunks,
                    "list_time": list_time,
                    "metadata": {"metadata_filter": metadata_filter},
                }

                logger.info(f"Listed {len(document_list)} documents in {list_time:.2f}s")
                return response

            except Exception as e:
                logger.error(f"Error listing documents: {e}")
                return {"success": False, "error": f"Error listing documents: {e}"}

        def _is_file_watcher_managed_document(self, document: Document) -> bool:
            """Check if a document is managed by the file watcher (exists in knowledgebase directory).

            Args:
                document: Document to check

            Returns:
                True if the document is managed by file watcher and should not be removed via API
            """
            try:
                if not document.path:
                    return False

                doc_path = Path(document.path).resolve()
                kb_path = self.config.knowledgebase_path.resolve()
                uploads_path = (kb_path / "uploads").resolve()

                # Check if document is within knowledgebase directory
                try:
                    doc_path.relative_to(kb_path)
                except ValueError:
                    # Document path is not within knowledgebase directory
                    return False

                # Check if document is NOT in uploads directory (uploads are user-managed)
                try:
                    doc_path.relative_to(uploads_path)
                    # Document is in uploads directory, so it's user-managed
                    return False
                except ValueError:
                    # Document is not in uploads, so it could be file-watcher-managed
                    pass

                # Check if the original file still exists
                if doc_path.exists():
                    return True

                return False

            except Exception as e:
                logger.error(f"Error checking if document is file-watcher-managed: {e}")
                return False

        @self.app.tool()
        async def rescan_documents() -> Dict[str, Any]:
            """Manually trigger a directory rescan to detect new, modified, or deleted documents.

            This tool is useful when:
            - Files were added/removed outside of normal monitoring
            - The automatic monitoring missed changes (e.g., in containerized environments)
            - You want to force a complete refresh of the document index

            The rescan will:
            1. Scan the documents directory for all supported files (.pdf, .md, .markdown)
            2. Compare with the internal file index to detect changes
            3. Process new and modified files
            4. Remove deleted files from the knowledgebase
            5. Return detailed statistics about the operation

            Returns:
                Detailed scan results including files processed, errors, and timing.
            """
            try:
                if not self.config.enable_manual_rescan:
                    return {
                        "success": False,
                        "error": "Manual rescan is disabled. Set PDFKB_ENABLE_MANUAL_RESCAN=true to enable.",
                    }

                if not hasattr(self, "file_monitor") or not self.file_monitor:
                    return {"success": False, "error": "File monitor is not available"}

                logger.info("🔄 Manual document rescan requested via MCP")

                # Perform the manual rescan
                result = await self.file_monitor.manual_rescan()

                # Update our document cache after rescan
                await self._populate_document_cache_from_vector_store()

                # Add success flag and format response
                result["success"] = True
                result["message"] = (
                    f"Rescan completed: {result['changes_processed']['new_files_processed']} new, "
                    f"{result['changes_processed']['modified_files_processed']} modified, "
                    f"{result['changes_processed']['deleted_files_processed']} deleted files processed"
                )

                logger.info(f"🔄 Manual document rescan completed: {result['message']}")

                return result

            except Exception as e:
                logger.error(f"Manual document rescan failed: {e}")
                return {"success": False, "error": f"Rescan failed: {e}"}

        @self.app.tool()
        async def remove_document(document_id: str) -> Dict[str, Any]:
            """Remove a specific document from the knowledgebase.

            Use this tool to permanently delete a document and all its associated data.
            If you need to find the document ID first, use list_documents to browse
            available documents and get their IDs. The document will be completely
            removed from search results after deletion.

            Args:
                document_id: Unique ID of the document to remove (get this from list_documents).

            Returns:
                Removal confirmation with document details or error information.
            """
            try:
                # Validate input
                if not document_id or not document_id.strip():
                    raise ValidationError("Document ID cannot be empty", "document_id")

                document_id = document_id.strip()

                # Check if document exists
                if document_id not in self._document_cache:
                    raise DocumentNotFoundError(document_id)

                document = self._document_cache[document_id]

                # Check if document is managed by file watcher
                if self._is_file_watcher_managed_document(document):
                    error_msg = (
                        f"Cannot remove document '{document.filename or document_id}' as it exists in the "
                        f"knowledgebase directory ({document.path}). To remove this document, delete the "
                        f"file from the filesystem directly."
                    )
                    logger.warning(f"Attempted to remove file-watcher-managed document: {document_id}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "error_type": "file_watcher_managed",
                        "document_path": document.path,
                    }

                logger.info(f"Removing document: {document_id}")
                start_time = time.time()

                # Remove from vector store
                await self.vector_store.delete_document(document_id)

                # For user-uploaded documents, also remove the physical file
                document_path = document.path
                try:
                    file_path = Path(document_path)
                    uploads_dir = self.config.knowledgebase_path / "uploads"

                    # Check if this is an uploaded file (in uploads directory)
                    try:
                        file_path.relative_to(uploads_dir)
                        # It's in uploads directory, safe to delete
                        if file_path.exists():
                            file_path.unlink()
                            logger.info(f"Deleted uploaded file: {file_path}")
                        else:
                            logger.warning(f"Uploaded file not found for deletion: {file_path}")
                    except ValueError:
                        # File is not in uploads directory, don't delete it
                        logger.debug(f"Document file not in uploads directory, preserving: {file_path}")

                except Exception as e:
                    logger.warning(f"Failed to delete uploaded file {document_path}: {e}")

                # Remove from document cache
                del self._document_cache[document_id]
                await self._save_document_cache()

                removal_time = time.time() - start_time

                response = {
                    "success": True,
                    "document_id": document_id,
                    "document_path": document.path,
                    "removal_time": removal_time,
                    "message": f"Document {document_id} removed successfully",
                }

                logger.info(f"Successfully removed document {document_id} in {removal_time:.2f}s")
                return response

            except ValidationError as e:
                logger.error(f"Validation error removing document: {e}")
                return {"success": False, "error": str(e)}
            except DocumentNotFoundError as e:
                logger.error(f"Document not found: {e}")
                return {"success": False, "error": str(e)}
            except VectorStoreError as e:
                logger.error(f"Vector store error removing document: {e}")
                return {"success": False, "error": str(e)}
            except Exception as e:
                logger.error(f"Unexpected error removing document {document_id}: {e}")
                return {"success": False, "error": f"Unexpected error: {e}"}

    async def _resolve_document_identifier(self, document_identifier: str) -> Optional[str]:
        """Resolve a document identifier to an internal document ID.

        Args:
            document_identifier: Either an internal document ID (doc_xxxxx) or a file path

        Returns:
            Internal document ID if found, None otherwise
        """
        # If it looks like an internal ID, check if it exists directly
        if document_identifier.startswith("doc_"):
            if document_identifier in self._document_cache:
                return document_identifier
            return None

        # Otherwise, treat as a file path and try to find by path
        return await self._find_document_by_path(document_identifier)

    async def _find_document_by_path(self, file_path: str) -> Optional[str]:
        """Find a document by its file path.

        Args:
            file_path: File path to search for (can be absolute or relative)

        Returns:
            Internal document ID if found, None otherwise
        """
        # Normalize the path
        path_obj = Path(file_path)

        # If not absolute, resolve against knowledgebase path
        if not path_obj.is_absolute():
            path_obj = self.config.knowledgebase_path / path_obj

        # Convert to string for comparison
        normalized_path = str(path_obj.resolve())

        # Search through document cache
        for doc_id, document in self._document_cache.items():
            # Compare resolved paths
            try:
                doc_path = Path(document.path).resolve()
                if str(doc_path) == normalized_path:
                    return doc_id
            except Exception:
                # If path resolution fails, try direct string comparison
                if document.path == file_path or document.path == str(path_obj):
                    return doc_id

        return None

    def _setup_resources(self) -> None:
        """Set up MCP resources."""

        @self.app.resource("doc://{document_identifier}")
        async def get_document(document_identifier: str) -> str:
            """Get a document (PDF or Markdown) by ID or file path.

            Args:
                document_identifier: Either an internal document ID (e.g., 'doc_4939b2617e65034a')
                                   or a file path (e.g., '/app/documents/a121.md' or 'a121.md')

            Returns:
                Document content as JSON string.
            """
            try:
                # Resolve the identifier to an internal document ID
                document_id = await self._resolve_document_identifier(document_identifier)

                if not document_id:
                    return json.dumps(
                        {
                            "error": f"Document not found: {document_identifier}",
                            "suggestion": "Use doc://list to see all available documents",
                            "identifier_type": "internal_id" if document_identifier.startswith("doc_") else "file_path",
                        }
                    )

                document = self._document_cache[document_id]
                document_data = document.to_dict(include_chunks=True)

                logger.info(f"Retrieved document: {document_identifier} -> {document_id}")
                return json.dumps(document_data, indent=2)

            except Exception as e:
                logger.error(f"Error retrieving document {document_identifier}: {e}")
                return json.dumps(
                    {
                        "error": f"Error retrieving document: {e}",
                        "suggestion": "Use doc://list to see all available documents",
                    }
                )

        @self.app.resource("doc://{document_identifier}/chunk/{chunk_indices}")
        async def get_document_chunks(document_identifier: str, chunk_indices: str) -> str:
            """Get specific chunks of a document by chunk index.

            Args:
                document_identifier: Either an internal document ID (e.g., 'doc_4939b2617e65034a')
                                   or a file path (e.g., '/app/documents/a121.md' or 'a121.md')
                chunk_indices: Chunk index or comma-separated indices (e.g., '0', '1,2,5', '0,3,4,7')

            Returns:
                Chunk content as JSON or plain text for single chunks.
            """
            try:
                # Resolve the identifier to an internal document ID
                document_id = await self._resolve_document_identifier(document_identifier)

                if not document_id:
                    return json.dumps(
                        {
                            "error": f"Document not found: {document_identifier}",
                            "suggestion": "Use doc://list to see all available documents",
                            "identifier_type": "internal_id" if document_identifier.startswith("doc_") else "file_path",
                        }
                    )

                document = self._document_cache[document_id]

                # Parse chunk indices
                try:
                    requested_indices = [int(idx.strip()) for idx in chunk_indices.split(",") if idx.strip().isdigit()]
                    if not requested_indices:
                        raise ValueError("No valid chunk indices provided")
                except ValueError as e:
                    return json.dumps(
                        {
                            "error": f"Invalid chunk indices: {chunk_indices}",
                            "suggestion": "Use comma-separated integers like '0', '1,2,5', or '0,3,4,7'",
                            "details": str(e),
                        }
                    )

                # Get chunks - first try from document, then from vector store if needed
                chunks_to_search = document.chunks
                if not chunks_to_search:
                    try:
                        chunks_to_search = await self.vector_store.get_document_chunks(document_id)
                        logger.info(
                            f"Fetched {len(chunks_to_search)} chunks from vector store for document {document_id}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to fetch chunks from vector store for {document_id}: {e}")
                        chunks_to_search = []

                if not chunks_to_search:
                    return json.dumps(
                        {
                            "error": f"No chunks found for document {document_identifier}",
                            "suggestion": "Document may not be processed yet or may have processing errors",
                        }
                    )

                # Create a mapping of chunk_index to chunk for efficient lookup
                chunk_map = {chunk.chunk_index: chunk for chunk in chunks_to_search}
                max_chunk_index = max(chunk_map.keys()) if chunk_map else -1

                # Find requested chunks
                found_chunks = []
                missing_indices = []
                for idx in requested_indices:
                    if idx in chunk_map:
                        found_chunks.append(chunk_map[idx])
                    else:
                        missing_indices.append(idx)

                if not found_chunks:
                    return json.dumps(
                        {
                            "error": f"No chunks found for indices: {requested_indices}",
                            "available_chunk_count": len(chunks_to_search),
                            "available_chunk_indices": sorted(chunk_map.keys()),
                            "max_chunk_index": max_chunk_index,
                            "suggestion": f"Use chunk indices between 0 and {max_chunk_index}",
                        }
                    )

                # Sort chunks by chunk_index for consistent ordering
                found_chunks.sort(key=lambda c: c.chunk_index)

                # Return format depends on whether single or multiple chunks requested
                if len(requested_indices) == 1 and len(found_chunks) == 1:
                    # Single chunk - return plain text
                    chunk = found_chunks[0]
                    logger.info(f"Retrieved chunk {chunk.chunk_index} from document {document_identifier}")
                    return chunk.text
                else:
                    # Multiple chunks - return structured JSON
                    result = {
                        "document_id": document_id,
                        "document_identifier": document_identifier,
                        "requested_indices": requested_indices,
                        "found_chunks": [
                            {
                                "chunk_index": chunk.chunk_index,
                                "chunk_id": chunk.id,
                                "text": chunk.text,
                                "page_number": chunk.page_number,
                                "metadata": chunk.metadata,
                            }
                            for chunk in found_chunks
                        ],
                        "total_found": len(found_chunks),
                    }

                    if missing_indices:
                        result["missing_indices"] = missing_indices
                        result["warning"] = f"Some requested chunks were not found: {missing_indices}"

                    logger.info(
                        f"Retrieved {len(found_chunks)} chunks (indices: {[c.chunk_index for c in found_chunks]}) "
                        f"from document {document_identifier}"
                    )
                    return json.dumps(result, indent=2)

            except Exception as e:
                logger.error(f"Error retrieving chunks {chunk_indices} from document {document_identifier}: {e}")
                return json.dumps(
                    {
                        "error": f"Error retrieving chunks: {e}",
                        "suggestion": "Use doc://list to see all available documents",
                    }
                )

        @self.app.resource("doc://list")
        async def list_all_documents() -> str:
            """List all available documents (PDFs and Markdown files).

            Returns:
                JSON string with document list and metadata.
            """
            try:
                documents = list(self._document_cache.values())

                # If cache is empty, try to populate from vector store
                if not documents:
                    await self._populate_document_cache_from_vector_store()
                    documents = list(self._document_cache.values())

                # Create summary list
                document_summaries = []
                for doc in documents:
                    summary = {
                        "id": doc.id,
                        "title": doc.title or doc.filename,
                        "path": doc.path,
                        "page_count": doc.page_count,
                        "chunk_count": doc.chunk_count,
                        "file_size": doc.file_size,
                        "added_at": doc.added_at.isoformat() if doc.added_at else None,
                        "has_embeddings": doc.has_embeddings,
                    }
                    document_summaries.append(summary)

                # Get additional statistics
                total_chunks = await self.vector_store.get_chunk_count()

                response = {
                    "documents": document_summaries,
                    "total_documents": len(document_summaries),
                    "total_chunks": total_chunks,
                    "knowledgebase_path": str(self.config.knowledgebase_path),
                    "cache_dir": str(self.config.cache_dir),
                }

                logger.info(f"Listed {len(document_summaries)} documents via resource")
                return json.dumps(response, indent=2)

            except Exception as e:
                logger.error(f"Error listing documents via resource: {e}")
                return json.dumps({"error": f"Error listing documents: {e}"})

    async def _on_file_change(self, file_path: Path, change_type: str) -> None:
        """Handle file system changes.

        Args:
            file_path: Path to the changed file.
            change_type: Type of change ('added', 'modified', 'deleted').
        """
        try:
            logger.info(f"File {change_type}: {file_path}")

            if change_type == "added" or change_type == "modified":
                # Automatically process new or modified PDFs
                if file_path.suffix.lower() == ".pdf":
                    logger.info(f"Auto-processing {change_type} PDF: {file_path}")

                    # Process the PDF
                    result = await self.document_processor.process_document(file_path)

                    if result.success and result.document:
                        # Update vector store
                        await self.vector_store.add_document(result.document)

                        # Update document cache
                        self._document_cache[result.document.id] = result.document
                        await self._save_document_cache()

                        logger.info(f"Successfully auto-processed: {file_path}")
                    else:
                        logger.error(f"Failed to auto-process {file_path}: {result.error}")

            elif change_type == "deleted":
                # Find and remove document from cache
                doc_to_remove = None
                for doc_id, doc in self._document_cache.items():
                    if doc.path == str(file_path):
                        doc_to_remove = doc_id
                        break

                if doc_to_remove:
                    await self.vector_store.delete_document(doc_to_remove)
                    del self._document_cache[doc_to_remove]
                    await self._save_document_cache()
                    logger.info(f"Removed deleted document: {file_path}")

        except Exception as e:
            logger.error(f"Error handling file change {file_path}: {e}")

    async def _load_document_cache(self) -> None:
        """Load document metadata from cache file."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                self._document_cache = {}
                for doc_id, doc_data in cache_data.items():
                    try:
                        document = Document.from_dict(doc_data)
                        self._document_cache[doc_id] = document
                    except Exception as e:
                        logger.warning(f"Failed to load document {doc_id} from cache: {e}")

                logger.info(f"Loaded {len(self._document_cache)} documents from cache")
            else:
                self._document_cache = {}
                logger.info("No document cache found, starting fresh")
                # Try to populate from vector store
                await self._populate_document_cache_from_vector_store()

        except Exception as e:
            logger.error(f"Failed to load document cache: {e}")
            self._document_cache = {}
            # Try to populate from vector store even if cache loading failed
            await self._populate_document_cache_from_vector_store()

    async def _populate_document_cache_from_vector_store(self) -> None:
        """Populate document cache from vector store when cache is empty."""
        try:
            # Only populate if cache is empty
            if not self._document_cache:
                logger.info("Populating document cache from vector store...")
                documents_info = await self.vector_store.list_documents()

                # Create minimal Document objects from vector store info
                for doc_info in documents_info:
                    # Create a minimal document with basic info
                    doc = Document(id=doc_info["id"], path=doc_info["path"], title=doc_info["title"])
                    self._document_cache[doc.id] = doc

                logger.info(f"Populated document cache with {len(self._document_cache)} documents from vector store")
        except Exception as e:
            logger.error(f"Failed to populate document cache from vector store: {e}")

    async def _synchronize_document_cache_with_vector_store(self) -> None:
        """Synchronize document cache with vector store to ensure consistency.

        This method ensures that all documents in the vector store are also present
        in the document cache, adding missing ones with basic information.
        """
        try:
            logger.debug("Synchronizing document cache with vector store...")

            # Get all documents from vector store
            vector_store_docs = await self.vector_store.list_documents()
            vector_store_ids = {doc["id"] for doc in vector_store_docs}

            # Get all document IDs from cache
            cache_ids = set(self._document_cache.keys())

            # Find missing documents (in vector store but not in cache)
            missing_ids = vector_store_ids - cache_ids

            if missing_ids:
                logger.info(f"Found {len(missing_ids)} documents in vector store that are missing from cache")

                # Add missing documents to cache
                for doc_info in vector_store_docs:
                    if doc_info["id"] in missing_ids:
                        try:
                            # Create document with available info from vector store
                            doc = Document(
                                id=doc_info["id"],
                                path=doc_info["path"],
                                title=doc_info["title"],
                                # Add other fields if available
                                checksum=doc_info.get("checksum", ""),
                                file_size=doc_info.get("file_size", 0),
                                page_count=doc_info.get("page_count", 0),
                                metadata=doc_info.get("metadata", {}),
                            )
                            self._document_cache[doc.id] = doc
                            logger.debug(f"Added missing document to cache: {doc.id} ({doc.title})")
                        except Exception as e:
                            logger.warning(f"Failed to add document {doc_info['id']} to cache: {e}")

                # Save updated cache
                await self._save_document_cache()
                logger.info(f"Synchronized document cache: added {len(missing_ids)} missing documents")
            else:
                logger.debug("Document cache is already synchronized with vector store")

            # Also check for documents in cache but not in vector store (cleanup)
            orphaned_ids = cache_ids - vector_store_ids
            if orphaned_ids:
                logger.warning(f"Found {len(orphaned_ids)} documents in cache that are not in vector store")
                for orphaned_id in orphaned_ids:
                    logger.debug(f"Removing orphaned document from cache: {orphaned_id}")
                    del self._document_cache[orphaned_id]

                # Save cleaned cache
                await self._save_document_cache()
                logger.info(f"Cleaned document cache: removed {len(orphaned_ids)} orphaned documents")

        except Exception as e:
            logger.error(f"Failed to synchronize document cache with vector store: {e}")

    async def _update_document_cache(self, document: Document) -> None:
        """Callback function to update the document cache when file monitor processes a document.

        Args:
            document: Document to add to cache.
        """
        try:
            self._document_cache[document.id] = document
            await self._save_document_cache()
            logger.debug(f"Updated document cache with document {document.id}")
        except Exception as e:
            logger.error(f"Failed to update document cache: {e}")

    async def _save_document_cache(self) -> None:
        """Save document metadata to cache file."""
        try:
            # Ensure metadata directory exists
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)

            cache_data = {}
            for doc_id, document in self._document_cache.items():
                cache_data[doc_id] = document.to_dict(include_chunks=False)

            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"Saved {len(cache_data)} documents to cache")

        except Exception as e:
            logger.error(f"Failed to save document cache: {e}")

    def get_http_app(self, path: str = "/"):
        """Get the FastMCP HTTP ASGI application for integration into FastAPI.

        Args:
            path: The internal path prefix for FastMCP routes (default: "/").
                  When mounting in FastAPI, use path="/" and let FastAPI handle
                  the mount prefix to avoid double path prefixes.

        Returns:
            ASGI application instance that can be mounted in FastAPI
        """
        return self.app.http_app(path=path)

    async def run(self) -> None:
        """Run the MCP server."""
        await self.initialize()

        if self.config.transport in ["http", "sse"]:
            transport_name = "HTTP" if self.config.transport == "http" else "SSE"
            logger.info(
                f"Running MCP server in {transport_name} mode on {self.config.server_host}:{self.config.server_port}"
            )

            await self.app.run_http_async(
                transport=self.config.transport,
                host=self.config.server_host,
                port=self.config.server_port,
                show_banner=True,
            )
        else:
            logger.info("Running MCP server in stdio mode")
            # Use run_async() instead of run() to work within existing event loop
            await self.app.run_async()

    def sync_run(self) -> None:
        """Synchronous wrapper for run() - runs in a separate thread with its own event loop."""
        logger.info("Starting MCP sync_run in thread")
        try:
            asyncio.run(self.run())
        except Exception as e:
            logger.error(f"MCP sync_run error: {e}")
            raise
        finally:
            logger.info("MCP sync_run completed in thread")

    async def shutdown(self) -> None:
        """Shutdown the server gracefully."""
        try:
            logger.info("Shutting down PDF Knowledgebase server...")

            if self.file_monitor:
                await self.file_monitor.stop_monitoring()

            if self.vector_store:
                await self.vector_store.close()

            # Save document cache
            await self._save_document_cache()

            logger.info("PDF Knowledgebase server shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """Entry point for the MCP server."""
    import argparse
    import signal
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PDF Knowledgebase MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  OPENAI_API_KEY          OpenAI API key (required for OpenAI embeddings)
  PDFKB_KNOWLEDGEBASE_PATH Path to PDF directory (default: ./documents)
  PDFKB_CACHE_DIR         Cache directory (default: <KNOWLEDGEBASE_PATH>/.cache)
  PDFKB_WEB_ENABLE        Enable web interface (true/false, default: false)
  PDFKB_WEB_PORT          Unified server port (default: 8000)
  PDFKB_WEB_HOST          Server host (default: localhost)
  PDFKB_PDF_PARSER        PDF parser to use (default: pymupdf4llm)
  PDFKB_DOCUMENT_CHUNKER  Text chunker to use (default: langchain)
  PDFKB_LOG_LEVEL         Logging level (default: INFO)

Examples:
  pdfkb-mcp                            # Run MCP-only, stdio transport
  pdfkb-mcp --transport http           # Run with HTTP transport (for Cline, modern clients)
  pdfkb-mcp --transport sse            # Run with SSE transport (for Roo, legacy clients)
  PDFKB_WEB_ENABLE=true pdfkb-mcp      # Run with web interface enabled
  pdfkb-mcp --enable-web               # Run unified server (web + MCP)
  pdfkb-mcp --config myconfig.env      # Use custom config file

Endpoints:
  Unified Mode (PDFKB_WEB_ENABLE=true):
    Web interface:  http://localhost:8000/
    MCP (HTTP):     http://localhost:8000/mcp/
    MCP (SSE):      http://localhost:8000/sse/
    API docs:       http://localhost:8000/docs

  MCP-only Mode (stdio transport is used by MCP clients like Claude Desktop)
        """,
    )

    parser.add_argument("--config", type=str, help="Path to environment configuration file")

    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="MCP transport mode (default: stdio, use http/sse for remote connections)",
    )

    parser.add_argument("--enable-web", action="store_true", help="Enable web interface")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Override logging level")

    parser.add_argument("--version", action="version", version=f'pdfkb-mcp {__import__("pdfkb").__version__}')

    args = parser.parse_args()

    # Load configuration from custom file if specified
    if args.config:
        from dotenv import load_dotenv

        load_dotenv(args.config, override=True)
        logger.info(f"Loaded configuration from: {args.config}")

    # Load main configuration
    config = ServerConfig.from_env()

    # Log full configuration for debugging
    logger.info("Loaded configuration details:")
    for key, value in config.__dict__.items():
        if isinstance(value, Path):
            logger.info(f"  {key}: {value}")
        elif isinstance(value, (list, dict)):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")

    # Override configuration from command line arguments
    if args.enable_web:
        config.web_enabled = True
    if args.transport:
        config.transport = args.transport

    if args.log_level:
        config.log_level = args.log_level

    # Configure logging with the configured level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info(f"Configuration: {config.knowledgebase_path}")
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info(f"Web interface: {'enabled' if config.web_enabled else 'disabled'}")

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # This will cause KeyboardInterrupt to be raised in the event loop
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if config.web_enabled:
            # Run integrated server (MCP + Web)
            logger.info("Running in unified server mode (MCP + Web via Hypercorn)")
            logger.info(f"Web interface will be available at: http://{config.web_host}:{config.web_port}")
            if config.transport in ["http", "sse"]:
                endpoint = "mcp" if config.transport == "http" else "sse"
                logger.info(
                    f"MCP endpoints will be available at: http://{config.web_host}:{config.web_port}/{endpoint}/"
                )
            logger.info(f"API documentation will be available at: http://{config.web_host}:{config.web_port}/docs")

            # Import here to avoid circular imports and ensure web dependencies are only required when needed
            from .web_server import IntegratedPDFKnowledgebaseServer

            integrated_server = IntegratedPDFKnowledgebaseServer(config)
            asyncio.run(integrated_server.run_integrated())
        else:
            # Run MCP server only
            logger.info("Running in MCP-only mode")
            server = PDFKnowledgebaseServer(config)
            asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
