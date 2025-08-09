"""Main MCP server implementation using FastMCP."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastmcp import FastMCP

from .config import ServerConfig
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
from .pdf_processor import PDFProcessor
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
        self.pdf_processor: Optional[PDFProcessor] = None
        self.vector_store: Optional[VectorStore] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.file_monitor: Optional[FileMonitor] = None
        self.cache_manager: Optional[IntelligentCacheManager] = None
        self.background_queue = background_queue
        self._web_document_service = None  # Optional reference to web document service

        # Document metadata cache
        self._document_cache: Dict[str, Document] = {}
        self._cache_file = self.config.metadata_path / "documents.json"

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

            self.vector_store = VectorStore(self.config)
            self.vector_store.set_embedding_service(self.embedding_service)
            await self.vector_store.initialize()

            self.pdf_processor = PDFProcessor(self.config, self.embedding_service, self.cache_manager)

            # Log startup configuration summary for diagnostics
            try:
                parser_name = (
                    type(self.pdf_processor.parser).__name__ if self.pdf_processor else str(self.config.pdf_parser)
                )
                chunker_name = (
                    type(self.pdf_processor.chunker).__name__ if self.pdf_processor else str(self.config.pdf_chunker)
                )
            except Exception:
                parser_name = str(self.config.pdf_parser)
                chunker_name = str(self.config.pdf_chunker)
            logger.info(
                "Startup configuration: Parser=%s, Chunker=%s, EmbeddingModel=%s, KnowledgebasePath=%s, CacheDir=%s",
                parser_name,
                chunker_name,
                self.config.embedding_model,
                self.config.knowledgebase_path,
                self.config.cache_dir,
            )
            # Handle re-processing of cached documents after components are initialized
            await self._handle_post_initialization_reprocessing()

            # Load document metadata cache
            await self._load_document_cache()

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
                self.pdf_processor,
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
            documents = await self.pdf_processor.reprocess_cached_documents()

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

            # Clear the stored changes
            self._config_changes = None

        except Exception as e:
            logger.error(f"Error in post-initialization re-processing: {e}")
            # Don't raise - let system continue

    def _setup_tools(self) -> None:
        """Set up MCP tools."""

        @self.app.tool()
        async def add_document(path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Add a PDF document to the knowledgebase for searching and analysis.

            Use this tool to ingest new PDF documents. Once added, the document content
            will be automatically processed, chunked, and made searchable via search_documents.
            You do not need to call any other tools after adding - the document becomes
            immediately available for searching.

            Args:
                path: Path to the PDF file to add to the knowledgebase.
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

                if not file_path.suffix.lower() == ".pdf":
                    raise ValidationError(f"File is not a PDF: {path}", "path")

                logger.info(f"Adding document: {path}")
                start_time = time.time()

                # Process the PDF
                result = await self.pdf_processor.process_pdf(file_path, metadata)

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
            query: str, limit: int = 5, metadata_filter: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Search for relevant content across the entire PDF knowledgebase.

            This is the primary tool for finding information. It automatically searches through
            all documents in the knowledgebase using semantic similarity - you do NOT need to
            call list_documents first to know what documents are available. Simply provide your
            search query and this tool will find the most relevant content across all PDFs.

            Args:
                query: Search query text describing what you're looking for.
                limit: Maximum number of results to return (default: 5).
                metadata_filter: Optional metadata filters to apply to narrow results.

            Returns:
                Search results with relevant document chunks, similarity scores, and metadata.
            """
            try:
                # Validate input
                if not query or not query.strip():
                    raise ValidationError("Query cannot be empty", "query")

                if limit <= 0:
                    raise ValidationError("Limit must be positive", "limit")

                logger.info(f"Searching for: {query} (limit: {limit})")
                start_time = time.time()

                # Create search query object
                search_query = SearchQuery(query=query.strip(), limit=limit, metadata_filter=metadata_filter)

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

    def _setup_resources(self) -> None:
        """Set up MCP resources."""

        @self.app.resource("pdf://{document_id}")
        async def get_document(document_id: str) -> str:
            """Get a PDF document by ID.

            Args:
                document_id: ID of the document to retrieve.

            Returns:
                Document content as JSON string.
            """
            try:
                if document_id not in self._document_cache:
                    raise DocumentNotFoundError(document_id)

                document = self._document_cache[document_id]
                document_data = document.to_dict(include_chunks=True)

                logger.info(f"Retrieved document: {document_id}")
                return json.dumps(document_data, indent=2)

            except DocumentNotFoundError as e:
                logger.error(f"Document not found: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error retrieving document {document_id}: {e}")
                return json.dumps({"error": f"Error retrieving document: {e}"})

        @self.app.resource("pdf://{document_id}/page/{page_number}")
        async def get_document_page(document_id: str, page_number: int) -> str:
            """Get a specific page of a PDF document.

            Args:
                document_id: ID of the document.
                page_number: Page number to retrieve.

            Returns:
                Page content as text.
            """
            try:
                if document_id not in self._document_cache:
                    raise DocumentNotFoundError(document_id)

                document = self._document_cache[document_id]

                # Find chunks for the specified page
                page_chunks = [chunk for chunk in document.chunks if chunk.page_number == page_number]

                if not page_chunks:
                    return f"No content found for page {page_number} in document {document_id}"

                # Combine chunk text for the page
                page_text = "\n\n".join(chunk.text for chunk in page_chunks)

                logger.info(f"Retrieved page {page_number} from document {document_id}")
                return page_text

            except DocumentNotFoundError as e:
                logger.error(f"Document not found: {e}")
                return f"Error: {e}"
            except Exception as e:
                logger.error(f"Error retrieving page {page_number} from document {document_id}: {e}")
                return f"Error retrieving page: {e}"

        @self.app.resource("pdf://list")
        async def list_all_documents() -> str:
            """List all available PDF documents.

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
                    result = await self.pdf_processor.process_pdf(file_path)

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

    async def run(self) -> None:
        """Run the MCP server."""
        await self.initialize()

        # Use run_async() instead of run() to work within existing event loop
        await self.app.run_async()

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
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PDF Knowledgebase MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  OPENAI_API_KEY          OpenAI API key (required)
  KNOWLEDGEBASE_PATH      Path to PDF directory (default: ./pdfs)
  CACHE_DIR              Cache directory (default: <KNOWLEDGEBASE_PATH>/.cache)
  PDFKB_ENABLE_WEB       Enable web interface (1/true/yes to enable, default: true)
  WEB_PORT               Web server port (default: 8080)
  WEB_HOST               Web server host (default: localhost)
  PDF_PARSER             PDF parser to use (default: pymupdf4llm)
  PDF_CHUNKER            Text chunker to use (default: langchain)
  LOG_LEVEL              Logging level (default: INFO)

Examples:
  pdfkb-mcp                          # Run with default settings (MCP + Web if PDFKB_ENABLE_WEB=true)
  PDFKB_ENABLE_WEB=false pdfkb-mcp   # Run MCP server only
  PDFKB_ENABLE_WEB=true pdfkb-mcp    # Run MCP server with web interface
  pdfkb-mcp --config myconfig.env    # Use custom config file
        """,
    )

    parser.add_argument("--config", type=str, help="Path to environment configuration file")

    parser.add_argument("--port", type=int, help="Override MCP server port (for stdio mode, this has no effect)")

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

    # Override log level if specified
    if args.log_level:
        config.log_level = args.log_level

    # Configure logging with the configured level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting PDF Knowledgebase MCP Server")
    logger.info(f"Version: {__import__('pdfkb').__version__}")
    logger.info(f"Configuration: {config.knowledgebase_path}")
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info(f"Web interface: {'enabled' if config.web_enabled else 'disabled'}")

    try:
        if config.web_enabled:
            # Run integrated server (MCP + Web)
            logger.info("Running in integrated mode (MCP + Web)")
            if config.web_enabled:
                logger.info(f"Web interface will be available at: http://{config.web_host}:{config.web_port}")
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
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
