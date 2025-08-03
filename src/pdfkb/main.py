"""Main MCP server implementation using FastMCP."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

logger = logging.getLogger(__name__)


class PDFKnowledgebaseServer:
    """Main MCP server implementation for PDF knowledgebase management."""

    def __init__(self, config: Optional[ServerConfig] = None):
        """Initialize the PDF knowledgebase server.

        Args:
            config: Server configuration. If None, loads from environment.
        """
        self.config = config or ServerConfig.from_env()
        self.app = FastMCP("PDF Knowledgebase")
        self.pdf_processor: Optional[PDFProcessor] = None
        self.vector_store: Optional[VectorStore] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.file_monitor: Optional[FileMonitor] = None
        self.cache_manager: Optional[IntelligentCacheManager] = None

        # Document metadata cache
        self._document_cache: Dict[str, Document] = {}
        self._cache_file = self.config.metadata_path / "documents.json"

        self._setup_tools()
        self._setup_resources()

    async def initialize(self) -> None:
        """Initialize all components asynchronously."""
        try:
            logger.info("Initializing PDF Knowledgebase server...")

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

            # Handle re-processing of cached documents after components are initialized
            await self._handle_post_initialization_reprocessing()

            self.file_monitor = FileMonitor(
                self.config, self.pdf_processor, self.vector_store, self._update_document_cache
            )
            await self.file_monitor.start_monitoring()

            # Load document metadata cache
            await self._load_document_cache()

            # Update intelligent cache fingerprints
            self.cache_manager.update_fingerprints()

            logger.info("PDF Knowledgebase server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise ConfigurationError(f"Server initialization failed: {e}")

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
            """Add a PDF document to the knowledgebase.

            Args:
                path: Path to the PDF file to add.
                metadata: Optional metadata to associate with the document.

            Returns:
                Processing result with document information or error details.
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
            """Search for relevant content across all PDFs.

            Args:
                query: Search query text.
                limit: Maximum number of results to return.
                metadata_filter: Optional metadata filters to apply.

            Returns:
                Search results with document chunks and metadata.
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
            """List all documents in the knowledgebase.

            Args:
                metadata_filter: Optional metadata filters to apply.

            Returns:
                List of document metadata and statistics.
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

        @self.app.tool()
        async def remove_document(document_id: str) -> Dict[str, Any]:
            """Remove a document from the knowledgebase.

            Args:
                document_id: ID of the document to remove.

            Returns:
                Removal confirmation or error details.
            """
            try:
                # Validate input
                if not document_id or not document_id.strip():
                    raise ValidationError("Document ID cannot be empty", "document_id")

                document_id = document_id.strip()

                # Check if document exists
                if document_id not in self._document_cache:
                    raise DocumentNotFoundError(document_id)

                logger.info(f"Removing document: {document_id}")
                start_time = time.time()

                document = self._document_cache[document_id]

                # Remove from vector store
                await self.vector_store.delete_document(document_id)

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
    import sys

    # Load configuration to get log level
    config = ServerConfig.from_env()

    # Configure logging with the configured level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create and run server
    server = PDFKnowledgebaseServer(config)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        asyncio.run(server.shutdown())
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
