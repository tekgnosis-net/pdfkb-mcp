"""Vector storage operations using Chroma database."""

import logging
from typing import Any, Dict, List, Optional

from .config import ServerConfig
from .exceptions import VectorStoreError
from .models import Chunk, Document, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages Chroma vector database operations."""

    def __init__(self, config: ServerConfig):
        """Initialize the vector store.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.client = None
        self.collection = None
        self.collection_name = "pdf_knowledgebase"
        self._embedding_service = None

    def set_embedding_service(self, embedding_service) -> None:
        """Set the embedding service for query embeddings.

        Args:
            embedding_service: EmbeddingService instance.
        """
        self._embedding_service = embedding_service

    async def initialize(self) -> None:
        """Initialize the Chroma client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.config.chroma_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"description": "PDF Knowledgebase documents"}
            )
            logger.info(f"Collection '{self.collection_name}' ready")

            logger.info("Vector store initialized successfully")

        except ImportError:
            raise VectorStoreError("ChromaDB package not installed. Install with: pip install chromadb", "initialize")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {e}", "initialize", e)

    async def add_document(self, document: Document) -> None:
        """Add a document's chunks to the vector store with deduplication.

        Args:
            document: Document to add to the vector store.
        """
        try:
            if not document.chunks:
                logger.warning(f"Document {document.id} has no chunks to add")
                return

            # Filter chunks with embeddings
            chunks_with_embeddings = [c for c in document.chunks if c.has_embedding]

            if not chunks_with_embeddings:
                logger.warning(f"Document {document.id} has no chunks with embeddings")
                return

            if self.collection is None:
                await self.initialize()

            # Check for existing chunks and filter out duplicates
            new_chunks = await self._filter_existing_chunks(chunks_with_embeddings)

            if not new_chunks:
                logger.info(f"All chunks from document {document.id} already exist in vector store")
                return

            # Prepare data for Chroma
            chunk_ids = [chunk.id for chunk in new_chunks]
            embeddings = [chunk.embedding for chunk in new_chunks]
            documents = [chunk.text for chunk in new_chunks]
            metadatas = [self._prepare_chunk_metadata(chunk, document) for chunk in new_chunks]

            # Add to collection in batches to avoid memory issues
            batch_size = min(100, len(chunk_ids))  # Chroma recommends smaller batches

            for i in range(0, len(chunk_ids), batch_size):
                end_idx = min(i + batch_size, len(chunk_ids))

                self.collection.add(
                    ids=chunk_ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                )

            skipped_count = len(chunks_with_embeddings) - len(new_chunks)
            logger.info(
                f"Added {len(new_chunks)} new chunks from document {document.id} to vector store "
                f"(skipped {skipped_count} duplicates)"
            )

        except Exception as e:
            raise VectorStoreError(f"Failed to add document to vector store: {e}", "add", e)

    async def _filter_existing_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Filter out chunks that already exist in the vector store.

        Args:
            chunks: List of chunks to check.

        Returns:
            List of chunks that don't already exist in the vector store.
        """
        try:
            if not chunks:
                return []

            # Get all chunk IDs
            chunk_ids = [chunk.id for chunk in chunks]

            # Query existing chunks by ID
            existing_results = self.collection.get(ids=chunk_ids, include=["metadatas"])

            existing_ids = set(existing_results["ids"]) if existing_results["ids"] else set()

            # Filter out existing chunks
            new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]

            logger.debug(f"Filtered {len(chunks)} chunks: {len(new_chunks)} new, {len(existing_ids)} already exist")

            return new_chunks

        except Exception as e:
            logger.error(f"Error filtering existing chunks: {e}")
            # On error, return all chunks to avoid losing data
            return chunks

    async def search(self, query: SearchQuery, query_embedding: List[float]) -> List[SearchResult]:
        """Search for similar chunks using vector similarity.

        Args:
            query: Search query parameters.
            query_embedding: Query embedding vector.

        Returns:
            List of search results ordered by similarity score.
        """
        try:
            if self.collection is None:
                await self.initialize()

            # Build where clause for metadata filtering
            where_clause = self._build_where_clause(query.metadata_filter)

            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=query.limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            # Process results
            search_results = []

            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i, (chunk_id, distance, text, metadata) in enumerate(
                    zip(
                        results["ids"][0],
                        results["distances"][0],
                        results["documents"][0],
                        results["metadatas"][0],
                    )
                ):
                    # Convert distance to similarity score (0-1, higher is better)
                    # Chroma uses L2 distance, so we convert to similarity
                    score = max(0, 1 - (distance / 2))  # Normalize L2 distance to similarity

                    if score < query.min_score:
                        continue

                    # Create chunk and document from metadata
                    chunk = self._chunk_from_metadata(chunk_id, text, metadata)
                    document = self._document_from_metadata(metadata)

                    search_results.append(SearchResult(chunk=chunk, score=score, document=document))

            logger.info(f"Found {len(search_results)} results for query: {query.query[:50]}...")
            return search_results

        except Exception as e:
            raise VectorStoreError(f"Failed to search vector store: {e}", "search", e)

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document from the vector store.

        Args:
            document_id: ID of the document to delete.
        """
        try:
            if self.collection is None:
                await self.initialize()

            # Delete all chunks for this document
            self.collection.delete(where={"document_id": document_id})

            logger.info(f"Deleted document {document_id} from vector store")

        except Exception as e:
            raise VectorStoreError(f"Failed to delete document from vector store: {e}", "delete", e)

    async def remove_document(self, document_id: str) -> int:
        """Remove all chunks for a document from the vector store.

        Args:
            document_id: ID of the document to remove.

        Returns:
            Number of chunks removed.
        """
        try:
            if self.collection is None:
                await self.initialize()

            # First, get the count of chunks for this document
            try:
                existing_results = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
                chunk_count = len(existing_results["ids"]) if existing_results["ids"] else 0
            except Exception:
                chunk_count = 0

            # Delete all chunks for this document
            if chunk_count > 0:
                self.collection.delete(where={"document_id": document_id})
                logger.info(f"Removed {chunk_count} chunks for document {document_id}")
            else:
                logger.info(f"No chunks found for document {document_id}")

            return chunk_count

        except Exception as e:
            raise VectorStoreError(f"Failed to remove document from vector store: {e}", "remove", e)

    async def list_documents(self) -> List[dict]:
        """Get all indexed documents.

        Returns:
            List of document information dictionaries.
        """
        try:
            if self.collection is None:
                await self.initialize()

            # Get all metadata to extract unique documents
            results = self.collection.get(include=["metadatas"])

            # Normalize possible shapes (flat or nested lists)
            metadatas = results.get("metadatas") or []
            if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], list):
                metadatas = [m for sub in metadatas for m in sub]

            # If stub returns nothing, fall back to empty list
            if metadatas is None:
                metadatas = []

            # Group by document_id and collect document information
            documents = {}
            for metadata in metadatas:
                doc_id = metadata.get("document_id") if isinstance(metadata, dict) else None
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "id": doc_id,
                        "title": (
                            metadata.get("document_title", "Unknown") if isinstance(metadata, dict) else "Unknown"
                        ),
                        "path": (metadata.get("document_path", "") if isinstance(metadata, dict) else ""),
                        "chunk_count": 0,
                    }

                if doc_id:
                    documents[doc_id]["chunk_count"] += 1

            return list(documents.values())

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    async def get_document_content(self, document_id: str) -> Optional[str]:
        """Retrieve full document content by combining all chunks.

        Args:
            document_id: ID of the document to retrieve.

        Returns:
            Combined document content or None if not found.
        """
        try:
            if self.collection is None:
                await self.initialize()

            # Get all chunks for this document
            results = self.collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])

            if not results["documents"]:
                return None

            # Sort chunks by chunk_index and combine
            chunk_data = list(zip(results["documents"], results["metadatas"]))
            chunk_data.sort(key=lambda x: x[1].get("chunk_index", 0))

            # Combine chunk texts
            content_parts = [chunk_text for chunk_text, _ in chunk_data]
            return "\n\n".join(content_parts)

        except Exception as e:
            logger.error(f"Failed to get document content: {e}")
            return None

    async def get_document_count(self) -> int:
        """Get the total number of documents in the vector store.

        Returns:
            Number of unique documents.
        """
        try:
            if self.collection is None:
                await self.initialize()

            # Get all metadata to count unique documents
            results = self.collection.get(include=["metadatas"])

            # Handle possible shapes from different clients/stubs:
            # - results["metadatas"] can be a flat list[dict] or a nested list[[dict]]
            metadatas = results.get("metadatas") or results.get("metadatas".lower(), []) or []
            if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], list):
                # Flatten one level if nested
                metadatas = [m for sub in metadatas for m in sub]

            # Count unique document IDs with robust fallback
            document_ids = set()
            for metadata in metadatas:
                if isinstance(metadata, dict):
                    doc_id = metadata.get("document_id")
                    if doc_id:
                        document_ids.add(doc_id)

            # If documents were found, return the unique count
            if document_ids:
                return len(document_ids)

            # Fallbacks:
            # 1) If ids are present, use their length
            ids = results.get("ids") or results.get("ids".lower(), []) or []
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = [i for sub in ids for i in sub]
            if ids:
                return len(ids)

            # 2) As a last resort, if the collection supports count(), return it
            if hasattr(self.collection, "count"):
                try:
                    return int(self.collection.count())
                except Exception:
                    pass

            return 0

        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0

    async def get_chunk_count(self) -> int:
        """Get the total number of chunks in the vector store.

        Returns:
            Number of chunks.
        """
        try:
            if self.collection is None:
                await self.initialize()

            # Prefer .count() when available
            if hasattr(self.collection, "count"):
                try:
                    return int(self.collection.count())
                except Exception:
                    pass

            # Fallback to using get()
            results = self.collection.get(include=["ids", "metadatas"])
            ids = results.get("ids") or results.get("ids".lower(), []) or []
            # ids can be flat list or nested list[[...]]
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = [i for sub in ids for i in sub]
            if ids:
                return len(ids)
            # As a final fallback, use metadatas length if provided
            metas = results.get("metadatas") or results.get("metadatas".lower(), []) or []
            if isinstance(metas, list) and metas and isinstance(metas[0], list):
                metas = [m for sub in metas for m in sub]
            return len(metas)

        except Exception as e:
            logger.error(f"Failed to count chunks: {e}")
            return 0

    def _prepare_chunk_metadata(self, chunk: Chunk, document: Document) -> Dict[str, Any]:
        """Prepare metadata for storing chunk in vector store.

        Args:
            chunk: Chunk to prepare metadata for.
            document: Document the chunk belongs to.

        Returns:
            Metadata dictionary for vector store.
        """
        metadata = {
            "document_id": document.id,
            "document_path": str(document.path),
            "document_title": document.title or document.filename,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number or 0,
        }

        # Add document metadata (flatten nested structures)
        for key, value in document.metadata.items():
            if value is not None:
                if isinstance(value, (dict, list)):
                    metadata[f"doc_{key}"] = str(value)
                else:
                    metadata[f"doc_{key}"] = value

        # Add chunk metadata (flatten nested structures)
        for key, value in chunk.metadata.items():
            if value is not None:
                if isinstance(value, (dict, list)):
                    metadata[f"chunk_{key}"] = str(value)
                else:
                    metadata[f"chunk_{key}"] = value

        # Ensure all values are compatible with Chroma
        cleaned_metadata = {}
        for key, value in metadata.items():
            if value is None:
                cleaned_metadata[key] = ""
            elif isinstance(value, bool):
                cleaned_metadata[key] = value
            elif isinstance(value, (int, float)):
                cleaned_metadata[key] = value
            else:
                cleaned_metadata[key] = str(value)

        return cleaned_metadata

    def _build_where_clause(self, metadata_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build a where clause for Chroma filtering.

        Args:
            metadata_filter: Metadata filter dictionary.

        Returns:
            Chroma where clause or None.
        """
        if not metadata_filter:
            return None

        # Convert filter to Chroma format
        where_clause = {}

        for key, value in metadata_filter.items():
            if isinstance(value, dict):
                # Handle operators like {"$gt": 5}
                where_clause[key] = value
            elif isinstance(value, list):
                # Handle list of values (OR condition)
                where_clause[key] = {"$in": value}
            else:
                # Direct equality
                where_clause[key] = value

        return where_clause

    def _chunk_from_metadata(self, chunk_id: str, text: str, metadata: Dict[str, Any]) -> Chunk:
        """Create a Chunk object from vector store metadata.

        Args:
            chunk_id: Chunk ID.
            text: Chunk text.
            metadata: Chunk metadata from vector store.

        Returns:
            Chunk object.
        """
        # Extract chunk-specific metadata
        chunk_metadata = {}
        for key, value in metadata.items():
            if key.startswith("chunk_"):
                chunk_metadata[key[6:]] = value  # Remove "chunk_" prefix

        return Chunk(
            id=chunk_id,
            document_id=metadata.get("document_id", ""),
            text=text,
            page_number=metadata.get("page_number"),
            chunk_index=metadata.get("chunk_index", 0),
            metadata=chunk_metadata,
        )

    def _document_from_metadata(self, metadata: Dict[str, Any]) -> Document:
        """Create a Document object from vector store metadata.

        Args:
            metadata: Document metadata from vector store.

        Returns:
            Document object (minimal, without chunks).
        """
        # Extract document-specific metadata
        doc_metadata = {}
        for key, value in metadata.items():
            if key.startswith("doc_"):
                doc_metadata[key[4:]] = value  # Remove "doc_" prefix

        return Document(
            id=metadata.get("document_id", ""),
            path=metadata.get("document_path", ""),
            title=metadata.get("document_title"),
            metadata=doc_metadata,
        )

    async def reset_database(self) -> None:
        """Reset the entire vector database by deleting and recreating the collection.

        This is used when configuration changes require re-processing all documents.
        """
        try:
            if self.client is None:
                await self.initialize()
                return

            logger.info("Resetting vector database...")

            # Delete the existing collection
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"Could not delete collection (may not exist): {e}")

            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"description": "PDF Knowledgebase documents"}
            )

            logger.info(f"Created new collection '{self.collection_name}'")

        except Exception as e:
            raise VectorStoreError(f"Failed to reset vector database: {e}", "reset", e)

    async def close(self) -> None:
        """Close the vector store connection."""
        try:
            # Chroma client doesn't require explicit closing
            self.client = None
            self.collection = None
            logger.info("Vector store closed")

        except Exception as e:
            logger.error(f"Error closing vector store: {e}")
