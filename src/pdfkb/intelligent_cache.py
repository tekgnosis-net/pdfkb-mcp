"""Intelligent cache management with step-specific configuration fingerprinting."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ServerConfig
from .exceptions import ConfigurationError


class IntelligentCacheManager:
    """Manages intelligent caching with step-specific configuration fingerprinting.

    This class provides granular cache invalidation by tracking configuration changes
    for specific processing stages: parsing, chunking, and embedding. This allows
    for more efficient cache management by only invalidating caches for stages
    whose configuration has actually changed.
    """

    def __init__(self, config: ServerConfig, cache_dir: Path):
        """Initialize the intelligent cache manager.

        Args:
            config: Server configuration instance.
            cache_dir: Base cache directory path.
        """
        self.config = config
        self.cache_dir = cache_dir
        self.fingerprints_dir = cache_dir / "metadata" / "fingerprints"

        # Ensure fingerprints directory exists
        self.fingerprints_dir.mkdir(parents=True, exist_ok=True)

    def get_parsing_fingerprint(self) -> str:
        """Generate fingerprint for parsing configuration.

        Returns:
            SHA-256 hash of parsing-related parameters.
        """
        parsing_params = {
            "pdf_parser": self.config.pdf_parser,
            "unstructured_pdf_processing_strategy": self.config.unstructured_pdf_processing_strategy,
            "marker_use_llm": getattr(self.config, "marker_use_llm", False),
            "marker_llm_model": getattr(self.config, "marker_llm_model", "gpt-4o"),
        }

        fingerprint_string = json.dumps(parsing_params, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode("utf-8")).hexdigest()

    def get_chunking_fingerprint(self) -> str:
        """Generate fingerprint for chunking configuration.

        Returns:
            SHA-256 hash of chunking-related parameters.
        """
        chunking_params = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "pdf_chunker": self.config.pdf_chunker,
        }

        # Add semantic chunker config if using semantic chunking
        if self.config.pdf_chunker == "semantic":
            chunking_params.update(
                {
                    "semantic_threshold_type": self.config.semantic_chunker_threshold_type,
                    "semantic_threshold_amount": self.config.semantic_chunker_threshold_amount,
                    "semantic_buffer_size": self.config.semantic_chunker_buffer_size,
                    "semantic_min_chunk_chars": self.config.semantic_chunker_min_chunk_chars,
                    "semantic_number_of_chunks": self.config.semantic_chunker_number_of_chunks,
                    "semantic_sentence_split_regex": self.config.semantic_chunker_sentence_split_regex,
                }
            )

        fingerprint_string = json.dumps(chunking_params, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode("utf-8")).hexdigest()

    def get_embedding_fingerprint(self) -> str:
        """Generate fingerprint for embedding configuration.

        Returns:
            SHA-256 hash of embedding-related parameters.
        """
        embedding_params = {
            "embedding_model": self.config.embedding_model,
        }

        fingerprint_string = json.dumps(embedding_params, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode("utf-8")).hexdigest()

    def _get_fingerprint_path(self, stage: str) -> Path:
        """Get the path to a stage-specific fingerprint file.

        Args:
            stage: Processing stage name (parsing, chunking, embedding).

        Returns:
            Path to the fingerprint file.
        """
        return self.fingerprints_dir / f"{stage}.json"

    def _save_stage_fingerprint(self, stage: str, fingerprint: str, config_params: Dict[str, Any]) -> None:
        """Save a stage-specific fingerprint to disk.

        Args:
            stage: Processing stage name.
            fingerprint: The fingerprint hash.
            config_params: Configuration parameters for this stage.

        Raises:
            ConfigurationError: If fingerprint cannot be saved.
        """
        try:
            fingerprint_data = {
                "fingerprint": fingerprint,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config_version": "1.0.0",  # Version for future compatibility
                "config": config_params,
            }

            fingerprint_path = self._get_fingerprint_path(stage)
            with open(fingerprint_path, "w", encoding="utf-8") as f:
                json.dump(fingerprint_data, f, indent=2)

        except Exception as e:
            raise ConfigurationError(f"Failed to save {stage} fingerprint: {e}")

    def _load_stage_fingerprint(self, stage: str) -> Dict[str, Any]:
        """Load a stage-specific fingerprint from disk.

        Args:
            stage: Processing stage name.

        Returns:
            Dictionary containing fingerprint data, or empty dict if not found or corrupted.
        """
        try:
            fingerprint_path = self._get_fingerprint_path(stage)
            if fingerprint_path.exists():
                with open(fingerprint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Validate required fields
                    if "fingerprint" in data and "timestamp" in data:
                        return data
            return {}
        except Exception:
            # Handle corrupted files gracefully by returning empty dict
            return {}

    def detect_config_changes(self) -> Dict[str, bool]:
        """Detect which processing stages have configuration changes.

        Returns:
            Dictionary mapping stage names to change status:
            {
                "parsing": bool,
                "chunking": bool,
                "embedding": bool
            }
        """
        changes = {}

        # Check parsing changes
        current_parsing = self.get_parsing_fingerprint()
        saved_parsing = self._load_stage_fingerprint("parsing")
        changes["parsing"] = not saved_parsing or current_parsing != saved_parsing.get("fingerprint")

        # Check chunking changes
        current_chunking = self.get_chunking_fingerprint()
        saved_chunking = self._load_stage_fingerprint("chunking")
        changes["chunking"] = not saved_chunking or current_chunking != saved_chunking.get("fingerprint")

        # Check embedding changes
        current_embedding = self.get_embedding_fingerprint()
        saved_embedding = self._load_stage_fingerprint("embedding")
        changes["embedding"] = not saved_embedding or current_embedding != saved_embedding.get("fingerprint")

        return changes

    def update_fingerprints(self) -> None:
        """Update all stage-specific fingerprints with current configuration.

        This should be called after successful processing to record the current
        configuration state.

        Raises:
            ConfigurationError: If fingerprints cannot be saved.
        """
        # Save parsing fingerprint
        parsing_config = {
            "pdf_parser": self.config.pdf_parser,
            "unstructured_pdf_processing_strategy": self.config.unstructured_pdf_processing_strategy,
            "marker_use_llm": getattr(self.config, "marker_use_llm", False),
            "marker_llm_model": getattr(self.config, "marker_llm_model", "gpt-4o"),
        }
        self._save_stage_fingerprint("parsing", self.get_parsing_fingerprint(), parsing_config)

        # Save chunking fingerprint
        chunking_config = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "pdf_chunker": self.config.pdf_chunker,
        }

        # Add semantic chunker config if using semantic chunking
        if self.config.pdf_chunker == "semantic":
            chunking_config.update(
                {
                    "semantic_threshold_type": self.config.semantic_chunker_threshold_type,
                    "semantic_threshold_amount": self.config.semantic_chunker_threshold_amount,
                    "semantic_buffer_size": self.config.semantic_chunker_buffer_size,
                    "semantic_min_chunk_chars": self.config.semantic_chunker_min_chunk_chars,
                    "semantic_number_of_chunks": self.config.semantic_chunker_number_of_chunks,
                    "semantic_sentence_split_regex": self.config.semantic_chunker_sentence_split_regex,
                }
            )

        self._save_stage_fingerprint("chunking", self.get_chunking_fingerprint(), chunking_config)

        # Save embedding fingerprint
        embedding_config = {
            "embedding_model": self.config.embedding_model,
        }
        self._save_stage_fingerprint("embedding", self.get_embedding_fingerprint(), embedding_config)

    def is_parsing_cache_valid(self, document_hash: str) -> bool:
        """Check if parsing cache is valid for a document.

        Args:
            document_hash: Hash identifier for the document.

        Returns:
            True if parsing cache is valid, False otherwise.
        """
        # For now, just check if parsing config hasn't changed
        # Future implementation could include document-specific validation
        changes = self.detect_config_changes()
        return not changes["parsing"]

    def is_chunking_cache_valid(self, document_hash: str) -> bool:
        """Check if chunking cache is valid for a document.

        Args:
            document_hash: Hash identifier for the document.

        Returns:
            True if chunking cache is valid, False otherwise.
        """
        # For now, just check if chunking config hasn't changed
        # Future implementation could include document-specific validation
        changes = self.detect_config_changes()
        return not changes["chunking"]

    def is_embedding_cache_valid(self, document_hash: str) -> bool:
        """Check if embedding cache is valid for a document.

        Args:
            document_hash: Hash identifier for the document.

        Returns:
            True if embedding cache is valid, False otherwise.
        """
        # For now, just check if embedding config hasn't changed
        # Future implementation could include document-specific validation
        changes = self.detect_config_changes()
        return not changes["embedding"]

    def get_stage_fingerprint_info(self, stage: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a stage's fingerprint.

        Args:
            stage: Processing stage name (parsing, chunking, embedding).

        Returns:
            Dictionary with fingerprint info, or None if not found.
        """
        return self._load_stage_fingerprint(stage) or None

    def clear_stage_fingerprint(self, stage: str) -> None:
        """Clear a stage's fingerprint file.

        Args:
            stage: Processing stage name to clear.
        """
        try:
            fingerprint_path = self._get_fingerprint_path(stage)
            if fingerprint_path.exists():
                fingerprint_path.unlink()
        except Exception:
            # Ignore errors when clearing fingerprints
            pass

    def clear_all_fingerprints(self) -> None:
        """Clear all stage fingerprints."""
        for stage in ["parsing", "chunking", "embedding"]:
            self.clear_stage_fingerprint(stage)
