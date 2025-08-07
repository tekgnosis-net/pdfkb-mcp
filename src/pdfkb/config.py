"""Configuration management for the PDF Knowledgebase server."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Runtime configuration for the PDF Knowledgebase server."""

    # Required settings
    openai_api_key: str

    # Optional settings with defaults
    knowledgebase_path: Path = field(default_factory=lambda: Path("./pdfs"))
    cache_dir: Path = field(default_factory=lambda: Path(""))
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 100
    vector_search_k: int = 5
    file_scan_interval: int = 60
    log_level: str = "INFO"
    supported_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    unstructured_pdf_processing_strategy: str = "fast"
    pdf_parser: str = "pymupdf4llm"
    pdf_chunker: str = "langchain"
    docling_config: Dict[str, Any] = field(default_factory=dict)
    # Marker LLM configuration
    marker_use_llm: bool = False
    marker_llm_model: str = "google/gemini-2.5-flash-lite"
    openrouter_api_key: str = ""
    # MinerU configuration
    mineru_lang: str = "en"
    mineru_method: str = "auto"
    mineru_vram: int = 16

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set cache_dir relative to knowledgebase_path if not explicitly provided
        if not self.cache_dir or self.cache_dir == Path(""):
            self.cache_dir = self.knowledgebase_path / ".cache"

        self._validate_config()
        self._ensure_directories()

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required")

        if not self.openai_api_key.startswith("sk-"):
            raise ConfigurationError("Invalid OpenAI API key format")

        if self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")

        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")

        if self.embedding_batch_size <= 0:
            raise ConfigurationError("embedding_batch_size must be positive")

        if self.vector_search_k <= 0:
            raise ConfigurationError("vector_search_k must be positive")

        if self.file_scan_interval <= 0:
            raise ConfigurationError("file_scan_interval must be positive")

        if self.unstructured_pdf_processing_strategy not in ["fast", "hi_res"]:
            raise ConfigurationError("unstructured_pdf_processing_strategy must be either 'fast' or 'hi_res'")

        if self.pdf_parser not in [
            "unstructured",
            "pymupdf4llm",
            "mineru",
            "marker",
            "docling",
            "llm",
        ]:
            raise ConfigurationError(
                "pdf_parser must be either 'unstructured', 'pymupdf4llm', 'mineru', 'marker', 'docling', or 'llm'"
            )

        if self.pdf_chunker not in ["langchain", "unstructured"]:
            raise ConfigurationError("pdf_chunker must be either 'langchain' or 'unstructured'")

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        try:
            self.knowledgebase_path.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Create cache subdirectories
            (self.cache_dir / "chroma").mkdir(exist_ok=True)
            (self.cache_dir / "metadata").mkdir(exist_ok=True)
            (self.cache_dir / "processing").mkdir(exist_ok=True)

        except Exception as e:
            raise ConfigurationError(f"Failed to create directories: {e}")

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables.

        Automatically loads from .env file if it exists in the current directory
        or parent directories.

        Returns:
            ServerConfig instance loaded from environment.

        Raises:
            ConfigurationError: If required environment variables are missing.
        """
        # Load .env file if it exists
        env_file = None
        current_path = Path.cwd()

        # Look for .env file in current directory and parent directories
        for path in [current_path] + list(current_path.parents):
            potential_env = path / ".env"
            if potential_env.exists():
                env_file = potential_env
                break

        if env_file:
            load_dotenv(env_file, override=False)  # Don't override existing env vars
            logger.info(f"Loaded environment variables from: {env_file}")

        # Get required settings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY environment variable is required")

        # Get optional settings with defaults
        config_kwargs = {
            "openai_api_key": openai_api_key,
        }

        # Parse optional path settings
        if knowledgebase_path := os.getenv("KNOWLEDGEBASE_PATH"):
            config_kwargs["knowledgebase_path"] = Path(knowledgebase_path)

        if cache_dir := os.getenv("CACHE_DIR"):
            config_kwargs["cache_dir"] = Path(cache_dir)

        # Parse optional integer settings
        if chunk_size := os.getenv("CHUNK_SIZE"):
            try:
                config_kwargs["chunk_size"] = int(chunk_size)
            except ValueError:
                raise ConfigurationError(f"Invalid CHUNK_SIZE: {chunk_size}")

        if chunk_overlap := os.getenv("CHUNK_OVERLAP"):
            try:
                config_kwargs["chunk_overlap"] = int(chunk_overlap)
            except ValueError:
                raise ConfigurationError(f"Invalid CHUNK_OVERLAP: {chunk_overlap}")

        if embedding_batch_size := os.getenv("EMBEDDING_BATCH_SIZE"):
            try:
                config_kwargs["embedding_batch_size"] = int(embedding_batch_size)
            except ValueError:
                raise ConfigurationError(f"Invalid EMBEDDING_BATCH_SIZE: {embedding_batch_size}")

        if vector_search_k := os.getenv("VECTOR_SEARCH_K"):
            try:
                config_kwargs["vector_search_k"] = int(vector_search_k)
            except ValueError:
                raise ConfigurationError(f"Invalid VECTOR_SEARCH_K: {vector_search_k}")

        if file_scan_interval := os.getenv("FILE_SCAN_INTERVAL"):
            try:
                config_kwargs["file_scan_interval"] = int(file_scan_interval)
            except ValueError:
                raise ConfigurationError(f"Invalid FILE_SCAN_INTERVAL: {file_scan_interval}")

        # Parse optional string settings
        if embedding_model := os.getenv("EMBEDDING_MODEL"):
            config_kwargs["embedding_model"] = embedding_model

        if log_level := os.getenv("LOG_LEVEL"):
            config_kwargs["log_level"] = log_level.upper()

        # Parse PDF processing strategy (backward compatibility)
        if pdf_strategy := os.getenv("PDF_PROCESSING_STRATEGY"):
            strategy = pdf_strategy.lower()
            if strategy not in ["fast", "hi_res"]:
                raise ConfigurationError(f"Invalid PDF_PROCESSING_STRATEGY: {pdf_strategy}. Must be 'fast' or 'hi_res'")
            config_kwargs["unstructured_pdf_processing_strategy"] = strategy

        # Parse Unstructured PDF processing strategy
        if unstructured_strategy := os.getenv("UNSTRUCTURED_PDF_PROCESSING_STRATEGY"):
            strategy = unstructured_strategy.lower()
            if strategy not in ["fast", "hi_res"]:
                raise ConfigurationError(
                    f"Invalid UNSTRUCTURED_PDF_PROCESSING_STRATEGY: {unstructured_strategy}. Must be 'fast' or 'hi_res'"
                )
            config_kwargs["unstructured_pdf_processing_strategy"] = strategy

        # Parse PDF parser selection
        if pdf_parser := os.getenv("PDF_PARSER"):
            parser = pdf_parser.lower()
            if parser not in ["unstructured", "pymupdf4llm", "mineru", "marker", "docling", "llm"]:
                raise ConfigurationError(
                    f"Invalid PDF_PARSER: {pdf_parser}. Must be 'unstructured', 'pymupdf4llm', "
                    "'mineru', 'marker', 'docling', or 'llm'"
                )
            config_kwargs["pdf_parser"] = parser

        # Parse docling-specific environment variables
        docling_config = {}

        if docling_ocr_engine := os.getenv("DOCLING_OCR_ENGINE"):
            docling_config["ocr_engine"] = docling_ocr_engine.lower()

        if docling_ocr_languages := os.getenv("DOCLING_OCR_LANGUAGES"):
            docling_config["ocr_languages"] = [lang.strip() for lang in docling_ocr_languages.split(",")]

        if docling_table_mode := os.getenv("DOCLING_TABLE_MODE"):
            table_mode = docling_table_mode.upper()
            if table_mode in ["FAST", "ACCURATE"]:
                docling_config["table_processing_mode"] = table_mode

        if docling_formula_enrichment := os.getenv("DOCLING_FORMULA_ENRICHMENT"):
            docling_config["formula_enrichment"] = docling_formula_enrichment.lower() in [
                "true",
                "1",
                "yes",
            ]

        if docling_timeout := os.getenv("DOCLING_PROCESSING_TIMEOUT"):
            try:
                docling_config["processing_timeout"] = int(docling_timeout)
            except ValueError:
                raise ConfigurationError(f"Invalid DOCLING_PROCESSING_TIMEOUT: {docling_timeout}")

        if docling_device := os.getenv("DOCLING_DEVICE"):
            device = docling_device.lower()
            if device in ["auto", "cpu", "cuda", "mps"]:
                docling_config["device_selection"] = device

        if docling_max_pages := os.getenv("DOCLING_MAX_PAGES"):
            try:
                docling_config["max_pages"] = int(docling_max_pages)
            except ValueError:
                raise ConfigurationError(f"Invalid DOCLING_MAX_PAGES: {docling_max_pages}")

        # Store docling config for parser initialization
        if docling_config:
            config_kwargs["docling_config"] = docling_config

        # Parse Marker LLM configuration
        if marker_use_llm := os.getenv("MARKER_USE_LLM"):
            config_kwargs["marker_use_llm"] = marker_use_llm.lower() in ["true", "1", "yes"]

        if marker_llm_model := os.getenv("MARKER_LLM_MODEL"):
            config_kwargs["marker_llm_model"] = marker_llm_model

        if openrouter_api_key := os.getenv("OPENROUTER_API_KEY"):
            config_kwargs["openrouter_api_key"] = openrouter_api_key

        # Parse MinerU configuration
        if mineru_lang := os.getenv("MINERU_LANG"):
            config_kwargs["mineru_lang"] = mineru_lang

        if mineru_method := os.getenv("MINERU_METHOD"):
            config_kwargs["mineru_method"] = mineru_method

        if mineru_vram := os.getenv("MINERU_VRAM"):
            try:
                config_kwargs["mineru_vram"] = int(mineru_vram)
            except ValueError:
                raise ConfigurationError(f"Invalid MINERU_VRAM: {mineru_vram}")

        # Parse PDF chunker selection
        if pdf_chunker := os.getenv("PDF_CHUNKER"):
            chunker = pdf_chunker.lower()
            if chunker not in ["langchain", "unstructured"]:
                raise ConfigurationError(f"Invalid PDF_CHUNKER: {pdf_chunker}. Must be 'langchain' or 'unstructured'")
            config_kwargs["pdf_chunker"] = chunker

        return cls(**config_kwargs)

    @property
    def chroma_path(self) -> Path:
        """Get the path to the Chroma database."""
        return self.cache_dir / "chroma"

    @property
    def metadata_path(self) -> Path:
        """Get the path to the metadata directory."""
        return self.cache_dir / "metadata"

    @property
    def processing_path(self) -> Path:
        """Get the path to the processing directory."""
        return self.cache_dir / "processing"

    def get_document_cache_path(self, document_id: str) -> Path:
        """Get the cache path for a specific document.

        Args:
            document_id: The document identifier.

        Returns:
            Path to the document's cache directory.
        """
        return self.processing_path / document_id

    def get_intelligent_cache_manager(self):
        """Get an IntelligentCacheManager instance for this configuration.

        Note: Import is done locally to avoid circular imports.

        Returns:
            IntelligentCacheManager instance.
        """
        from .intelligent_cache import IntelligentCacheManager

        return IntelligentCacheManager(self, self.cache_dir)

    def get_parsing_fingerprint(self) -> str:
        """Generate fingerprint for parsing configuration using intelligent cache manager.

        Returns:
            SHA-256 hash of parsing-related parameters.
        """
        cache_manager = self.get_intelligent_cache_manager()
        return cache_manager.get_parsing_fingerprint()

    def get_chunking_fingerprint(self) -> str:
        """Generate fingerprint for chunking configuration using intelligent cache manager.

        Returns:
            SHA-256 hash of chunking-related parameters.
        """
        cache_manager = self.get_intelligent_cache_manager()
        return cache_manager.get_chunking_fingerprint()

    def get_embedding_fingerprint(self) -> str:
        """Generate fingerprint for embedding configuration using intelligent cache manager.

        Returns:
            SHA-256 hash of embedding-related parameters.
        """
        cache_manager = self.get_intelligent_cache_manager()
        return cache_manager.get_embedding_fingerprint()

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
        cache_manager = self.get_intelligent_cache_manager()
        return cache_manager.detect_config_changes()

    def update_intelligent_fingerprints(self) -> None:
        """Update all stage-specific fingerprints with current configuration.

        This should be called after successful processing to record the current
        configuration state using the intelligent cache manager.

        Raises:
            ConfigurationError: If fingerprints cannot be saved.
        """
        cache_manager = self.get_intelligent_cache_manager()
        cache_manager.update_fingerprints()

    def has_parsing_config_changed(self) -> bool:
        """Check if parsing configuration has changed.

        Returns:
            True if parsing configuration has changed.
        """
        changes = self.detect_config_changes()
        return changes["parsing"]

    def has_chunking_config_changed(self) -> bool:
        """Check if chunking configuration has changed.

        Returns:
            True if chunking configuration has changed.
        """
        changes = self.detect_config_changes()
        return changes["chunking"]

    def has_embedding_config_changed(self) -> bool:
        """Check if embedding configuration has changed.

        Returns:
            True if embedding configuration has changed.
        """
        changes = self.detect_config_changes()
        return changes["embedding"]
