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

    # Embedding provider configuration
    embedding_provider: str = "local"  # "local" or "openai"

    # Local embedding configuration
    local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # High quality 0.6B model
    local_embedding_batch_size: int = 32
    embedding_device: str = ""  # Empty string for auto-detect, or "cpu"/"mps"/"cuda"
    embedding_cache_size: int = 10000
    max_sequence_length: int = 512
    use_model_optimization: bool = False  # Disabled due to torch.compile issues
    model_cache_dir: str = "~/.cache/pdfkb-mcp/models"
    embedding_dimension: int = 0  # 0 uses model default
    fallback_to_openai: bool = False
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
    # Web server configuration
    web_enabled: bool = True
    web_port: int = 8080
    web_host: str = "localhost"
    web_cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])

    # Hybrid search configuration
    enable_hybrid_search: bool = True
    hybrid_search_weights: Dict[str, float] = field(
        default_factory=lambda: {"vector": 0.6, "text": 0.4}  # Semantic search weight  # BM25 search weight
    )
    rrf_k: int = 60  # RRF constant parameter

    # Whoosh-specific configuration
    whoosh_index_dir: str = ""  # Auto-set to cache_dir/whoosh
    whoosh_analyzer: str = "standard"  # Whoosh analyzer type
    whoosh_min_score: float = 0.0  # Minimum BM25 score threshold

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set cache_dir relative to knowledgebase_path if not explicitly provided
        if not self.cache_dir or self.cache_dir == Path(""):
            self.cache_dir = self.knowledgebase_path / ".cache"

        # Set whoosh_index_dir if not explicitly provided
        if not self.whoosh_index_dir:
            self.whoosh_index_dir = str(self.cache_dir / "whoosh")

        self._validate_config()
        self._ensure_directories()

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # OpenAI key is only required if using OpenAI embeddings
        if self.embedding_provider == "openai":
            if not self.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY is required when using OpenAI embeddings")
            if not self.openai_api_key.startswith("sk-"):
                raise ConfigurationError("Invalid OpenAI API key format")
        elif self.embedding_provider == "local":
            # For local embeddings, we can use a dummy key
            if not self.openai_api_key:
                self.openai_api_key = "sk-local-embeddings-dummy-key"
        else:
            raise ConfigurationError(f"Invalid embedding_provider: {self.embedding_provider}")

        if self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")

        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")

        if self.embedding_batch_size <= 0:
            raise ConfigurationError("embedding_batch_size must be positive")

        # Validate local embedding configuration
        if self.embedding_provider == "local":
            if self.local_embedding_batch_size <= 0:
                raise ConfigurationError("local_embedding_batch_size must be positive")
            if self.embedding_cache_size < 0:
                raise ConfigurationError("embedding_cache_size cannot be negative")
            if self.max_sequence_length <= 0:
                raise ConfigurationError("max_sequence_length must be positive")
            if self.embedding_device and self.embedding_device not in ["", "cpu", "cuda", "mps"]:
                raise ConfigurationError("embedding_device must be 'cpu', 'cuda', 'mps', or empty for auto-detect")

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

        # Validate web server configuration
        if self.web_port <= 0 or self.web_port > 65535:
            raise ConfigurationError("web_port must be between 1 and 65535")

        if not self.web_host:
            raise ConfigurationError("web_host cannot be empty")

        # Validate hybrid search configuration
        if self.enable_hybrid_search:
            if "vector" not in self.hybrid_search_weights or "text" not in self.hybrid_search_weights:
                raise ConfigurationError("hybrid_search_weights must contain 'vector' and 'text' keys")

            total_weight = self.hybrid_search_weights["vector"] + self.hybrid_search_weights["text"]
            if abs(total_weight - 1.0) > 0.001:  # Allow small floating point errors
                raise ConfigurationError("hybrid_search_weights must sum to 1.0")

            if self.rrf_k <= 0:
                raise ConfigurationError("rrf_k must be positive")

            if self.whoosh_min_score < 0 or self.whoosh_min_score > 1:
                raise ConfigurationError("whoosh_min_score must be between 0 and 1")

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        try:
            self.knowledgebase_path.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Create cache subdirectories
            (self.cache_dir / "chroma").mkdir(exist_ok=True)
            (self.cache_dir / "metadata").mkdir(exist_ok=True)
            (self.cache_dir / "processing").mkdir(exist_ok=True)

            # Create whoosh directory if hybrid search is enabled
            if self.enable_hybrid_search:
                Path(self.whoosh_index_dir).mkdir(parents=True, exist_ok=True)

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

        # Check embedding provider first to determine if OpenAI key is required
        embedding_provider = os.getenv("PDFKB_EMBEDDING_PROVIDER", "local").lower()

        # Get OpenAI API key (optional for local embeddings)
        openai_api_key = os.getenv("PDFKB_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_API_KEY") and not os.getenv("PDFKB_OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is deprecated, use PDFKB_OPENAI_API_KEY instead")

        # Only require OpenAI key if using OpenAI embeddings
        if embedding_provider == "openai" and not openai_api_key:
            raise ConfigurationError(
                "PDFKB_OPENAI_API_KEY environment variable is required when using OpenAI embeddings"
            )

        # Use dummy key for local embeddings if not provided
        if not openai_api_key:
            openai_api_key = "sk-local-embeddings-dummy-key"

        # Get optional settings with defaults
        config_kwargs = {
            "openai_api_key": openai_api_key,
            "embedding_provider": embedding_provider,
        }

        # Parse optional path settings
        knowledgebase_path = os.getenv("PDFKB_KNOWLEDGEBASE_PATH") or os.getenv("KNOWLEDGEBASE_PATH")
        if os.getenv("KNOWLEDGEBASE_PATH") and not os.getenv("PDFKB_KNOWLEDGEBASE_PATH"):
            logger.warning("KNOWLEDGEBASE_PATH is deprecated, use PDFKB_KNOWLEDGEBASE_PATH instead")
        if knowledgebase_path:
            config_kwargs["knowledgebase_path"] = Path(knowledgebase_path)

        cache_dir = os.getenv("PDFKB_CACHE_DIR") or os.getenv("CACHE_DIR")
        if os.getenv("CACHE_DIR") and not os.getenv("PDFKB_CACHE_DIR"):
            logger.warning("CACHE_DIR is deprecated, use PDFKB_CACHE_DIR instead")
        if cache_dir:
            config_kwargs["cache_dir"] = Path(cache_dir)

        # Parse optional integer settings
        chunk_size = os.getenv("PDFKB_CHUNK_SIZE") or os.getenv("CHUNK_SIZE")
        if os.getenv("CHUNK_SIZE") and not os.getenv("PDFKB_CHUNK_SIZE"):
            logger.warning("CHUNK_SIZE is deprecated, use PDFKB_CHUNK_SIZE instead")
        if chunk_size:
            try:
                config_kwargs["chunk_size"] = int(chunk_size)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_CHUNK_SIZE: {chunk_size}")

        chunk_overlap = os.getenv("PDFKB_CHUNK_OVERLAP") or os.getenv("CHUNK_OVERLAP")
        if os.getenv("CHUNK_OVERLAP") and not os.getenv("PDFKB_CHUNK_OVERLAP"):
            logger.warning("CHUNK_OVERLAP is deprecated, use PDFKB_CHUNK_OVERLAP instead")
        if chunk_overlap:
            try:
                config_kwargs["chunk_overlap"] = int(chunk_overlap)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_CHUNK_OVERLAP: {chunk_overlap}")

        embedding_batch_size = os.getenv("PDFKB_EMBEDDING_BATCH_SIZE") or os.getenv("EMBEDDING_BATCH_SIZE")
        if os.getenv("EMBEDDING_BATCH_SIZE") and not os.getenv("PDFKB_EMBEDDING_BATCH_SIZE"):
            logger.warning("EMBEDDING_BATCH_SIZE is deprecated, use PDFKB_EMBEDDING_BATCH_SIZE instead")
        if embedding_batch_size:
            try:
                config_kwargs["embedding_batch_size"] = int(embedding_batch_size)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_EMBEDDING_BATCH_SIZE: {embedding_batch_size}")

        vector_search_k = os.getenv("PDFKB_VECTOR_SEARCH_K") or os.getenv("VECTOR_SEARCH_K")
        if os.getenv("VECTOR_SEARCH_K") and not os.getenv("PDFKB_VECTOR_SEARCH_K"):
            logger.warning("VECTOR_SEARCH_K is deprecated, use PDFKB_VECTOR_SEARCH_K instead")
        if vector_search_k:
            try:
                config_kwargs["vector_search_k"] = int(vector_search_k)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_VECTOR_SEARCH_K: {vector_search_k}")

        file_scan_interval = os.getenv("PDFKB_FILE_SCAN_INTERVAL") or os.getenv("FILE_SCAN_INTERVAL")
        if os.getenv("FILE_SCAN_INTERVAL") and not os.getenv("PDFKB_FILE_SCAN_INTERVAL"):
            logger.warning("FILE_SCAN_INTERVAL is deprecated, use PDFKB_FILE_SCAN_INTERVAL instead")
        if file_scan_interval:
            try:
                config_kwargs["file_scan_interval"] = int(file_scan_interval)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_FILE_SCAN_INTERVAL: {file_scan_interval}")

        # Parse optional string settings
        embedding_model = os.getenv("PDFKB_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL")
        if os.getenv("EMBEDDING_MODEL") and not os.getenv("PDFKB_EMBEDDING_MODEL"):
            logger.warning("EMBEDDING_MODEL is deprecated, use PDFKB_EMBEDDING_MODEL instead")
        if embedding_model:
            config_kwargs["embedding_model"] = embedding_model

        # Embedding provider already parsed and set above

        # Parse local embedding configuration
        local_embedding_model = os.getenv("PDFKB_LOCAL_EMBEDDING_MODEL")
        if local_embedding_model:
            config_kwargs["local_embedding_model"] = local_embedding_model

        local_embedding_batch_size = os.getenv("PDFKB_LOCAL_EMBEDDING_BATCH_SIZE")
        if local_embedding_batch_size:
            try:
                config_kwargs["local_embedding_batch_size"] = int(local_embedding_batch_size)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_LOCAL_EMBEDDING_BATCH_SIZE: {local_embedding_batch_size}")

        embedding_device = os.getenv("PDFKB_EMBEDDING_DEVICE")
        if embedding_device:
            config_kwargs["embedding_device"] = embedding_device

        embedding_cache_size = os.getenv("PDFKB_EMBEDDING_CACHE_SIZE")
        if embedding_cache_size:
            try:
                config_kwargs["embedding_cache_size"] = int(embedding_cache_size)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_EMBEDDING_CACHE_SIZE: {embedding_cache_size}")

        max_sequence_length = os.getenv("PDFKB_MAX_SEQUENCE_LENGTH")
        if max_sequence_length:
            try:
                config_kwargs["max_sequence_length"] = int(max_sequence_length)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_MAX_SEQUENCE_LENGTH: {max_sequence_length}")

        use_model_optimization = os.getenv("PDFKB_USE_MODEL_OPTIMIZATION")
        if use_model_optimization:
            config_kwargs["use_model_optimization"] = use_model_optimization.lower() in ["true", "1", "yes"]

        model_cache_dir = os.getenv("PDFKB_MODEL_CACHE_DIR")
        if model_cache_dir:
            config_kwargs["model_cache_dir"] = model_cache_dir

        embedding_dimension = os.getenv("PDFKB_EMBEDDING_DIMENSION")
        if embedding_dimension:
            try:
                config_kwargs["embedding_dimension"] = int(embedding_dimension)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_EMBEDDING_DIMENSION: {embedding_dimension}")

        fallback_to_openai = os.getenv("PDFKB_FALLBACK_TO_OPENAI")
        if fallback_to_openai:
            config_kwargs["fallback_to_openai"] = fallback_to_openai.lower() in ["true", "1", "yes"]

        log_level = os.getenv("PDFKB_LOG_LEVEL") or os.getenv("LOG_LEVEL")
        if os.getenv("LOG_LEVEL") and not os.getenv("PDFKB_LOG_LEVEL"):
            logger.warning("LOG_LEVEL is deprecated, use PDFKB_LOG_LEVEL instead")
        if log_level:
            config_kwargs["log_level"] = log_level.upper()

        # Parse PDF processing strategy (backward compatibility)
        pdf_strategy = os.getenv("PDFKB_PDF_PROCESSING_STRATEGY") or os.getenv("PDF_PROCESSING_STRATEGY")
        if os.getenv("PDF_PROCESSING_STRATEGY") and not os.getenv("PDFKB_PDF_PROCESSING_STRATEGY"):
            logger.warning("PDF_PROCESSING_STRATEGY is deprecated, use PDFKB_PDF_PROCESSING_STRATEGY instead")
        if pdf_strategy:
            strategy = pdf_strategy.lower()
            if strategy not in ["fast", "hi_res"]:
                raise ConfigurationError(
                    f"Invalid PDFKB_PDF_PROCESSING_STRATEGY: {pdf_strategy}. Must be 'fast' or 'hi_res'"
                )
            config_kwargs["unstructured_pdf_processing_strategy"] = strategy

        # Parse Unstructured PDF processing strategy
        unstructured_strategy = os.getenv("PDFKB_UNSTRUCTURED_PDF_PROCESSING_STRATEGY") or os.getenv(
            "UNSTRUCTURED_PDF_PROCESSING_STRATEGY"
        )
        if os.getenv("UNSTRUCTURED_PDF_PROCESSING_STRATEGY") and not os.getenv(
            "PDFKB_UNSTRUCTURED_PDF_PROCESSING_STRATEGY"
        ):
            logger.warning(
                "UNSTRUCTURED_PDF_PROCESSING_STRATEGY is deprecated, "
                "use PDFKB_UNSTRUCTURED_PDF_PROCESSING_STRATEGY instead"
            )
        if unstructured_strategy:
            strategy = unstructured_strategy.lower()
            if strategy not in ["fast", "hi_res"]:
                raise ConfigurationError(
                    f"Invalid PDFKB_UNSTRUCTURED_PDF_PROCESSING_STRATEGY: {unstructured_strategy}. "
                    f"Must be 'fast' or 'hi_res'"
                )
            config_kwargs["unstructured_pdf_processing_strategy"] = strategy

        # Parse PDF parser selection
        pdf_parser = os.getenv("PDFKB_PDF_PARSER") or os.getenv("PDF_PARSER")
        if os.getenv("PDF_PARSER") and not os.getenv("PDFKB_PDF_PARSER"):
            logger.warning("PDF_PARSER is deprecated, use PDFKB_PDF_PARSER instead")
        if pdf_parser:
            parser = pdf_parser.lower()
            if parser not in ["unstructured", "pymupdf4llm", "mineru", "marker", "docling", "llm"]:
                raise ConfigurationError(
                    f"Invalid PDFKB_PDF_PARSER: {pdf_parser}. Must be 'unstructured', 'pymupdf4llm', "
                    "'mineru', 'marker', 'docling', or 'llm'"
                )
            config_kwargs["pdf_parser"] = parser

        # Parse docling-specific environment variables
        docling_config = {}

        docling_ocr_engine = os.getenv("PDFKB_DOCLING_OCR_ENGINE") or os.getenv("DOCLING_OCR_ENGINE")
        if os.getenv("DOCLING_OCR_ENGINE") and not os.getenv("PDFKB_DOCLING_OCR_ENGINE"):
            logger.warning("DOCLING_OCR_ENGINE is deprecated, use PDFKB_DOCLING_OCR_ENGINE instead")
        if docling_ocr_engine:
            docling_config["ocr_engine"] = docling_ocr_engine.lower()

        docling_ocr_languages = os.getenv("PDFKB_DOCLING_OCR_LANGUAGES") or os.getenv("DOCLING_OCR_LANGUAGES")
        if os.getenv("DOCLING_OCR_LANGUAGES") and not os.getenv("PDFKB_DOCLING_OCR_LANGUAGES"):
            logger.warning("DOCLING_OCR_LANGUAGES is deprecated, use PDFKB_DOCLING_OCR_LANGUAGES instead")
        if docling_ocr_languages:
            docling_config["ocr_languages"] = [lang.strip() for lang in docling_ocr_languages.split(",")]

        docling_table_mode = os.getenv("PDFKB_DOCLING_TABLE_MODE") or os.getenv("DOCLING_TABLE_MODE")
        if os.getenv("DOCLING_TABLE_MODE") and not os.getenv("PDFKB_DOCLING_TABLE_MODE"):
            logger.warning("DOCLING_TABLE_MODE is deprecated, use PDFKB_DOCLING_TABLE_MODE instead")
        if docling_table_mode:
            table_mode = docling_table_mode.upper()
            if table_mode in ["FAST", "ACCURATE"]:
                docling_config["table_processing_mode"] = table_mode

        docling_formula_enrichment = os.getenv("PDFKB_DOCLING_FORMULA_ENRICHMENT") or os.getenv(
            "DOCLING_FORMULA_ENRICHMENT"
        )
        if os.getenv("DOCLING_FORMULA_ENRICHMENT") and not os.getenv("PDFKB_DOCLING_FORMULA_ENRICHMENT"):
            logger.warning("DOCLING_FORMULA_ENRICHMENT is deprecated, use PDFKB_DOCLING_FORMULA_ENRICHMENT instead")
        if docling_formula_enrichment:
            docling_config["formula_enrichment"] = docling_formula_enrichment.lower() in [
                "true",
                "1",
                "yes",
            ]

        docling_timeout = os.getenv("PDFKB_DOCLING_PROCESSING_TIMEOUT") or os.getenv("DOCLING_PROCESSING_TIMEOUT")
        if os.getenv("DOCLING_PROCESSING_TIMEOUT") and not os.getenv("PDFKB_DOCLING_PROCESSING_TIMEOUT"):
            logger.warning("DOCLING_PROCESSING_TIMEOUT is deprecated, use PDFKB_DOCLING_PROCESSING_TIMEOUT instead")
        if docling_timeout:
            try:
                docling_config["processing_timeout"] = int(docling_timeout)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_DOCLING_PROCESSING_TIMEOUT: {docling_timeout}")

        docling_device = os.getenv("PDFKB_DOCLING_DEVICE") or os.getenv("DOCLING_DEVICE")
        if os.getenv("DOCLING_DEVICE") and not os.getenv("PDFKB_DOCLING_DEVICE"):
            logger.warning("DOCLING_DEVICE is deprecated, use PDFKB_DOCLING_DEVICE instead")
        if docling_device:
            device = docling_device.lower()
            if device in ["auto", "cpu", "cuda", "mps"]:
                docling_config["device_selection"] = device

        docling_max_pages = os.getenv("PDFKB_DOCLING_MAX_PAGES") or os.getenv("DOCLING_MAX_PAGES")
        if os.getenv("DOCLING_MAX_PAGES") and not os.getenv("PDFKB_DOCLING_MAX_PAGES"):
            logger.warning("DOCLING_MAX_PAGES is deprecated, use PDFKB_DOCLING_MAX_PAGES instead")
        if docling_max_pages:
            try:
                docling_config["max_pages"] = int(docling_max_pages)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_DOCLING_MAX_PAGES: {docling_max_pages}")

        # Store docling config for parser initialization
        if docling_config:
            config_kwargs["docling_config"] = docling_config

        # Parse Marker LLM configuration
        marker_use_llm = os.getenv("PDFKB_MARKER_USE_LLM") or os.getenv("MARKER_USE_LLM")
        if os.getenv("MARKER_USE_LLM") and not os.getenv("PDFKB_MARKER_USE_LLM"):
            logger.warning("MARKER_USE_LLM is deprecated, use PDFKB_MARKER_USE_LLM instead")
        if marker_use_llm:
            config_kwargs["marker_use_llm"] = marker_use_llm.lower() in ["true", "1", "yes"]

        marker_llm_model = os.getenv("PDFKB_MARKER_LLM_MODEL") or os.getenv("MARKER_LLM_MODEL")
        if os.getenv("MARKER_LLM_MODEL") and not os.getenv("PDFKB_MARKER_LLM_MODEL"):
            logger.warning("MARKER_LLM_MODEL is deprecated, use PDFKB_MARKER_LLM_MODEL instead")
        if marker_llm_model:
            config_kwargs["marker_llm_model"] = marker_llm_model

        openrouter_api_key = os.getenv("PDFKB_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if os.getenv("OPENROUTER_API_KEY") and not os.getenv("PDFKB_OPENROUTER_API_KEY"):
            logger.warning("OPENROUTER_API_KEY is deprecated, use PDFKB_OPENROUTER_API_KEY instead")
        if openrouter_api_key:
            config_kwargs["openrouter_api_key"] = openrouter_api_key

        # Parse MinerU configuration
        mineru_lang = os.getenv("PDFKB_MINERU_LANG") or os.getenv("MINERU_LANG")
        if os.getenv("MINERU_LANG") and not os.getenv("PDFKB_MINERU_LANG"):
            logger.warning("MINERU_LANG is deprecated, use PDFKB_MINERU_LANG instead")
        if mineru_lang:
            config_kwargs["mineru_lang"] = mineru_lang

        mineru_method = os.getenv("PDFKB_MINERU_METHOD") or os.getenv("MINERU_METHOD")
        if os.getenv("MINERU_METHOD") and not os.getenv("PDFKB_MINERU_METHOD"):
            logger.warning("MINERU_METHOD is deprecated, use PDFKB_MINERU_METHOD instead")
        if mineru_method:
            config_kwargs["mineru_method"] = mineru_method

        mineru_vram = os.getenv("PDFKB_MINERU_VRAM") or os.getenv("MINERU_VRAM")
        if os.getenv("MINERU_VRAM") and not os.getenv("PDFKB_MINERU_VRAM"):
            logger.warning("MINERU_VRAM is deprecated, use PDFKB_MINERU_VRAM instead")
        if mineru_vram:
            try:
                config_kwargs["mineru_vram"] = int(mineru_vram)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_MINERU_VRAM: {mineru_vram}")

        # Parse PDF chunker selection
        pdf_chunker = os.getenv("PDFKB_PDF_CHUNKER") or os.getenv("PDF_CHUNKER")
        if os.getenv("PDF_CHUNKER") and not os.getenv("PDFKB_PDF_CHUNKER"):
            logger.warning("PDF_CHUNKER is deprecated, use PDFKB_PDF_CHUNKER instead")
        if pdf_chunker:
            chunker = pdf_chunker.lower()
            if chunker not in ["langchain", "unstructured"]:
                raise ConfigurationError(
                    f"Invalid PDFKB_PDF_CHUNKER: {pdf_chunker}. Must be 'langchain' or 'unstructured'"
                )
            config_kwargs["pdf_chunker"] = chunker

        # Parse web server configuration
        if web_enabled := os.getenv("PDFKB_ENABLE_WEB"):
            config_kwargs["web_enabled"] = web_enabled.lower() in ["true", "1", "yes"]

        web_port = os.getenv("PDFKB_WEB_PORT") or os.getenv("WEB_PORT")
        if os.getenv("WEB_PORT") and not os.getenv("PDFKB_WEB_PORT"):
            logger.warning("WEB_PORT is deprecated, use PDFKB_WEB_PORT instead")
        if web_port:
            try:
                config_kwargs["web_port"] = int(web_port)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_WEB_PORT: {web_port}")

        web_host = os.getenv("PDFKB_WEB_HOST") or os.getenv("WEB_HOST")
        if os.getenv("WEB_HOST") and not os.getenv("PDFKB_WEB_HOST"):
            logger.warning("WEB_HOST is deprecated, use PDFKB_WEB_HOST instead")
        if web_host:
            config_kwargs["web_host"] = web_host

        web_cors_origins = os.getenv("PDFKB_WEB_CORS_ORIGINS") or os.getenv("WEB_CORS_ORIGINS")
        if os.getenv("WEB_CORS_ORIGINS") and not os.getenv("PDFKB_WEB_CORS_ORIGINS"):
            logger.warning("WEB_CORS_ORIGINS is deprecated, use PDFKB_WEB_CORS_ORIGINS instead")
        if web_cors_origins:
            config_kwargs["web_cors_origins"] = [origin.strip() for origin in web_cors_origins.split(",")]

        # Parse hybrid search configuration
        if enable_hybrid := os.getenv("PDFKB_ENABLE_HYBRID_SEARCH"):
            config_kwargs["enable_hybrid_search"] = enable_hybrid.lower() in ["true", "1", "yes"]

        vector_weight = os.getenv("PDFKB_HYBRID_VECTOR_WEIGHT")
        text_weight = os.getenv("PDFKB_HYBRID_TEXT_WEIGHT")
        if vector_weight and text_weight:
            try:
                config_kwargs["hybrid_search_weights"] = {"vector": float(vector_weight), "text": float(text_weight)}
            except ValueError:
                raise ConfigurationError(f"Invalid hybrid search weights: vector={vector_weight}, text={text_weight}")

        rrf_k = os.getenv("PDFKB_RRF_K")
        if rrf_k:
            try:
                config_kwargs["rrf_k"] = int(rrf_k)
            except ValueError:
                raise ConfigurationError(f"Invalid PDFKB_RRF_K: {rrf_k}")

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
