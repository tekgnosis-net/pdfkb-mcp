"""Local embedding service using HuggingFace models."""

import asyncio
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .config import ServerConfig
from .embeddings_base import EmbeddingService
from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache implementation for embeddings."""

    def __init__(self, maxsize: int = 10000):
        """Initialize LRU cache.

        Args:
            maxsize: Maximum number of items to store.
        """
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> Optional[List[float]]:
        """Get item from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: List[float]) -> None:
        """Put item in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        if key in self.cache:
            # Move to end if already exists
            self.cache.move_to_end(key)
        else:
            # Add new item
            self.cache[key] = value
            # Remove oldest if over capacity
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class LocalEmbeddingService(EmbeddingService):
    """Local embedding service using HuggingFace models."""

    # Supported models with their specifications
    MODEL_SPECS = {
        "Qwen/Qwen3-Embedding-0.6B": {
            "dimension": 1024,
            "max_sequence_length": 32000,
            "size_gb": 1.2,
            "description": "Lightweight, 32K context",
        },
        "Qwen/Qwen3-Embedding-4B": {
            "dimension": 2560,
            "max_sequence_length": 32000,
            "size_gb": 8.0,
            "description": "High quality, 32K context",
        },
        "intfloat/multilingual-e5-large-instruct": {
            "dimension": 1024,
            "max_sequence_length": 512,
            "size_gb": 0.8,
            "description": "Multilingual, instruction-following",
        },
        "BAAI/bge-m3": {
            "dimension": 1024,
            "max_sequence_length": 8192,
            "size_gb": 2.0,
            "description": "Multilingual, 8K context",
        },
        "jinaai/jina-embeddings-v3": {
            "dimension": 1024,
            "max_sequence_length": 8192,
            "size_gb": 1.3,
            "description": "570M params, task-specific",
        },
    }

    # Legacy dimension mapping for compatibility
    MODEL_DIMENSIONS = {k: v["dimension"] for k, v in MODEL_SPECS.items()}

    def __init__(self, config: ServerConfig):
        """Initialize the local embedding service.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.model_name = config.local_embedding_model
        self.batch_size = config.local_embedding_batch_size
        self.device = None
        self.model = None
        self.tokenizer = None
        self._embedding_cache = LRUCache(maxsize=config.embedding_cache_size)
        self._model_cache_dir = Path(config.model_cache_dir).expanduser()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the model and tokenizer."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Select device
            self.device = self._select_device(self.config.embedding_device)
            logger.info(f"Using device: {self.device}")

            # Create cache directory
            self._model_cache_dir.mkdir(parents=True, exist_ok=True)

            # Load model and tokenizer
            logger.info(f"Loading model: {self.model_name}")

            # Try to load from cache first
            model_path = self._get_model_cache_path()

            try:
                if model_path.exists():
                    logger.info(f"Loading model from cache: {model_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    self.model = AutoModel.from_pretrained(
                        str(model_path),
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    )
                else:
                    logger.info(f"Downloading model: {self.model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, cache_dir=str(self._model_cache_dir), trust_remote_code=True
                    )
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        cache_dir=str(self._model_cache_dir),
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                        trust_remote_code=True,
                    )
            except Exception as e:
                # Fallback to default model if the requested one fails
                logger.warning(f"Failed to load {self.model_name}: {e}. Falling back to Qwen/Qwen3-Embedding-0.6B")
                self.model_name = "Qwen/Qwen3-Embedding-0.6B"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(self._model_cache_dir))
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=str(self._model_cache_dir),
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Apply optimizations if available and requested
            if self.config.use_model_optimization and hasattr(torch, "compile"):
                try:
                    logger.info("Applying torch.compile optimization")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    logger.warning(f"Failed to apply torch.compile: {e}")

            self._initialized = True
            logger.info(f"Local embedding service initialized with model: {self.model_name}")

        except ImportError as e:
            raise EmbeddingError(
                f"Required packages not installed. Install with: pip install torch transformers: {e}",
                self.model_name,
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize local embedding service: {e}", self.model_name, e)

    def _select_device(self, preferred_device: Optional[str]) -> str:
        """Select the best available device.

        Args:
            preferred_device: User-specified device preference.

        Returns:
            Selected device string.
        """
        try:
            import torch

            # Check user preference first
            if preferred_device:
                if preferred_device == "mps" and torch.backends.mps.is_available():
                    return "mps"
                elif preferred_device == "cuda" and torch.cuda.is_available():
                    return "cuda"
                elif preferred_device == "cpu":
                    return "cpu"
                else:
                    logger.warning(f"Requested device '{preferred_device}' not available, auto-detecting")

            # Auto-detect best available device
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"

        except ImportError:
            return "cpu"

    def _get_model_cache_path(self) -> Path:
        """Get the cache path for the current model.

        Returns:
            Path to model cache directory.
        """
        # Create a safe directory name from model name
        safe_name = self.model_name.replace("/", "_").replace("\\", "_")
        return self._model_cache_dir / safe_name

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if not self._initialized:
            await self.initialize()

        try:
            all_embeddings = []
            current_batch_size = self.batch_size

            # Create progress bar for embedding generation
            with tqdm(total=len(texts), desc="Generating embeddings", unit="text") as pbar:
                for i in range(0, len(texts), current_batch_size):
                    batch = texts[i : i + current_batch_size]
                    try:
                        # Run in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        embeddings = await loop.run_in_executor(None, self._generate_batch_sync, batch)
                        all_embeddings.extend(embeddings)
                        pbar.update(len(batch))
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Reduce batch size and retry
                            current_batch_size = max(1, current_batch_size // 2)
                            logger.warning(f"OOM error, reducing batch size to {current_batch_size}")

                            # Clear memory if using GPU
                            if self.device != "cpu":
                                import torch

                                if self.device == "cuda":
                                    torch.cuda.empty_cache()
                                elif self.device == "mps":
                                    torch.mps.empty_cache()

                            # Retry with smaller batch
                            smaller_batch = batch[:current_batch_size]
                            embeddings = await loop.run_in_executor(None, self._generate_batch_sync, smaller_batch)
                            all_embeddings.extend(embeddings)
                            pbar.update(len(smaller_batch))

                            # Process remaining items from the batch
                            if len(batch) > current_batch_size:
                                remaining = batch[current_batch_size:]
                                remaining_embeddings = await loop.run_in_executor(
                                    None, self._generate_batch_sync, remaining
                                )
                                all_embeddings.extend(remaining_embeddings)
                                pbar.update(len(remaining))
                        else:
                            raise

            logger.info(f"Generated embeddings for {len(texts)} texts")
            return all_embeddings

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}", self.model_name, e)

    def _generate_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch synchronously.

        Args:
            texts: Batch of text strings.

        Returns:
            List of embedding vectors.
        """
        import torch
        import torch.nn.functional as F

        # Check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            # Use hash of text as cache key
            cache_key = hashlib.md5(text.encode()).hexdigest()
            cached = self._embedding_cache.get(cache_key)
            if cached is not None:
                embeddings.append(cached)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)

        if uncached_texts:
            # Get max sequence length for current model
            max_seq_len = self.MODEL_SPECS.get(self.model_name, {}).get(
                "max_sequence_length", self.config.max_sequence_length
            )

            # Tokenize
            inputs = self.tokenizer(
                uncached_texts,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Handle different model output formats
                if hasattr(outputs, "last_hidden_state"):
                    # Standard transformer output
                    token_embeddings = outputs.last_hidden_state
                elif hasattr(outputs, "pooler_output"):
                    # Some models have a pooler output
                    pooled = outputs.pooler_output
                else:
                    # Fallback to first element
                    token_embeddings = outputs[0]

                # Apply mean pooling if we have token embeddings
                if "pooler_output" not in dir(outputs):
                    attention_mask = inputs["attention_mask"]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                        input_mask_expanded.sum(1), min=1e-9
                    )

                # Normalize embeddings
                pooled = F.normalize(pooled, p=2, dim=1)

            # Convert to list and cache
            new_embeddings = pooled.cpu().numpy().tolist()
            for idx, embedding, text in zip(uncached_indices, new_embeddings, uncached_texts):
                embeddings[idx] = embedding
                # Cache with hash of text as key
                cache_key = hashlib.md5(text.encode()).hexdigest()
                self._embedding_cache.put(cache_key, embedding)

        return embeddings

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    async def test_connection(self) -> bool:
        """Test the embedding service.

        Returns:
            True if service is working, False otherwise.
        """
        try:
            if not self._initialized:
                await self.initialize()
            test_embedding = await self.generate_embedding("test")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Local embedding service test failed: {e}")
            return False

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model.

        Returns:
            Embedding dimension.
        """
        # Return known dimensions or default
        if self.model_name in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.model_name]

        # Try to infer from model if initialized
        if self.model is not None:
            try:
                # Get config from model
                if hasattr(self.model, "config"):
                    if hasattr(self.model.config, "hidden_size"):
                        return self.model.config.hidden_size
                    elif hasattr(self.model.config, "dim"):
                        return self.model.config.dim
            except Exception:
                pass

        # Default dimension
        return 768

    def get_model_info(self) -> Dict:
        """Get information about the current embedding model.

        Returns:
            Dictionary with model information.
        """
        model_spec = self.MODEL_SPECS.get(self.model_name, {})
        return {
            "provider": "local",
            "model": self.model_name,
            "dimension": self.get_embedding_dimension(),
            "max_sequence_length": model_spec.get("max_sequence_length", self.config.max_sequence_length),
            "model_size_gb": model_spec.get("size_gb", "unknown"),
            "description": model_spec.get("description", ""),
            "batch_size": self.batch_size,
            "device": self.device or "not initialized",
            "cache_size": self._embedding_cache.maxsize,
        }
