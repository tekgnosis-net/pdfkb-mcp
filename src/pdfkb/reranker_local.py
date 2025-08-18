"""Local reranking service using Qwen3-Reranker models."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .config import ServerConfig
from .exceptions import EmbeddingError
from .reranker_base import RerankerService

logger = logging.getLogger(__name__)


class LocalRerankerService(RerankerService):
    """Local reranking service using Qwen3-Reranker models."""

    # Supported models with their specifications
    MODEL_SPECS = {
        # Standard models
        "Qwen/Qwen3-Reranker-0.6B": {
            "size_gb": 1.2,
            "description": "Lightweight reranker, fast",
        },
        "Qwen/Qwen3-Reranker-4B": {
            "size_gb": 8.0,
            "description": "High quality reranker",
        },
        "Qwen/Qwen3-Reranker-8B": {
            "size_gb": 16.0,
            "description": "Maximum quality reranker",
        },
        # GGUF quantized models
        "Mungert/Qwen3-Reranker-0.6B-GGUF": {
            "size_gb": 0.3,
            "description": "Quantized lightweight reranker, very fast",
            "is_gguf": True,
        },
        "Mungert/Qwen3-Reranker-4B-GGUF": {
            "size_gb": 2.0,
            "description": "Quantized high quality reranker",
            "is_gguf": True,
        },
        "Mungert/Qwen3-Reranker-8B-GGUF": {
            "size_gb": 4.0,
            "description": "Quantized maximum quality reranker",
            "is_gguf": True,
        },
    }

    def __init__(self, config: ServerConfig):
        """Initialize the local reranker service.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.model_name = config.reranker_model
        self.device = None
        self.model = None
        self.tokenizer = None
        self._model_cache_dir = Path(config.reranker_model_cache_dir).expanduser()
        self._initialized = False
        self._is_gguf = False
        self._gguf_filename = None

        # Default task description for document ranking
        self.task_description = "Given a web search query, retrieve relevant passages that answer the query"

        # Reranker specific attributes (will be set during initialization)
        self.token_true_id = None
        self.token_false_id = None
        self.max_length = 8192
        self.prefix_tokens = None
        self.suffix_tokens = None

    async def initialize(self) -> None:
        """Initialize the model and tokenizer."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Select device
            self.device = self._select_device(self.config.reranker_device)
            logger.info(f"Using device for reranker: {self.device}")

            # Create cache directory
            self._model_cache_dir.mkdir(parents=True, exist_ok=True)

            # Load model and tokenizer
            logger.info(f"Loading reranker model: {self.model_name}")

            # Check if this is a GGUF model
            model_spec = self.MODEL_SPECS.get(self.model_name, {})
            self._is_gguf = model_spec.get("is_gguf", False) or "GGUF" in self.model_name

            try:
                if self._is_gguf and self.config.reranker_gguf_quantization:
                    # Use GGUF quantized model with specific quantization
                    self._gguf_filename = self._get_gguf_filename(
                        self.model_name, self.config.reranker_gguf_quantization
                    )
                    logger.info(f"Using GGUF quantization: {self._gguf_filename}")

                    # Load tokenizer with GGUF file
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=str(self._model_cache_dir),
                        padding_side="left",
                        trust_remote_code=True,
                        gguf_file=self._gguf_filename,
                    )

                    # Load model with GGUF file
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=str(self._model_cache_dir),
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                        trust_remote_code=True,
                        gguf_file=self._gguf_filename,
                    )
                else:
                    # Load standard model
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=str(self._model_cache_dir),
                        padding_side="left",  # Required for reranking
                        trust_remote_code=True,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=str(self._model_cache_dir),
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                        trust_remote_code=True,
                    )

                # Set up token IDs for yes/no scoring
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

                # Set up prefix and suffix for the prompt format
                prefix = (
                    "<|im_start|>system\nJudge whether the Document meets the requirements "
                    "based on the Query and the Instruct provided. Note that the answer can "
                    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
                )
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

            except Exception as e:
                # Fallback to default model if the requested one fails
                logger.warning(f"Failed to load {self.model_name}: {e}. Falling back to Qwen/Qwen3-Reranker-0.6B")
                self.model_name = "Qwen/Qwen3-Reranker-0.6B"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, cache_dir=str(self._model_cache_dir), padding_side="left", trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=str(self._model_cache_dir),
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    trust_remote_code=True,
                )

                # Set up token IDs for yes/no scoring
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

                # Set up prefix and suffix for the prompt format
                prefix = (
                    "<|im_start|>system\nJudge whether the Document meets the requirements "
                    "based on the Query and the Instruct provided. Note that the answer can "
                    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
                )
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            self._initialized = True
            logger.info(f"Local reranker service initialized with model: {self.model_name}")

        except ImportError as e:
            raise EmbeddingError(
                f"Required packages not installed. Install with: pip install torch transformers: {e}",
                self.model_name,
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize local reranker service: {e}", self.model_name, e)

    def _select_device(self, preferred_device: str) -> str:
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

    def _get_gguf_filename(self, model_name: str, quantization: str) -> str:
        """Get the GGUF filename for a given model and quantization.

        Args:
            model_name: The model repository name.
            quantization: The quantization type (e.g., Q6_K, Q8_0).

        Returns:
            The GGUF filename.
        """
        # Extract base model name from the repository
        if "0.6B" in model_name:
            base_name = "Qwen3-Reranker-0.6B"
        elif "4B" in model_name:
            base_name = "Qwen3-Reranker-4B"
        elif "8B" in model_name:
            base_name = "Qwen3-Reranker-8B"
        else:
            # Fallback: try to extract from model name
            base_name = model_name.split("/")[-1].replace("-GGUF", "")

        # Format: ModelName-quantization.gguf (lowercase quantization)
        # Handle special cases like Q6_K -> q6_k_m
        quant_lower = quantization.lower()
        if quant_lower == "q6_k":
            quant_lower = "q6_k_m"
        elif quant_lower == "q4_k":
            quant_lower = "q4_k_m"
        elif quant_lower == "q5_k":
            quant_lower = "q5_k_m"

        return f"{base_name}-{quant_lower}.gguf"

    async def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.

        Returns:
            List of tuples containing (original_index, relevance_score) sorted by relevance.
        """
        if not documents:
            return []

        if not self._initialized:
            await self.initialize()

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._rerank_sync, query, documents)

        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            # Return original order with equal scores as fallback
            return [(i, 1.0) for i in range(len(documents))]

    def _format_instruction(self, query: str, doc: str) -> str:
        """Format query and document with instruction template.

        Args:
            query: The search query
            doc: The document text

        Returns:
            Formatted instruction string
        """
        return f"<Instruct>: {self.task_description}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]):
        """Process input pairs with prefix and suffix tokens.

        Args:
            pairs: List of formatted instruction strings

        Returns:
            Tokenized and padded inputs ready for the model
        """
        # Tokenize without prefix/suffix first
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        # Add prefix and suffix to each input
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens

        # Pad and convert to tensors
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    def _compute_scores(self, inputs) -> List[float]:
        """Compute relevance scores from model outputs.

        Args:
            inputs: Tokenized inputs

        Returns:
            List of relevance scores
        """
        import torch

        with torch.no_grad():
            # Get logits from the model
            batch_scores = self.model(**inputs).logits[:, -1, :]

            # Extract yes/no token scores
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]

            # Stack and apply log_softmax
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)

            # Get the probability of "yes" (relevant)
            scores = batch_scores[:, 1].exp().tolist()

        return scores

    def _rerank_sync(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Synchronous reranking implementation.

        Args:
            query: The search query.
            documents: List of document texts to rerank.

        Returns:
            List of tuples containing (original_index, relevance_score) sorted by relevance.
        """
        # Format each query-document pair
        pairs = [self._format_instruction(query, doc) for doc in documents]

        # Process inputs with prefix/suffix
        inputs = self._process_inputs(pairs)

        # Compute relevance scores
        scores = self._compute_scores(inputs)

        # Create indexed results, handling NaN values
        indexed_scores = []
        for i, score in enumerate(scores):
            # Handle NaN values by setting them to 0
            if score != score:  # NaN check
                logger.warning(f"NaN score for document {i}, setting to 0.0")
                indexed_scores.append((i, 0.0))
            else:
                indexed_scores.append((i, float(score)))

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Reranked {len(documents)} documents. Top score: {indexed_scores[0][1]:.4f}")

        return indexed_scores

    async def test_connection(self) -> bool:
        """Test the reranker service.

        Returns:
            True if service is working, False otherwise.
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Test with simple query and document
            test_results = await self.rerank("test query", ["test document"])
            return len(test_results) > 0

        except Exception as e:
            logger.error(f"Local reranker service test failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get information about the current reranker model.

        Returns:
            Dictionary with model information.
        """
        model_spec = self.MODEL_SPECS.get(self.model_name, {})
        info = {
            "provider": "local",
            "model": self.model_name,
            "model_size_gb": model_spec.get("size_gb", "unknown"),
            "description": model_spec.get("description", ""),
            "device": self.device or "not initialized",
        }
        if self._is_gguf and self._gguf_filename:
            info["gguf_quantization"] = self.config.reranker_gguf_quantization
            info["gguf_filename"] = self._gguf_filename
        return info
