"""Tests for configuration change detection and cache invalidation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from pdfkb.config import ServerConfig
from pdfkb.main import PDFKnowledgebaseServer


class TestConfigurationFingerprinting:
    """Test cases for configuration fingerprinting logic."""

    def test_stage_fingerprints_are_hex(self, tmp_path):
        """Test that stage fingerprints are generated correctly."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="text-embedding-3-small",
            unstructured_pdf_processing_strategy="fast",
            pdf_parser="unstructured",
        )

        fps = [
            config.get_parsing_fingerprint(),
            config.get_chunking_fingerprint(),
            config.get_embedding_fingerprint(),
        ]
        for fp in fps:
            assert len(fp) == 64
            assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_changes_with_critical_params(self, tmp_path):
        """Test that fingerprint changes when critical parameters change."""
        base_config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="text-embedding-3-small",
            unstructured_pdf_processing_strategy="fast",
            pdf_parser="unstructured",
        )

        # These will be used in the fingerprint change tests
        base_parsing_fp = base_config.get_parsing_fingerprint()
        base_chunking_fp = base_config.get_chunking_fingerprint()
        base_embedding_fp = base_config.get_embedding_fingerprint()

        # Test changing each critical parameter
        test_configs = [
            {"chunk_size": 1500, "stage": "chunking"},
            {"chunk_overlap": 100, "stage": "chunking"},
            {"embedding_model": "text-embedding-3-large", "stage": "embedding"},
            {"unstructured_pdf_processing_strategy": "hi_res", "stage": "parsing"},
            {"pdf_parser": "mineru", "stage": "parsing"},
        ]

        for changes in test_configs:
            modified_config = ServerConfig(
                openai_api_key="sk-test-key",
                knowledgebase_path=tmp_path / "pdfs",
                cache_dir=tmp_path / "cache",
                chunk_size=changes.get("chunk_size", 1000),
                chunk_overlap=changes.get("chunk_overlap", 200),
                embedding_model=changes.get("embedding_model", "text-embedding-3-small"),
                unstructured_pdf_processing_strategy=changes.get("unstructured_pdf_processing_strategy", "fast"),
                pdf_parser=changes.get("pdf_parser", "unstructured"),
            )

            stage = changes["stage"]
            if stage == "parsing":
                assert (
                    modified_config.get_parsing_fingerprint() != base_parsing_fp
                ), f"Parsing fingerprint should change when {list(changes.keys())[0]} changes"
            elif stage == "chunking":
                assert (
                    modified_config.get_chunking_fingerprint() != base_chunking_fp
                ), f"Chunking fingerprint should change when {list(changes.keys())[0]} changes"
            elif stage == "embedding":
                assert (
                    modified_config.get_embedding_fingerprint() != base_embedding_fp
                ), f"Embedding fingerprint should change when {list(changes.keys())[0]} changes"

    def test_parallel_processing_config_defaults(self, tmp_path):
        """Test that parallel processing configuration has correct defaults."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
        )

        # Test defaults
        assert config.max_parallel_parsing == 1
        assert config.max_parallel_embedding == 1
        assert config.background_queue_workers == 2
        assert config.thread_pool_size == 1

    def test_parallel_processing_config_from_env(self, tmp_path, monkeypatch):
        """Test that parallel processing configuration can be set from environment."""
        monkeypatch.setenv("PDFKB_MAX_PARALLEL_PARSING", "4")
        monkeypatch.setenv("PDFKB_MAX_PARALLEL_EMBEDDING", "2")
        monkeypatch.setenv("PDFKB_BACKGROUND_QUEUE_WORKERS", "8")
        monkeypatch.setenv("PDFKB_THREAD_POOL_SIZE", "4")

        config = ServerConfig.from_env()
        config.knowledgebase_path = tmp_path / "pdfs"
        config.cache_dir = tmp_path / "cache"

        assert config.max_parallel_parsing == 4
        assert config.max_parallel_embedding == 2
        assert config.background_queue_workers == 8
        assert config.thread_pool_size == 4

    def test_fingerprint_unchanged_with_non_critical_params(self, tmp_path):
        """Test that fingerprint doesn't change with non-critical parameters."""
        config1 = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="text-embedding-3-small",
            unstructured_pdf_processing_strategy="fast",
            pdf_parser="unstructured",
            embedding_batch_size=50,  # Non-critical
            vector_search_k=10,  # Non-critical
            file_scan_interval=30,  # Non-critical
        )

        config2 = ServerConfig(
            openai_api_key="sk-different-key",  # Non-critical
            knowledgebase_path=tmp_path / "different_pdfs",  # Non-critical
            cache_dir=tmp_path / "different_cache",  # Non-critical
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="text-embedding-3-small",
            unstructured_pdf_processing_strategy="fast",
            pdf_parser="unstructured",
            embedding_batch_size=100,  # Different non-critical
            vector_search_k=5,  # Different non-critical
            file_scan_interval=60,  # Different non-critical
        )

        # Non-critical differences should not affect stage fingerprints
        assert config1.get_parsing_fingerprint() == config2.get_parsing_fingerprint()
        assert config1.get_chunking_fingerprint() == config2.get_chunking_fingerprint()
        assert config1.get_embedding_fingerprint() == config2.get_embedding_fingerprint()

    def test_save_and_load_fingerprints(self, tmp_path):
        """Test saving and loading intelligent fingerprints via IntelligentCacheManager."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
        )
        # Update all fingerprints using intelligent cache
        config.update_intelligent_fingerprints()

        cache_manager = config.get_intelligent_cache_manager()
        # Ensure stage fingerprint files exist and contain expected fields
        for stage in ["parsing", "chunking", "embedding"]:
            info = cache_manager.get_stage_fingerprint_info(stage)
            assert info is not None
            assert "fingerprint" in info
            assert "timestamp" in info
            assert "config" in info

    def test_detect_config_changes_first_run(self, tmp_path):
        """Test that first run is detected as changes for all stages."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
        )
        changes = config.detect_config_changes()
        assert changes["parsing"] is True
        assert changes["chunking"] is True
        assert changes["embedding"] is True

    def test_detect_config_changes_same_config(self, tmp_path):
        """Test that same config is detected as unchanged for all stages after saving fingerprints."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
        )
        # Save current stage fingerprints
        config.update_intelligent_fingerprints()

        # Should not detect change
        changes = config.detect_config_changes()
        assert changes["parsing"] is False
        assert changes["chunking"] is False
        assert changes["embedding"] is False

    def test_detect_config_changes_different_config(self, tmp_path):
        """Test that different config is detected as changed on relevant stage."""
        config1 = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
            chunk_size=1000,
        )
        config1.update_intelligent_fingerprints()

        # Create config with different critical parameter (chunking)
        config2 = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
            chunk_size=1500,  # Different chunk size
        )

        changes = config2.detect_config_changes()
        assert changes["chunking"] is True
        # Other stages may show first-run until updated; ensure at least chunking flagged


class TestServerConfigurationChangeHandling:
    """Test cases for server-level configuration change handling."""

    @pytest.mark.asyncio
    async def test_handle_config_changes_first_run(self, tmp_path):
        """Test handling config changes on first run."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
        )

        server = PDFKnowledgebaseServer(config)

        # Mock vector store
        with patch("pdfkb.main.VectorStore") as mock_vector_store_class:
            mock_vector_store = Mock()
            mock_vector_store.initialize = AsyncMock()
            mock_vector_store.reset_database = AsyncMock()
            mock_vector_store.close = AsyncMock()
            mock_vector_store_class.return_value = mock_vector_store

            # Ensure fingerprints directory is empty to simulate first run
            server.cache_manager = config.get_intelligent_cache_manager()
            # Clear any existing stage fingerprints
            server.cache_manager.clear_all_fingerprints()

            await server._handle_intelligent_config_changes()

            # Should reset database on first run
            mock_vector_store.reset_database.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_config_changes_unchanged_config(self, tmp_path):
        """Test handling config changes when config is unchanged."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
        )

        # Save current config to simulate previous run (intelligent fingerprints)
        config.update_intelligent_fingerprints()

        server = PDFKnowledgebaseServer(config)

        # Mock vector store
        with patch("pdfkb.main.VectorStore") as mock_vector_store_class:
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store

            server.cache_manager = config.get_intelligent_cache_manager()
            # Save fingerprints to simulate "unchanged" state
            server.cache_manager.update_fingerprints()
            await server._handle_intelligent_config_changes()

            # Should not create vector store for reset
            mock_vector_store_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_config_changes_clears_caches(self, tmp_path):
        """Test that config changes clear all relevant caches."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
        )

        server = PDFKnowledgebaseServer(config)

        # Create fake cache files
        server._cache_file.parent.mkdir(parents=True, exist_ok=True)
        server._cache_file.write_text('{"fake": "cache"}')

        file_index_path = config.metadata_path / "file_index.json"
        file_index_path.parent.mkdir(parents=True, exist_ok=True)
        file_index_path.write_text('{"fake": "index"}')

        # Mock vector store
        with patch("pdfkb.main.VectorStore") as mock_vector_store_class:
            mock_vector_store = Mock()
            mock_vector_store.initialize = AsyncMock()
            mock_vector_store.reset_database = AsyncMock()
            mock_vector_store.close = AsyncMock()
            mock_vector_store_class.return_value = mock_vector_store

            server.cache_manager = config.get_intelligent_cache_manager()
            await server._handle_intelligent_config_changes()

            # Cache files should be deleted
            assert not server._cache_file.exists()
            assert not file_index_path.exists()

    @pytest.mark.asyncio
    async def test_initialize_saves_fingerprint(self, tmp_path):
        """Test that server initialization saves current config fingerprint."""
        config = ServerConfig(
            openai_api_key="sk-test-key",
            knowledgebase_path=tmp_path / "pdfs",
            cache_dir=tmp_path / "cache",
        )

        server = PDFKnowledgebaseServer(config)

        # Mock all components
        with patch.multiple(
            "pdfkb.main",
            EmbeddingService=Mock(return_value=Mock(initialize=AsyncMock())),
            VectorStore=Mock(
                return_value=Mock(
                    initialize=AsyncMock(),
                    reset_database=AsyncMock(),
                    close=AsyncMock(),
                    set_embedding_service=Mock(),
                )
            ),
            PDFProcessor=Mock(),
            FileMonitor=Mock(return_value=Mock(start_monitoring=AsyncMock())),
        ):
            server._load_document_cache = AsyncMock()

            await server.initialize()

            # Should save intelligent fingerprints after initialization
            cache_manager = server.cache_manager
            assert cache_manager is not None
            # After initialize, fingerprints should be updated and present
            for stage in ["parsing", "chunking", "embedding"]:
                info = cache_manager.get_stage_fingerprint_info(stage)
                assert info is not None
                assert "fingerprint" in info


class TestConfigurationChangeIntegration:
    """Integration tests for configuration change scenarios."""

    @pytest.mark.asyncio
    async def test_chunk_size_change_triggers_reset(self, tmp_path):
        """Test that changing chunk size triggers database reset."""
        # This would be an integration test that:
        # 1. Creates server with initial config
        # 2. Processes some documents
        # 3. Changes chunk size
        # 4. Verifies database is reset and documents are reprocessed
        pass

    @pytest.mark.asyncio
    async def test_embedding_model_change_triggers_reset(self, tmp_path):
        """Test that changing embedding model triggers database reset."""
        # Similar integration test for embedding model changes
        pass
