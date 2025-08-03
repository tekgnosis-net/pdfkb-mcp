"""Tests for the intelligent cache management system."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pdfkb.config import ServerConfig
from src.pdfkb.intelligent_cache import IntelligentCacheManager


class TestIntelligentCacheManager:
    """Test cases for IntelligentCacheManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"

        # Mock environment variables for ServerConfig
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-123",
            },
        ):
            self.config = ServerConfig.from_env()
            self.config.cache_dir = self.cache_dir

        self.cache_manager = IntelligentCacheManager(self.config, self.cache_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_step_specific_fingerprints(self):
        """Test that step-specific fingerprints are generated correctly."""
        # Test parsing fingerprint
        parsing_fp = self.cache_manager.get_parsing_fingerprint()
        assert isinstance(parsing_fp, str)
        assert len(parsing_fp) == 64  # SHA-256 hex length

        # Test chunking fingerprint
        chunking_fp = self.cache_manager.get_chunking_fingerprint()
        assert isinstance(chunking_fp, str)
        assert len(chunking_fp) == 64

        # Test embedding fingerprint
        embedding_fp = self.cache_manager.get_embedding_fingerprint()
        assert isinstance(embedding_fp, str)
        assert len(embedding_fp) == 64

        # Fingerprints should be different for different config aspects
        assert parsing_fp != chunking_fp
        assert chunking_fp != embedding_fp
        assert parsing_fp != embedding_fp

    def test_fingerprint_consistency(self):
        """Test that fingerprints are consistent for same configuration."""
        # Get fingerprints multiple times
        parsing_fp1 = self.cache_manager.get_parsing_fingerprint()
        parsing_fp2 = self.cache_manager.get_parsing_fingerprint()

        chunking_fp1 = self.cache_manager.get_chunking_fingerprint()
        chunking_fp2 = self.cache_manager.get_chunking_fingerprint()

        embedding_fp1 = self.cache_manager.get_embedding_fingerprint()
        embedding_fp2 = self.cache_manager.get_embedding_fingerprint()

        # Should be identical
        assert parsing_fp1 == parsing_fp2
        assert chunking_fp1 == chunking_fp2
        assert embedding_fp1 == embedding_fp2

    def test_fingerprint_changes_with_config(self):
        """Test that fingerprints change when configuration changes."""
        # Get initial fingerprints
        initial_parsing = self.cache_manager.get_parsing_fingerprint()
        initial_chunking = self.cache_manager.get_chunking_fingerprint()
        initial_embedding = self.cache_manager.get_embedding_fingerprint()

        # Change parsing config
        self.config.pdf_parser = "pymupdf4llm"
        new_parsing = self.cache_manager.get_parsing_fingerprint()
        assert new_parsing != initial_parsing

        # Change chunking config
        self.config.chunk_size = 2000
        new_chunking = self.cache_manager.get_chunking_fingerprint()
        assert new_chunking != initial_chunking

        # Change embedding config
        self.config.embedding_model = "text-embedding-3-small"
        new_embedding = self.cache_manager.get_embedding_fingerprint()
        assert new_embedding != initial_embedding

    def test_detect_config_changes_first_run(self):
        """Test change detection on first run (no saved fingerprints)."""
        changes = self.cache_manager.detect_config_changes()

        # All should be True on first run
        assert changes["parsing"] is True
        assert changes["chunking"] is True
        assert changes["embedding"] is True

    def test_detect_config_changes_after_save(self):
        """Test change detection after saving fingerprints."""
        # Save fingerprints
        self.cache_manager.update_fingerprints()

        # Check changes - should be False now
        changes = self.cache_manager.detect_config_changes()
        assert changes["parsing"] is False
        assert changes["chunking"] is False
        assert changes["embedding"] is False

    def test_detect_config_changes_after_modification(self):
        """Test change detection after modifying configuration."""
        # Save initial fingerprints
        self.cache_manager.update_fingerprints()

        # Modify parsing config
        self.config.pdf_parser = "pymupdf4llm"

        changes = self.cache_manager.detect_config_changes()
        assert changes["parsing"] is True
        assert changes["chunking"] is False  # Should be unchanged
        assert changes["embedding"] is False  # Should be unchanged

    def test_fingerprint_file_structure(self):
        """Test that fingerprint files are created with correct structure."""
        self.cache_manager.update_fingerprints()

        # Check that files exist
        parsing_file = self.cache_manager._get_fingerprint_path("parsing")
        chunking_file = self.cache_manager._get_fingerprint_path("chunking")
        embedding_file = self.cache_manager._get_fingerprint_path("embedding")

        assert parsing_file.exists()
        assert chunking_file.exists()
        assert embedding_file.exists()

        # Check file contents
        with open(parsing_file, "r") as f:
            parsing_data = json.load(f)

        assert "fingerprint" in parsing_data
        assert "timestamp" in parsing_data
        assert "config_version" in parsing_data
        assert "config" in parsing_data
        assert parsing_data["config_version"] == "1.0.0"
        assert "pdf_parser" in parsing_data["config"]
        assert "unstructured_pdf_processing_strategy" in parsing_data["config"]

    def test_corrupted_fingerprint_handling(self):
        """Test handling of corrupted fingerprint files."""
        # Create corrupted fingerprint file
        fingerprint_path = self.cache_manager._get_fingerprint_path("parsing")
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(fingerprint_path, "w") as f:
            f.write("invalid json content")

        # Should handle gracefully
        changes = self.cache_manager.detect_config_changes()
        assert changes["parsing"] is True  # Should treat as changed

    def test_cache_validation_methods(self):
        """Test cache validation methods."""
        # Initially should be invalid (no fingerprints saved)
        assert not self.cache_manager.is_parsing_cache_valid("test_doc")
        assert not self.cache_manager.is_chunking_cache_valid("test_doc")
        assert not self.cache_manager.is_embedding_cache_valid("test_doc")

        # After saving fingerprints, should be valid
        self.cache_manager.update_fingerprints()
        assert self.cache_manager.is_parsing_cache_valid("test_doc")
        assert self.cache_manager.is_chunking_cache_valid("test_doc")
        assert self.cache_manager.is_embedding_cache_valid("test_doc")

    def test_clear_fingerprints(self):
        """Test clearing fingerprint files."""
        # Save fingerprints
        self.cache_manager.update_fingerprints()

        # Clear one stage
        self.cache_manager.clear_stage_fingerprint("parsing")

        changes = self.cache_manager.detect_config_changes()
        assert changes["parsing"] is True
        assert changes["chunking"] is False
        assert changes["embedding"] is False

        # Clear all
        self.cache_manager.clear_all_fingerprints()

        changes = self.cache_manager.detect_config_changes()
        assert changes["parsing"] is True
        assert changes["chunking"] is True
        assert changes["embedding"] is True


class TestServerConfigIntegration:
    """Test ServerConfig integration with IntelligentCacheManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-123",
            },
        ):
            self.config = ServerConfig.from_env()
            self.config.cache_dir = self.temp_dir / "cache"

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_config_step_specific_methods(self):
        """Test that ServerConfig step-specific methods work."""
        parsing_fp = self.config.get_parsing_fingerprint()
        chunking_fp = self.config.get_chunking_fingerprint()
        embedding_fp = self.config.get_embedding_fingerprint()

        assert isinstance(parsing_fp, str)
        assert isinstance(chunking_fp, str)
        assert isinstance(embedding_fp, str)
        assert len(parsing_fp) == 64
        assert len(chunking_fp) == 64
        assert len(embedding_fp) == 64

    def test_config_change_detection_methods(self):
        """Test ServerConfig change detection methods."""
        changes = self.config.detect_config_changes()
        assert isinstance(changes, dict)
        assert "parsing" in changes
        assert "chunking" in changes
        assert "embedding" in changes

        # Test individual change methods
        assert self.config.has_parsing_config_changed()
        assert self.config.has_chunking_config_changed()
        assert self.config.has_embedding_config_changed()

    def test_backward_compatibility(self):
        """Test that current methods work correctly."""
        # Test that all new methods work
        parsing_fp = self.config.get_parsing_fingerprint()
        chunking_fp = self.config.get_chunking_fingerprint()
        embedding_fp = self.config.get_embedding_fingerprint()

        assert isinstance(parsing_fp, str)
        assert isinstance(chunking_fp, str)
        assert isinstance(embedding_fp, str)
        assert len(parsing_fp) == 64
        assert len(chunking_fp) == 64
        assert len(embedding_fp) == 64

    def test_intelligent_fingerprint_update(self):
        """Test the new intelligent fingerprint update method."""
        # Should not raise any errors
        self.config.update_intelligent_fingerprints()

        # After updating, changes should be False
        changes = self.config.detect_config_changes()
        assert not changes["parsing"]
        assert not changes["chunking"]
        assert not changes["embedding"]


def test_integration_with_real_config():
    """Test integration with a real configuration scenario."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "cache"

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key-123",
                "CHUNK_SIZE": "1500",
                "PDF_PARSER": "mineru",
                "EMBEDDING_MODEL": "text-embedding-3-small",
            },
        ):
            config = ServerConfig.from_env()
            config.cache_dir = cache_dir

            # Test that all methods work with real config
            cache_manager = config.get_intelligent_cache_manager()

            # Get fingerprints
            cache_manager.get_parsing_fingerprint()
            cache_manager.get_chunking_fingerprint()
            cache_manager.get_embedding_fingerprint()

            # Save fingerprints
            cache_manager.update_fingerprints()

            # Verify files were created
            fingerprints_dir = cache_dir / "metadata" / "fingerprints"
            assert (fingerprints_dir / "parsing.json").exists()
            assert (fingerprints_dir / "chunking.json").exists()
            assert (fingerprints_dir / "embedding.json").exists()

            # Test change detection
            changes = cache_manager.detect_config_changes()
            assert not any(changes.values())  # No changes after saving


if __name__ == "__main__":
    pytest.main([__file__])
