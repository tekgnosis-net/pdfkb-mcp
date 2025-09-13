"""Tests for summarizer base classes."""

import pytest

from pdfkb.summarizer_base import DocumentSummary


class TestDocumentSummary:
    """Test DocumentSummary data class."""

    def test_valid_summary(self):
        """Test creating a valid document summary."""
        summary = DocumentSummary(
            title="Test Document",
            short_description="A test document for validation",
            long_description=(
                "This is a longer description that provides more detail about the test document content and purpose."
            ),
        )

        assert summary.title == "Test Document"
        assert summary.short_description == "A test document for validation"
        assert "longer description" in summary.long_description
        assert summary.confidence is None

    def test_summary_with_confidence(self):
        """Test creating a summary with confidence score."""
        summary = DocumentSummary(
            title="Test Document",
            short_description="A test document",
            long_description="Detailed description of the test document.",
            confidence=0.95,
        )

        assert summary.confidence == 0.95

    def test_empty_title_validation(self):
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="Title cannot be empty"):
            DocumentSummary(
                title="",
                short_description="A test document",
                long_description="Detailed description",
            )

    def test_whitespace_title_validation(self):
        """Test that whitespace-only title raises ValueError."""
        with pytest.raises(ValueError, match="Title cannot be empty"):
            DocumentSummary(
                title="   ",
                short_description="A test document",
                long_description="Detailed description",
            )

    def test_empty_short_description_validation(self):
        """Test that empty short description raises ValueError."""
        with pytest.raises(ValueError, match="Short description cannot be empty"):
            DocumentSummary(
                title="Test Document",
                short_description="",
                long_description="Detailed description",
            )

    def test_empty_long_description_validation(self):
        """Test that empty long description raises ValueError."""
        with pytest.raises(ValueError, match="Long description cannot be empty"):
            DocumentSummary(
                title="Test Document",
                short_description="A test document",
                long_description="",
            )
