"""Tests for the Markdown parser implementation."""

import tempfile
from pathlib import Path

import pytest

from src.pdfkb.parsers.parser_markdown import MarkdownParser


class TestMarkdownParser:
    """Test suite for the Markdown parser."""

    @pytest.fixture
    def parser(self):
        """Create a markdown parser instance."""
        return MarkdownParser()

    @pytest.fixture
    def sample_markdown(self):
        """Sample markdown content for testing."""
        return """---
title: Test Document
author: Test Author
date: 2024-01-01
tags: [test, markdown, parser]
---

# Introduction

This is a test document for the markdown parser.

## Section 1

Some content in section 1 with **bold** and *italic* text.

### Subsection 1.1

- List item 1
- List item 2
- List item 3

## Section 2

```python
def hello_world():
    print("Hello, World!")
```

A [link](https://example.com) to somewhere.
"""

    @pytest.fixture
    def sample_markdown_no_frontmatter(self):
        """Sample markdown without frontmatter."""
        return """# My Document Title

This is a document without frontmatter.

## Content Section

Some regular content here.
"""

    @pytest.mark.asyncio
    async def test_parse_with_frontmatter(self, parser, sample_markdown):
        """Test parsing markdown with YAML frontmatter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(sample_markdown)
            temp_path = Path(f.name)

        try:
            result = await parser.parse(temp_path)

            # Check content is returned
            assert len(result.pages) > 0
            assert result.pages[0].markdown_content is not None
            assert "# Introduction" in result.pages[0].markdown_content
            assert "This is a test document" in result.pages[0].markdown_content

            # Check frontmatter was parsed
            assert result.metadata["title"] == "Test Document"
            assert result.metadata["author"] == "Test Author"
            # YAML may parse date as date object or string
            assert str(result.metadata["date"]) == "2024-01-01"
            assert result.metadata["tags"] == ["test", "markdown", "parser"]

            # Check file metadata
            assert result.metadata["document_type"] == "markdown"
            assert result.metadata["source_filename"] == temp_path.name

            # Check statistics
            assert result.metadata["line_count"] > 0
            assert result.metadata["word_count"] > 0
            assert result.metadata["heading_count"] == 4  # h1, h2, h3 headers
            assert result.metadata["code_block_count"] == 1
            assert result.metadata["link_count"] == 1

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_parse_without_frontmatter(self, sample_markdown_no_frontmatter):
        """Test parsing markdown without frontmatter."""
        parser = MarkdownParser(config={"parse_frontmatter": True, "extract_title": True})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(sample_markdown_no_frontmatter)
            temp_path = Path(f.name)

        try:
            result = await parser.parse(temp_path)

            # Check content
            assert "# My Document Title" in result.pages[0].markdown_content
            assert "This is a document without frontmatter" in result.pages[0].markdown_content

            # Check title extraction from H1
            assert result.metadata.get("title") == "My Document Title"

            # Check no frontmatter metadata
            assert "author" not in result.metadata
            assert "date" not in result.metadata

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_parse_with_disabled_frontmatter(self, sample_markdown):
        """Test parsing with frontmatter parsing disabled."""
        parser = MarkdownParser(config={"parse_frontmatter": False})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(sample_markdown)
            temp_path = Path(f.name)

        try:
            result = await parser.parse(temp_path)

            # Check frontmatter is included in content when parsing is disabled
            assert "---" in result.pages[0].markdown_content
            assert "title: Test Document" in result.pages[0].markdown_content

            # Check no frontmatter in metadata
            assert "title" not in result.metadata or result.metadata["title"] != "Test Document"
            assert "author" not in result.metadata

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_parse_with_toml_frontmatter(self):
        """Test parsing markdown with TOML frontmatter."""
        content = """+++
title = "TOML Document"
author = "TOML Author"
+++

# Content

This uses TOML frontmatter.
"""
        parser = MarkdownParser()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            result = await parser.parse(temp_path)

            # TOML parsing requires toml library, might not work without it
            # But content should still be returned
            assert "# Content" in result.pages[0].markdown_content
            assert "This uses TOML frontmatter" in result.pages[0].markdown_content

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_parse_empty_file(self, parser):
        """Test parsing an empty markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            result = await parser.parse(temp_path)

            assert result.pages[0].markdown_content == ""
            assert result.metadata["line_count"] == 1  # Empty string splits to ['']
            assert result.metadata["word_count"] == 0
            assert result.metadata["char_count"] == 0

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self, parser):
        """Test parsing a nonexistent file raises an error."""
        nonexistent_path = Path("/tmp/nonexistent_file_12345.md")

        with pytest.raises(Exception):
            await parser.parse(nonexistent_path)

    def test_extract_title_from_content(self, parser):
        """Test title extraction from various content formats."""
        # Test H1 extraction
        content1 = "# Main Title\nSome content"
        title1 = parser._extract_title_from_content(content1)
        assert title1 == "Main Title"

        # Test with formatting in title
        content2 = "# Title with **bold** and *italic*"
        title2 = parser._extract_title_from_content(content2)
        assert title2 == "Title with bold and italic"

        # Test with link in title
        content3 = "# Title with [link](url)"
        title3 = parser._extract_title_from_content(content3)
        assert title3 == "Title with link"

        # Test no H1 header
        content4 = "## Only H2\nSome content"
        title4 = parser._extract_title_from_content(content4)
        # Should return None when no H1 is present
        assert title4 is None

        # Test empty content
        content5 = ""
        title5 = parser._extract_title_from_content(content5)
        assert title5 is None
