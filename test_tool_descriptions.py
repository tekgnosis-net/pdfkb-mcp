#!/usr/bin/env python3
"""Test script to validate the improved MCP tool descriptions."""

import os
import tempfile
from pathlib import Path

from src.pdfkb.config import ServerConfig
from src.pdfkb.main import PDFKnowledgebaseServer


def test_tool_descriptions():
    """Test that tool descriptions contain the expected improvements."""
    print("Testing MCP tool descriptions...")

    # Create temp config with proper API key format
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Set environment variables for config (bypass validation)
        os.environ["PDFKB_OPENAI_API_KEY"] = "sk-test1234567890abcdefghijklmnopqrstuvwxyz1234567890abcdefg"
        os.environ["PDFKB_KNOWLEDGEBASE_PATH"] = str(temp_path / "pdfs")
        os.environ["PDFKB_CACHE_DIR"] = str(temp_path / "cache")
        os.environ["PDFKB_EMBEDDING_MODEL"] = "text-embedding-3-small"

        # Load from environment
        config = ServerConfig.from_env()

        # Initialize server (this calls _setup_tools)
        server = PDFKnowledgebaseServer(config)

        # Access the FastMCP app and check tool handlers
        tool_descriptions = {}

        for handler in server.app._tool_handlers:
            tool_descriptions[handler.name] = handler.handler.__doc__ or ""

        # Test search_documents description
        search_desc = tool_descriptions.get("search_documents", "")
        print("\n=== search_documents description ===")
        print(f"Length: {len(search_desc)} characters")

        expected_phrases = [
            "primary tool for finding information",
            "automatically searches through all documents",
            "do NOT need to call list_documents first",
            "entire PDF knowledgebase",
        ]

        for phrase in expected_phrases:
            if phrase.lower() in search_desc.lower():
                print(f"✓ Contains: '{phrase}'")
            else:
                print(f"✗ Missing: '{phrase}'")

        # Test list_documents description
        list_desc = tool_descriptions.get("list_documents", "")
        print("\n=== list_documents description ===")
        print(f"Length: {len(list_desc)} characters")

        expected_phrases = [
            "Use this tool ONLY when",
            "DO NOT use this tool before searching",
            "use search_documents directly instead",
            "management and browsing purposes",
        ]

        for phrase in expected_phrases:
            if phrase.lower() in list_desc.lower():
                print(f"✓ Contains: '{phrase}'")
            else:
                print(f"✗ Missing: '{phrase}'")

        # Test add_document description
        add_desc = tool_descriptions.get("add_document", "")
        print("\n=== add_document description ===")
        print(f"Length: {len(add_desc)} characters")

        expected_phrases = [
            "document becomes immediately available for searching",
            "You do not need to call any other tools after adding",
        ]

        for phrase in expected_phrases:
            if phrase.lower() in add_desc.lower():
                print(f"✓ Contains: '{phrase}'")
            else:
                print(f"✗ Missing: '{phrase}'")

        # Test remove_document description
        remove_desc = tool_descriptions.get("remove_document", "")
        print("\n=== remove_document description ===")
        print(f"Length: {len(remove_desc)} characters")

        expected_phrases = ["use list_documents to browse available documents", "get this from list_documents"]

        for phrase in expected_phrases:
            if phrase.lower() in remove_desc.lower():
                print(f"✓ Contains: '{phrase}'")
            else:
                print(f"✗ Missing: '{phrase}'")

        print("\n=== Summary ===")
        print(f"Found {len(tool_descriptions)} tools:")
        for tool_name in sorted(tool_descriptions.keys()):
            print(f"  - {tool_name}")

        print("\nTool descriptions validation complete!")


if __name__ == "__main__":
    test_tool_descriptions()
