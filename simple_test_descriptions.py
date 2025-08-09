#!/usr/bin/env python3
"""Simple test to validate the improved MCP tool descriptions."""


def test_tool_descriptions():
    """Test that tool descriptions contain the expected improvements by examining the source code."""
    print("Testing MCP tool descriptions in source code...")

    # Read the main.py file
    with open("src/pdfkb/main.py", "r") as f:
        content = f.read()

    # Test search_documents description improvements
    print("\n=== Testing search_documents description ===")
    expected_phrases = [
        "primary tool for finding information",
        "automatically searches through",
        "you do NOT need to",
        "entire PDF knowledgebase",
    ]

    search_found = 0
    for phrase in expected_phrases:
        if phrase.lower() in content.lower():
            print(f"✓ Contains: '{phrase}'")
            search_found += 1
        else:
            print(f"✗ Missing: '{phrase}'")

    # Test list_documents description improvements
    print("\n=== Testing list_documents description ===")
    expected_phrases = [
        "Use this tool ONLY when",
        "DO NOT use this tool before searching",
        "use search_documents directly instead",
        "management and browsing purposes",
    ]

    list_found = 0
    for phrase in expected_phrases:
        if phrase.lower() in content.lower():
            print(f"✓ Contains: '{phrase}'")
            list_found += 1
        else:
            print(f"✗ Missing: '{phrase}'")

    # Test add_document description improvements
    print("\n=== Testing add_document description ===")
    expected_phrases = ["immediately available for searching", "You do not need to call any other tools after adding"]

    add_found = 0
    for phrase in expected_phrases:
        if phrase.lower() in content.lower():
            print(f"✓ Contains: '{phrase}'")
            add_found += 1
        else:
            print(f"✗ Missing: '{phrase}'")

    # Test remove_document description improvements
    print("\n=== Testing remove_document description ===")
    expected_phrases = ["use list_documents to browse", "get this from list_documents"]

    remove_found = 0
    for phrase in expected_phrases:
        if phrase.lower() in content.lower():
            print(f"✓ Contains: '{phrase}'")
            remove_found += 1
        else:
            print(f"✗ Missing: '{phrase}'")

    # Summary
    print("\n=== Summary ===")
    print(f"search_documents: {search_found}/4 expected phrases found")
    print(f"list_documents: {list_found}/4 expected phrases found")
    print(f"add_document: {add_found}/2 expected phrases found")
    print(f"remove_document: {remove_found}/2 expected phrases found")

    total_found = search_found + list_found + add_found + remove_found
    total_expected = 12

    if total_found == total_expected:
        print(f"\n🎉 SUCCESS: All {total_found}/{total_expected} expected improvements found!")
        return True
    else:
        print(f"\n❌ PARTIAL: {total_found}/{total_expected} expected improvements found")
        return False


if __name__ == "__main__":
    success = test_tool_descriptions()
    exit(0 if success else 1)
