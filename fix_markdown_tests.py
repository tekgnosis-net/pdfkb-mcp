#!/usr/bin/env python3
"""Fix markdown parser tests to use page-aware format."""

import re
from pathlib import Path

test_file = Path("/Users/juanqui/Git/pdfkb-mcp/tests/test_markdown_parser.py")
content = test_file.read_text()

# Replace all result.markdown_content references
replacements = [
    (
        r"assert result\.markdown_content is not None",
        "assert len(result.pages) > 0\n            assert result.pages[0].markdown_content is not None",
    ),
    (r'assert "(.*?)" in result\.markdown_content', r'assert "\1" in result.pages[0].markdown_content'),
    (r'assert result\.markdown_content == "(.*?)"', r'assert result.pages[0].markdown_content == "\1"'),
    (r"len\(result\.markdown_content\)", "len(result.pages[0].markdown_content)"),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

test_file.write_text(content)
print("Fixed markdown parser tests!")
