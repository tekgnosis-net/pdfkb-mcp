"""Document parser implementations (PDF and Markdown support)."""

from .parser import DocumentParser, ParseResult, PDFParser  # PDFParser is backward compat alias
from .parser_docling import DoclingParser
from .parser_llm import LLMParser
from .parser_markdown import MarkdownParser
from .parser_marker import MarkerPDFParser
from .parser_mineru import MinerUPDFParser
from .parser_pymupdf4llm import PyMuPDF4LLMParser
from .parser_unstructured import UnstructuredPDFParser

__all__ = [
    "DocumentParser",
    "PDFParser",  # Backward compatibility alias
    "ParseResult",
    "MarkdownParser",
    "UnstructuredPDFParser",
    "PyMuPDF4LLMParser",
    "MinerUPDFParser",
    "MarkerPDFParser",
    "DoclingParser",
    "LLMParser",
]
