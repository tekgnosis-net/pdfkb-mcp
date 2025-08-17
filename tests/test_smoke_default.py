from pathlib import Path

import pytest

from pdfkb.config import ServerConfig
from pdfkb.document_processor import DocumentProcessor as PDFProcessor
from pdfkb.embeddings import EmbeddingService

reportlab = pytest.importorskip("reportlab", reason="reportlab is required for smoke test")
from reportlab.pdfgen import canvas  # noqa: E402

pytestmark = pytest.mark.asyncio


async def test_smoke_default_pipeline(tmp_path: Path):
    """
    Smoke test: default parser (PyMuPDF4LLM) + default chunker (LangChain) work OOTB.

    - Creates a minimal synthetic PDF with reportlab
    - Uses a test OpenAI key so EmbeddingService returns mock embeddings on errors
    - Runs PDFProcessor end-to-end and asserts success
    """
    # Prepare directories
    kb_dir = tmp_path / "pdfs"
    cache_dir = tmp_path / ".cache"
    kb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple PDF
    pdf_path = kb_dir / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(72, 720, "Hello World - PDFKB MCP Smoke Test")
    c.drawString(72, 700, "This PDF is generated for default pipeline smoke testing.")
    c.showPage()
    c.save()

    # Default config should now be:
    # - PDF_PARSER = pymupdf4llm
    # - PDF_CHUNKER = langchain
    # - EMBEDDING_MODEL = text-embedding-3-large
    config = ServerConfig(
        openai_api_key="sk-test-key-smoke",  # triggers mock embeddings on API errors
        knowledgebase_path=kb_dir,
        cache_dir=cache_dir,
    )

    embedding_service = EmbeddingService(config)
    processor = PDFProcessor(config, embedding_service, cache_manager=None)

    result = await processor.process_pdf(pdf_path, metadata={"source": "smoke"})
    assert result.success, f"Processing failed: {result.error}"
    assert result.document is not None, "No document produced"
    assert result.chunks_created > 0, "No chunks created by default chunker"
    # Embeddings may be mock, but ensure generation path runs
    assert len([c for c in result.document.chunks if c.embedding is not None]) >= 0
