"""Test fixtures and shims for optional heavy dependencies.

This conftest will inject light-weight stub modules for parts of the
`unstructured.partition` package when they are not importable. That lets
unit tests patch `unstructured.partition.pdf.partition_pdf` (and
`unstructured.partition.text.partition_text`) without requiring the full
heavy optional dependency set to be installed in CI or developer venvs.

The stub only gets installed when the real import fails, so it won't
interfere when `unstructured` is present.
"""

import os
import sys
import types


def _ensure_unstructured_partition_stubs():
    try:
        # If import succeeds, nothing to do
        import importlib

        importlib.import_module("unstructured.partition.pdf")
        importlib.import_module("unstructured.partition.text")
        return
    except Exception:
        # Fall through and install minimal stubs
        pass

    # Ensure parent packages exist so imports and patch targets resolve
    if "unstructured" not in sys.modules:
        pkg = types.ModuleType("unstructured")
        # mark as package
        pkg.__path__ = []
        sys.modules["unstructured"] = pkg

    if "unstructured.partition" not in sys.modules:
        part_pkg = types.ModuleType("unstructured.partition")
        part_pkg.__path__ = []
        sys.modules["unstructured.partition"] = part_pkg

    # Minimal pdf partition stub
    if "unstructured.partition.pdf" not in sys.modules:
        pdf_mod = types.ModuleType("unstructured.partition.pdf")

        def partition_pdf(filename=None, **kwargs):
            # return a minimal list of elements that parser code converts to strings
            return ["__stub_pdf_element_1__", "__stub_pdf_element_2__"]

        pdf_mod.partition_pdf = partition_pdf
        sys.modules["unstructured.partition.pdf"] = pdf_mod

    # Minimal text partition stub (used by chunker fallback)
    if "unstructured.partition.text" not in sys.modules:
        text_mod = types.ModuleType("unstructured.partition.text")

        def partition_text(text=None, **kwargs):
            return ["__stub_text_element_1__", "__stub_text_element_2__"]

    text_mod.partition_text = partition_text
    sys.modules["unstructured.partition.text"] = text_mod


def _ensure_optional_dependency_stubs():
    """Install lightweight stubs for other optional packages that tests
    often import at collection time. These are only installed when the
    real package isn't available.
    """
    import importlib

    # whoosh: used by TextIndex. Prefer real whoosh, but provide a tiny
    # fallback that exposes the minimal API used in tests to avoid
    # collection-time ImportError.
    try:
        importlib.import_module("whoosh")
    except Exception:

        if "whoosh" not in sys.modules:
            whoosh = types.ModuleType("whoosh")
            sys.modules["whoosh"] = whoosh

        # whoosh.index
        if "whoosh.index" not in sys.modules:
            idx_mod = types.ModuleType("whoosh.index")

            class DummyIndex:
                def __init__(self, path=None, schema=None):
                    self.schema = schema

                def writer(self):
                    class W:
                        def update_document(self, **kwargs):
                            return None

                        def commit(self):
                            return None

                        def cancel(self):
                            return None

                        def delete_by_term(self, *args, **kwargs):
                            return None

                    return W()

                def searcher(self, *args, **kwargs):
                    class S:
                        def __enter__(self):
                            return self

                        def __exit__(self, exc_type, exc, tb):
                            return False

                        def search(self, *args, **kwargs):
                            return []

                        def documents(self):
                            return []

                        def doc_count_all(self):
                            return 0

                    return S()

                def close(self):
                    return None

            def exists_in(path):
                return False

            def create_in(path, schema):
                return DummyIndex(path, schema)

            def open_dir(path):
                return DummyIndex(path)

            idx_mod.exists_in = exists_in
            idx_mod.create_in = create_in
            idx_mod.open_dir = open_dir
            sys.modules["whoosh.index"] = idx_mod

        # whoosh.analysis (provide analyzers used in schema selection)
        try:
            importlib.import_module("whoosh.analysis")
        except Exception:
            ana_mod = types.ModuleType("whoosh.analysis")

            def StandardAnalyzer():
                return lambda x: x

            def StemmingAnalyzer():
                return lambda x: x

            ana_mod.StandardAnalyzer = StandardAnalyzer
            ana_mod.StemmingAnalyzer = StemmingAnalyzer
            sys.modules["whoosh.analysis"] = ana_mod

        # whoosh.fields and other small pieces
        try:
            importlib.import_module("whoosh.fields")
        except Exception:
            fields_mod = types.ModuleType("whoosh.fields")

            def ID(**kwargs):
                return str

            def NUMERIC(**kwargs):
                return int

            def STORED():
                return dict

            def TEXT(**kwargs):
                return str

            class Schema(dict):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

            fields_mod.ID = ID
            fields_mod.NUMERIC = NUMERIC
            fields_mod.STORED = STORED
            fields_mod.TEXT = TEXT
            fields_mod.Schema = Schema
            sys.modules["whoosh.fields"] = fields_mod

        # whoosh.qparser and scoring minimal fallbacks
        try:
            importlib.import_module("whoosh.qparser")
        except Exception:
            qp = types.ModuleType("whoosh.qparser")

            class QueryParser:
                def __init__(self, field, schema):
                    self.field = field

                def parse(self, q):
                    return q

            qp.QueryParser = QueryParser
            sys.modules["whoosh.qparser"] = qp

        try:
            importlib.import_module("whoosh.scoring")
        except Exception:
            sc = types.ModuleType("whoosh.scoring")

            class BM25F:
                pass

            sc.BM25F = BM25F
            sys.modules["whoosh.scoring"] = sc

    # aiohttp: some modules import aiohttp at top-level; provide a light stub
    try:
        importlib.import_module("aiohttp")
    except Exception:
        aio = types.ModuleType("aiohttp")
        sys.modules["aiohttp"] = aio

    # langchain_core.embeddings. Tests import Embeddings base class.
    try:
        importlib.import_module("langchain_core.embeddings")
    except Exception:
        lc = types.ModuleType("langchain_core")
        emb = types.ModuleType("langchain_core.embeddings")

        class Embeddings:
            def embed_documents(self, docs):
                return [[0.0] * 1 for _ in docs]

        emb.Embeddings = Embeddings
        sys.modules["langchain_core"] = lc
        sys.modules["langchain_core.embeddings"] = emb

    # marker: create package placeholder so tests can patch internals
    try:
        importlib.import_module("marker")
    except Exception:
        mark = types.ModuleType("marker")
        mark.__path__ = []
        sys.modules["marker"] = mark

        # submodules used by parser_marker tests
        mod_names = [
            "marker.config",
            "marker.config.parser",
            "marker.converters",
            "marker.converters.pdf",
            "marker.models",
            "marker.output",
        ]
        for name in mod_names:
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m

        # provide placeholders for attributes tests patch
        try:
            sys.modules["marker.config.parser"].ConfigParser = object
        except Exception:
            pass

        try:
            sys.modules["marker.converters.pdf"].PdfConverter = object
        except Exception:
            pass

        try:
            sys.modules["marker.models"].create_model_dict = lambda *a, **k: {}
        except Exception:
            pass

        try:
            sys.modules["marker.output"].text_from_rendered = lambda *a, **k: ""
        except Exception:
            pass

    # reportlab: tests may skip on importorskip; provide minimal module to avoid skips
    try:
        importlib.import_module("reportlab")
    except Exception:
        rep = types.ModuleType("reportlab")
        # provide pdfgen.canvas submodule used in tests
        pdfgen = types.ModuleType("reportlab.pdfgen")

        class DummyCanvas:
            def __init__(self, *a, **k):
                pass

            def save(self):
                return None

            def drawString(self, x, y, text):
                # minimal implementation used by tests that render simple PDFs
                return None

        canvas_mod = types.ModuleType("reportlab.pdfgen.canvas")
        canvas_mod.Canvas = DummyCanvas

        sys.modules["reportlab"] = rep
        sys.modules["reportlab.pdfgen"] = pdfgen
        sys.modules["reportlab.pdfgen.canvas"] = canvas_mod

    # docling: some integration tests patch objects inside docling; provide
    # minimal stub modules so those patch targets can be resolved when the
    # real package isn't installed.
    try:
        importlib.import_module("docling")
    except Exception:
        doc = types.ModuleType("docling")
        doc.__path__ = []
        # Mark this as a test stub so integration checks can detect it's not a real install
        setattr(doc, "__pdfkb_stub__", True)
        sys.modules["docling"] = doc

        # Submodules used in tests
        submods = [
            "docling.document_converter",
            "docling.datamodel",
            "docling.datamodel.pipeline_options",
        ]
        for name in submods:
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m

        # Provide a dummy PdfPipelineOptions class so tests that patch it
        # can resolve the target.
        try:

            class PdfPipelineOptions:
                def __init__(self, *a, **k):
                    pass

            sys.modules["docling.datamodel.pipeline_options"].PdfPipelineOptions = PdfPipelineOptions
        except Exception:
            pass

        # Provide other minimal attributes used by tests/patch targets
        try:
            # DocumentConverter is patched in several tests
            # Provide a minimal DocumentConverter with a `convert` method that
            # returns an object similar to docling's ConversionResult so parser
            # code can call `conversion_result.document.export_to_markdown()`.
            class DummyConversionDocument:
                def __init__(self):
                    self.pages = []

                def export_to_markdown(self):
                    return "# Docling Stub\n\nConverted"

            class DummyConversionResult:
                def __init__(self):
                    self.status = getattr(self, "status", None)
                    self.document = DummyConversionDocument()

            class DummyDocumentConverter:
                def __init__(self, *a, **k):
                    pass

                def convert(self, path):
                    # Return a simple ConversionResult-like object
                    return DummyConversionResult()

            sys.modules["docling.document_converter"].DocumentConverter = DummyDocumentConverter
        except Exception:
            pass

        try:
            # PdfFormatOption referenced in some code paths
            sys.modules["docling.document_converter"].PdfFormatOption = object
        except Exception:
            pass

        try:
            # InputFormat used by parser code
            if "docling.datamodel.base_models" not in sys.modules:
                sys.modules["docling.datamodel.base_models"] = types.ModuleType("docling.datamodel.base_models")

            # Provide a simple enum-like InputFormat with PDF attribute so
            # parser code that references InputFormat.PDF works with the stub.
            class DummyInputFormat:
                PDF = "pdf"

            sys.modules["docling.datamodel.base_models"].InputFormat = DummyInputFormat
        except Exception:
            pass

        try:
            # Some versions expose OCR option classes - provide placeholder
            sys.modules["docling.datamodel.pipeline_options"].TesseractOcrOptions = object
        except Exception:
            pass

    # langchain_text_splitters: provide light-weight local fallback used by
    # LangChainChunker when langchain_text_splitters isn't installed.
    try:
        importlib.import_module("langchain_text_splitters")
    except Exception:
        lts = types.ModuleType("langchain_text_splitters")

        class MarkdownHeaderTextSplitter:
            def __init__(self, headers_to_split_on=None):
                self.headers = headers_to_split_on or [("#", "Header 1")]

            def split_text(self, text: str):
                # Very small heuristic: split on top-level headers
                import re

                pattern = r"(?m)^(# .*)$"
                parts = []
                if not text:
                    return []
                # If headers present, split preserving header lines
                headers = re.split(pattern, text)
                # Rejoin pairs (header, body)
                i = 0
                if len(headers) == 1:
                    return [text]
                while i < len(headers):
                    if i + 1 < len(headers):
                        hdr = headers[i].strip()
                        body = headers[i + 1].strip()
                        combined = f"{hdr}\n\n{body}" if hdr else body
                        parts.append(combined)
                        i += 2
                    else:
                        parts.append(headers[i].strip())
                        i += 1
                return parts

        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str):
                if not text:
                    return []
                chunks = []
                i = 0
                while i < len(text):
                    chunk = text[i : i + self.chunk_size]
                    chunks.append(chunk)
                    i += self.chunk_size - self.chunk_overlap
                return chunks

        lts.MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter
        lts.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        sys.modules["langchain_text_splitters"] = lts

    # aiohttp: provide a minimal ClientSession implementation used by tests that patch
    # higher-level behaviors rather than exercising network.
    try:
        importlib.import_module("aiohttp")
    except Exception:
        aio = types.ModuleType("aiohttp")

        class DummyClientSession:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *a, **k):
                return types.SimpleNamespace(status=200, text=lambda: "{}")

            async def get(self, *a, **k):
                return types.SimpleNamespace(status=200, text=lambda: "{}")

        sys.modules["aiohttp"].ClientSession = DummyClientSession


def pytest_configure(config):
    # Ensure stubs early during test collection so patch(...) in tests can find
    # the target module path.
    _ensure_unstructured_partition_stubs()
    _ensure_optional_dependency_stubs()
    # Force CPU for embedding tests in CI/dev environments to avoid CUDA OOMs
    # when tests are run on machines with busy GPUs. Tests that need GPU can
    # override this explicitly.
    os.environ.setdefault("PDFKB_EMBEDDING_DEVICE", "cpu")
