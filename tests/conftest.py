"""Test fixtures and shims for optional heavy dependencies.

This conftest will inject light-weight stub modules for parts of the
`unstructured.partition` package when they are not importable. That lets
unit tests patch `unstructured.partition.pdf.partition_pdf` (and
`unstructured.partition.text.partition_text`) without requiring the full
heavy optional dependency set to be installed in CI or developer venvs.

The stub only gets installed when the real import fails, so it won't
interfere when `unstructured` is present.
"""

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


def pytest_configure(config):
    # Ensure stubs early during test collection so patch(...) in tests can find
    # the target module path.
    _ensure_unstructured_partition_stubs()
