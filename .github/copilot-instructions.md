# Copilot / AI agent instructions for pdfkb-mcp

This file captures the essential, discoverable knowledge an AI coding agent needs to be immediately productive in this repository.

Keep it concise. When in doubt, open the files referenced below before making edits.

1) Big picture (why & architecture)
- This project is a Python-based MCP server (ASGI) plus a lightweight web UI and a set of optional PDF parser integrations.
- Major components:
  - `src/pdfkb/` — core application code (server, background queue, parsers, embeddings, vector store).
  - `src/webui/` — optional web frontend static assets and UI glue.
  - `tests/` — pytest-based test suite (unit/integration/smoke).
  - Dockerfile / docker-entrypoint.sh — container image and runtime setup logic (parser isolation lives here).
- Design rationale: parsers pull large native wheels (PyTorch, pypdfium2, OpenCV). The repo isolates parser extras into per-parser targets (/opt/parsers/<parser>) or venvs to avoid collection-time import failures and native ABI conflicts.

2) Key files to read before editing runtime/CI behavior
- `docker-entrypoint.sh` — selects per-parser venv or pip --target dir; contains the runtime install fallback logic.
- `Dockerfile` — build stages; changes here affect baked parser venvs and image size.
- `.github/workflows/ci.yml` — CI build/test/publish workflow; baking heavy parsers is gated here.
- `pyproject.toml` — package extras and parser extras; authoritative package names/versions live here.

3) Common developer workflows & quick commands
- Run unit tests locally (fast path, avoid heavy parsers):
```bash
PDFKB_PDF_PARSER=pymupdf4llm PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=false pytest -q
```
- Run full test matrix (may require heavy native wheels and long downloads):
```bash
pytest
```
- Start the app locally (dev script):
```bash
./run-local.sh
```
- Build and run in Docker (example):
```bash
docker build -t pdfkb-mcp .
docker run --rm -e PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true -e PDFKB_PARSERS_TO_INSTALL=marker -v /tmp/pdfkb-parsers:/opt/parsers pdfkb-mcp
```

4) Project-specific conventions and patterns
- Parser isolation: the runtime prefers `/opt/parsers/<parser>/venv` (virtualenv) or `/opt/parsers/<parser>` (pip --target). Avoid changing this without considering collection-time imports in pytest.
- Opt-in runtime installs: `PDFKB_ALLOW_RUNTIME_PARSER_INSTALL` must be explicitly set to allow container startup to attempt installing heavy parser extras. CI generally avoids runtime installs.
- Wheel cache: `/opt/parsers/wheels` is used/created by runtime installer to reduce repeated downloads (persist via volume if possible).
- Environment variables are primary configuration surface (PDFKB_*). Check `docker-entrypoint.sh` and `src/pdfkb/config.py` for the full set.

5) Integration and dependencies to watch for
- Heavy native dependencies: PyTorch, triton, pypdfium2 (PDFium), OpenCV, SciPy — these are platform- and OS-dependent and can break imports at collection time.
- Embeddings providers: local vs `openai` vs `huggingface` — ensure API keys are provided for remote providers (env vars PDFKB_OPENAI_API_KEY, PDFKB_HUGGINGFACE_API_KEY / HF_TOKEN).
- External services: CI builds images and pushes to GHCR; local dev uses `docker-compose.dev.yml`.

6) Making changes that touch build/runtime/CI
- If you change parser package names or extras, update `pyproject.toml` and ensure CI (or Dockerfile) bakes those changes or document how to enable runtime install.
- Prefer small, local tests first: run the specific pytest file(s) that exercise the changed code (`pytest tests/test_*.py::test_name -q`).
- After editing `docker-entrypoint.sh`, run a quick syntax check:
```bash
bash -n docker-entrypoint.sh
```

7) What an AI agent should do before opening a PR
- Read `docker-entrypoint.sh`, `Dockerfile`, `pyproject.toml`, and the relevant tests under `tests/` for the area you intend to change.
- Run focused tests locally with an appropriate `PDFKB_PDF_PARSER` to avoid heavy downloads.
- If adding or modifying parser installs, update docs (README or `docs/`) and add/adjust CI steps to validate import smoke checks for that parser.

8) Where to add docs & examples
- Add small usage docs under `docs/` (e.g., `docs/parsers.md`) for parser baking vs runtime-installs, and update `README.md` with recommended env examples and volume mounts for `/opt/parsers`.

If anything in this file is unclear or you want me to include additional examples (CI job snippets, Dockerfile bake steps, or exact package spec lines from `pyproject.toml`), tell me which area to expand and I'll iterate.