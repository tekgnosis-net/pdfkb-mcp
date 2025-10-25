# PDF Parser Guide

This guide is being expanded. For now:
- Defaults: parser = `pymupdf4llm`; embeddings provider = `local` (Qwen/Qwen3-Embedding-0.6B); chunker = `langchain`; hybrid search = enabled; reranker = disabled; summarizer = disabled.
- See the [Quick Start](quick-start.md), [Troubleshooting](troubleshooting.md), and [Docker Deployment](docker-deployment.md) guides for working configurations.

For immediate help, see:
- [Quick Start Guide](quick-start.md) for default parser setup
- [Troubleshooting Guide](troubleshooting.md) for parser issues

This guide will include:
- Parser comparison and selection
- PyMuPDF4LLM, Marker, Docling, MinerU, LLM parser configurations
- Performance benchmarks
- Use case recommendations

Build-time vs runtime parser selection
------------------------------------

There are two distinct knobs you can use to control which PDF parser is
available and used by the container:

- Build-time: `--build-arg PDF_PARSER=<parser>` — controls what the Docker
	builder bakes into the image. Use this to pre-install heavy parser
	dependencies (large wheels like PyTorch, PDFium/OpenCV, etc.) into per-parser
	virtualenvs under `/opt/parsers`. Typical values: `docling`, `marker`,
	`mineru`, `pymupdf4llm`, or `all`.
- Runtime: `PDFKB_PDF_PARSER=<parser>` (container environment variable) —
	controls which parser the entrypoint selects when the container starts. The
	entrypoint will prefer a baked virtualenv in `/opt/parsers/<parser>/venv` or
	a pip `--target` layout in `/opt/parsers/<parser>`. If neither exists, and
	`PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true`, the container may attempt a
	best-effort runtime install (not recommended for production).

Key recommendations
-------------------

- For fast, reliable PR builds: build only the parser you need. Example (CI):

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
	--build-arg PDF_PARSER=docling \
	-t pdfkb-mcp:pr-docling .
```

- For production or release images where you want multiple parsers baked,
	build with `PDF_PARSER=all`. This is heavier and may require extra disk and
	BuildKit cache to avoid transient failures during build.

- To avoid surprises where you build for one parser but the container still
	defaults to another, the image sets the runtime `PDFKB_PDF_PARSER` from the
	build arg when present. So a build with `--build-arg PDF_PARSER=docling` will
	default the container runtime to `docling` unless overridden at `docker run`.

OpenCV / cv2 and system libraries
---------------------------------

Some parsers (e.g., Marker, Docling) require OpenCV (`cv2`) and system GL
libraries at runtime. The image ensures the necessary system packages
(`libgl1-mesa-glx`, `libglib2.0-0`, etc.) are installed in the runtime stage
and `opencv-python-headless` is installed into the image so `import cv2` will
work out of the box for baked parser venvs. If you rely on runtime installs,
mount `/opt/parsers` and `/opt/parsers/wheels` to persist installs between
container runs.

CI recommendations
------------------

- Use BuildKit and persist pip/cache between runs (the repository workflows
	already use buildx cache) so heavy wheel downloads aren't repeated and
	transient disk pressure is reduced.
- For PRs: build a single parser image (fast). For nightly or release: run a
	separate `PDF_PARSER=all` job that bakes every parser into the image.

Examples
--------

Build docling-only image:

```bash
DOCKER_BUILDKIT=1 docker build --build-arg PDF_PARSER=docling -t pdfkb-mcp:docling .
```

Run the container and override runtime parser if needed:

```bash
docker run -e PDFKB_PDF_PARSER=marker -p 8000:8000 pdfkb-mcp:docling
```

This will prefer a baked `marker` venv if present; otherwise the entrypoint
will attempt a runtime install if allowed.

**Coming soon**: full parser comparison, configuration snippets, and
performance notes.

