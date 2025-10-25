# Parser installation & baking (runtime vs build-time)

This page documents the repository's approach for handling optional PDF parsers and gives concrete commands to:
- bake parser runtimes at Docker build-time (recommended for CI/production)
- or install them at container start (opt-in, useful for quick local testing)

Key concepts
- Per-parser isolation: each optional parser should be isolated under /opt/parsers to avoid dependency conflicts. There are two supported layouts:
  - baked virtualenv: /opt/parsers/<parser>/venv (preferred)
  - pip --target layout: /opt/parsers/<parser> (fallback)
- Wheels cache: runtime installer uses /opt/parsers/wheels to cache downloaded wheels and reduce repeated downloads.
- Opt-in installs: runtime installs only run when PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true to avoid surprise downloads in CI/production.

Runtime installer (opt-in)
- Env vars used by the entrypoint installer:
  - PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true  # enable runtime install
  - PDFKB_PARSERS_TO_INSTALL=marker,docling   # comma-separated list
  - PDFKB_INSTALL_ALL_PARSERS=true            # install recommended set (risky — heavy downloads)
  - PDFKB_PARSER_INSTALL_CONCURRENCY=2       # number of simultaneous installs
  - PDFKB_PARSER_INSTALL_TIMEOUT_SEC=1200    # per-parser timeout in seconds

Example (local quick test — persist installs to avoid repeated downloads):
```bash
mkdir -p /tmp/pdfkb-parsers
docker run --rm \
  -e PDFKB_ALLOW_RUNTIME_PARSER_INSTALL=true \
  -e PDFKB_PARSERS_TO_INSTALL=marker,docling \
  -v /tmp/pdfkb-parsers:/opt/parsers \
  -v /tmp/pdfkb-data:/app/documents \
  ghcr.io/tekgnosis-net/pdfkb-mcp:latest
```

Progress & health visibility
- The entrypoint writes per-parser status into `/opt/parsers/<parser>/.install_status` while installing.
- It also aggregates a summary file at `/app/parser_install_status` inside the container. If you mount `/app` or inspect the running container you can read this file to see install progress.

Build-time baking (recommended for CI/production)
- Building parser venvs at image-build time avoids large runtime downloads and test-time import failures. The repository's `Dockerfile` supports baking per-parser venvs in the builder stage and copying them to `/opt/parsers` in the runtime image.
- Build-time knobs (examples):
  - PDF_PARSER build-arg: controls which extras to install (e.g. marker, mineru, all). CI uses `PDF_PARSER=all` to produce a single combined image.

Example local image build (bakes parser venvs):
```bash
docker build --tag pdfkb-mcp:local --build-arg PDF_PARSER=all .
docker run --rm -it pdfkb-mcp:local sh -c "ls -la /opt/parsers && /opt/parsers/marker/venv/bin/python -c 'import marker; print(marker.__version__)'"
```

CI notes
- The GitHub Actions workflow runs a parser import smoke check against the built image. It prefers the per-parser venv interpreter (`/opt/parsers/<parser>/venv/bin/python`) and falls back to adding `/opt/parsers/<parser>` to `sys.path`.
- Because baking heavy parsers increases image size and build time, consider gating `PDF_PARSER=all` builds on a manual or scheduled job in CI.

Practical recommendations
- For local development: prefer `PDFKB_PDF_PARSER=pymupdf4llm` and leave runtime installs disabled to keep tests fast.
- For reproducible CI and production: bake parser venvs at build time and publish the baked image.
- If you must install at runtime: always mount `/opt/parsers` to persist installs and reduce repeated downloads.

If you want, I can add a short health endpoint that surfaces `/app/parser_install_status` for orchestrators (Kubernetes readiness), or add a sample CI job that builds the `PDF_PARSER=all` image only on a scheduled cron. Tell me which and I'll implement it.

*** End Patch