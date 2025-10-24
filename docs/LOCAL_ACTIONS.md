# Running GitHub Actions locally (pre-commit tests)

This document explains how to run GitHub Actions workflows locally so you can validate CI before pushing to `main`.

Tools
- `act` (https://github.com/nektos/act): a local runner for GitHub Actions.

Limitations & notes
- `act` attempts to emulate GitHub runners but doesn't perfectly replicate the hosted environment (especially for large, GPU, or CUDA artifacts). Heavy binary installs (PyTorch, CUDA toolkits, big wheels) may behave differently.
- Some workflows that rely on GitHub-hosted services or secrets need to be adapted locally (provide env vars or local services like Docker/Redis).

Quick setup (Linux)
1. Install Docker (required by `act`) and ensure your user can run docker commands.
2. Install `act` (example using latest release):

```bash
# using package manager (if available) or brew
sudo curl -sSfL https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
# or download binary release from https://github.com/nektos/act/releases
```

3. Run a workflow locally

```bash
# from repo root
# Run the tests workflow (the name of the job in .github/workflows/run-tests.yml)
# Use a larger image to reduce mismatch for Linux runners if available
act -j tests -P ubuntu-latest=nektos/act-environments-ubuntu:22.04
```

Environment & secrets
- Provide secrets that the workflow expects using `-s NAME=value` or a `.secrets` file:

```bash
# single secret
act -j tests -s DOCKERHUB_USERNAME=you -s DOCKERHUB_TOKEN=...

# or use a file
act --secret-file .secrets -j tests
```

Tips to reduce friction
- Replace heavy install steps with cached wheels or local indexes where possible.
- For GPU/CUDA heavy deps (torch), prefer building a minimal CI path in which integration tests requiring GPUs are gated behind a separate workflow.
- Use `--reuse` and `--cache` flags in `act` to speed repeated runs.

Advanced: run the workflow inside a container matching GitHub's runner
- `act` allows mapping `ubuntu-latest` to a local container image. Use images that already have some heavy deps preinstalled to avoid long installs.
- Example: build a custom image with preinstalled torch/transformers and reference it with `-P ubuntu-latest=your/image:tag`.

When `act` is insufficient
- For the heaviest tests (full Docker builds, GPU tests) run them in a dedicated CI branch in GitHub but minimize failures by pre-validating locally (install step, unit tests).
- You can run the same docker build & smoke scripts locally (see `scripts/build_and_test_image.sh`) to validate runtime behavior (marker parsing and context-shift) before pushing.


