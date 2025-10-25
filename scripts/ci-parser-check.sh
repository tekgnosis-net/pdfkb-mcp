#!/usr/bin/env bash
set -euo pipefail

# CI helper: verify baked parsers inside a running container named pdfkb-pr-smoke
CONTAINER=${1:-pdfkb-pr-smoke}

echo "Checking container: ${CONTAINER}"

if ! docker ps --filter "name=${CONTAINER}" --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "Container ${CONTAINER} is not running"
  docker ps -a || true
  exit 1
fi

# Read requested runtime default parser from environment in the container
REQUESTED=$(docker exec ${CONTAINER} sh -c 'printf "%s" "${PDFKB_PDF_PARSER:-}"') || true
if [ -z "${REQUESTED}" ]; then
  echo "Requested (image default) parser: <none>"
else
  echo "Requested (image default) parser: ${REQUESTED}"
fi

# List baked parser dirs under /opt/parsers
BAKED=$(docker exec ${CONTAINER} sh -c 'ls -1 /opt/parsers 2>/dev/null || true') || true
if [ -z "${BAKED}" ]; then
  echo "Baked parser dirs: <none>"
else
  echo "Baked parser dirs:\n${BAKED}"
fi

# Helper: attempt import for a baked parser
attempt_import() {
  local p="$1"
  echo "-- Testing import for baked parser: $p"
  if docker exec ${CONTAINER} sh -c "test -x /opt/parsers/$p/venv/bin/python" >/dev/null 2>&1; then
    docker exec ${CONTAINER} sh -c "/opt/parsers/$p/venv/bin/python - <<'PY'
import sys,traceback
try:
    __import__('${p}')
    print('OK: imported ${p} via venv')
except Exception:
    traceback.print_exc()
    sys.exit(2)
PY"
  else
    docker exec ${CONTAINER} sh -c "python3 - <<'PY'
import sys,traceback
sys.path.insert(0, '/opt/parsers/${p}')
try:
    __import__('${p}')
    print('OK: imported ${p} via PYTHONPATH')
except Exception:
    traceback.print_exc()
    sys.exit(2)
PY"
  fi
}

# If a requested parser is set, ensure it's present and importable
if [ -n "${REQUESTED}" ]; then
  if docker exec ${CONTAINER} sh -c "test -d /opt/parsers/${REQUESTED}" >/dev/null 2>&1; then
    echo "Requested parser '${REQUESTED}' is present under /opt/parsers"
    attempt_import "${REQUESTED}"
  else
    echo "ERROR: requested parser '${REQUESTED}' not found under /opt/parsers" >&2
    docker logs ${CONTAINER} || true
    exit 1
  fi
else
  echo "No requested parser set in image; skipping mandatory check"
fi

# For any other baked parsers, attempt import but don't fail the job if they fail
for p in ${BAKED}; do
  [ -z "$p" ] && continue
  # skip the requested one (already tested)
  if [ "${p}" = "${REQUESTED}" ]; then
    continue
  fi
  echo "Attempting import for optional baked parser: $p (non-fatal)"
  if ! attempt_import "$p"; then
    echo "Warning: import failed for optional parser $p (continuing)" || true
  fi
done

echo "Parser verification completed"
