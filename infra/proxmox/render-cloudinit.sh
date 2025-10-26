#!/usr/bin/env bash
# Render a cloud-init template with placeholders like {{VAR}} using an optional vars file.
# Usage: render-cloudinit.sh TEMPLATE_PATH [VARS_FILE]
# VARS_FILE format: KEY=VALUE per line (shell-compatible). Values are not quoted or may be quoted.

set -euo pipefail
TEMPLATE_PATH=${1:-}
VARS_FILE=${2:-}

if [ -z "$TEMPLATE_PATH" ]; then
  echo "Usage: $0 TEMPLATE_PATH [VARS_FILE]" >&2
  exit 1
fi

if [ ! -f "$TEMPLATE_PATH" ]; then
  echo "Template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

# Load vars into environment for rendering
if [ -n "$VARS_FILE" ]; then
  if [ ! -f "$VARS_FILE" ]; then
    echo "Vars file not found: $VARS_FILE" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  set -a
  # shellcheck disable=SC1090
  . "$VARS_FILE"
  set +a
fi

# Use a short Python script to replace {{VAR}} with environment values safely
python3 - "$TEMPLATE_PATH" <<'PY'
import sys,os,re
from pathlib import Path
tpl_path = Path(sys.argv[1])
text = tpl_path.read_text()
# pattern to find {{VAR}}
pattern = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")

def repl(m):
    key = m.group(1)
    return os.environ.get(key, m.group(0))

out = pattern.sub(repl, text)
print(out, end='')
PY

exit 0
