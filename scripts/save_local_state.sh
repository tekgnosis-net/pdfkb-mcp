#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   echo '{...}' | ./scripts/save_local_state.sh chat
#   echo '{...}' | ./scripts/save_local_state.sh todos
#
# This helper writes stdin to the corresponding file under .local_state with a
# timestamp header. It will not commit or push; it's a local convenience.

STATE_DIR="$(dirname "$0")/../.local_state"
CHAT_FILE="${STATE_DIR}/chat_history.md"
TODOS_FILE="${STATE_DIR}/todos.md"

mkdir -p "${STATE_DIR}"

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <chat|todos>" >&2
  exit 2
fi

kind="$1"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

case "$kind" in
  chat)
    out="$CHAT_FILE"
    header="# Chat snapshot (${ts})\n\n"
    ;;
  todos)
    out="$TODOS_FILE"
    header="# Todos snapshot (${ts})\n\n"
    ;;
  *)
    echo "Unknown kind: $kind" >&2
    exit 2
    ;;
esac

tmpfile=$(mktemp)
cat - > "$tmpfile"

# Prepend header and append to the file (keeping previous history)
{
  printf "%s" "$header"
  cat "$tmpfile"
  printf "\n\n"
} >> "$out"

rm -f "$tmpfile"
echo "Saved $kind snapshot to $out"
