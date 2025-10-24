#!/usr/bin/env bash
# monitor_ci.sh - Local GitHub Actions runner monitor using `gh` CLI
# Usage: scripts/monitor_ci.sh [repo] [branch] [timeout_seconds]
# Defaults: repo from git remote (owner/repo), branch=main, timeout=900 (15m)
set -euo pipefail

REPO=${1:-}
BRANCH=${2:-main}
TIMEOUT=${3:-900}

if [ -z "$REPO" ]; then
  # try to infer from git remote
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    origin_url=$(git config --get remote.origin.url || true)
    if [ -n "$origin_url" ]; then
      # Extract owner/repo from common remote URL forms and strip optional .git
      if [[ "$origin_url" =~ github.com[:/](.+) ]]; then
        REPO="${BASH_REMATCH[1]}"
        REPO="${REPO%.git}"
      fi
    fi
  fi
fi

if [ -z "$REPO" ]; then
  echo "Repository must be specified as owner/repo or inferable from git remote." >&2
  echo "Usage: $0 [owner/repo] [branch] [timeout_seconds]" >&2
  exit 2
fi

for cmd in gh jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required. Please install it before running this script." >&2
    exit 2
  fi
done

echo "Monitoring GitHub Actions runs for repo: $REPO branch: $BRANCH (timeout: ${TIMEOUT}s)"

END=$((SECONDS + TIMEOUT))
LAST_RUN=""

while true; do
  # Request fields supported by modern gh versions (databaseId and number)
  run_json=$(gh run list --repo "$REPO" --branch "$BRANCH" --limit 1 --json databaseId,number,status,conclusion,url,workflowName,createdAt 2>/dev/null || true)

  if [ -z "$run_json" ] || [ "$run_json" = "null" ]; then
    echo "No workflow runs found on branch '$BRANCH' yet. Waiting..."
    if [ $SECONDS -ge $END ]; then
      echo "Timeout waiting for any run" >&2
      exit 2
    fi
    sleep 8
    continue
  fi

  # Extract the first run object
  run=$(echo "$run_json" | jq '.[0] // empty') || run=""
  if [ -z "$run" ] || [ "$run" = "null" ]; then
    echo "No workflow run object yet. Waiting..."
    sleep 5
    continue
  fi

  dbid=$(echo "$run" | jq -r '.databaseId // empty')
  num=$(echo "$run" | jq -r '.number // empty')
  status=$(echo "$run" | jq -r '.status // empty')
  conclusion=$(echo "$run" | jq -r '.conclusion // empty')
  url=$(echo "$run" | jq -r '.url // empty')
  wf=$(echo "$run" | jq -r '.workflowName // empty')
  created=$(echo "$run" | jq -r '.createdAt // empty')

  # Present a helpful summary line
  echo "Latest run: id=${dbid:-$num} workflow=$wf status=$status conclusion=$conclusion url=$url created=$created"

  # Choose an identifier for gh run view: prefer databaseId, fallback to number
  run_id="$dbid"
  if [ -z "$run_id" ]; then
    run_id="$num"
  fi

  if [ -z "$run_id" ]; then
    echo "Run has no usable id yet; waiting..."
  else
    if [ "$status" != "in_progress" ] && [ "$status" != "queued" ]; then
      echo "Run finished: status=$status conclusion=$conclusion"
      echo "Fetching logs for run $run_id (may be large)..."
      gh run view "$run_id" --repo "$REPO" --log || echo "Failed to fetch logs for run $run_id"
      exit 0
    fi

    if [ "$run_id" != "$LAST_RUN" ]; then
      echo "Observing run $run_id (workflow: $wf). Waiting for completion..."
      LAST_RUN="$run_id"
    fi
  fi

  if [ $SECONDS -ge $END ]; then
    echo "Timeout waiting for run to complete (after ${TIMEOUT}s) — last observed status: $status" >&2
    echo "Run URL: $url" >&2
    exit 2
  fi

  sleep 10
done
