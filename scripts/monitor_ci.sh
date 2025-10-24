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
      # support git@github.com:owner/repo.git and https://github.com/owner/repo.git
      REPO=$(echo "$origin_url" | sed -E 's#(git@|https?://)([^/:]+)[:/]([^/]+)/(.+)(\.git)?#\3/\4#')
    fi
  fi
fi

if [ -z "$REPO" ]; then
  echo "Repository must be specified as owner/repo or inferable from git remote." >&2
  echo "Usage: $0 [owner/repo] [branch] [timeout_seconds]" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required. Install from https://cli.github.com/" >&2
  exit 2
fi

echo "Monitoring GitHub Actions runs for repo: $REPO branch: $BRANCH (timeout: ${TIMEOUT}s)"

END=$((SECONDS + TIMEOUT))
LAST_ID=""

while true; do
  # Get the latest run for the branch
  run_json=$(gh run list --repo "$REPO" --branch "$BRANCH" --limit 1 --json id,status,conclusion,htmlUrl,workflowName,createdAt --jq '.[0]') || true
  if [ -z "$run_json" ] || [ "$run_json" = "null" ]; then
    echo "No workflow runs found on branch '$BRANCH' yet. Waiting..."
    if [ $SECONDS -ge $END ]; then
      echo "Timeout waiting for any run" >&2
      exit 2
    fi
    sleep 8
    continue
  fi

  # Extract fields safely
  id=$(echo "$run_json" | sed -n 's/.*"id": *\([0-9]*\).*/\1/p' || echo "")
  status=$(echo "$run_json" | sed -n 's/.*"status": *"\([^"]*\)".*/\1/p' || echo "")
  conclusion=$(echo "$run_json" | sed -n 's/.*"conclusion": *\("[^"]*"\|null\).*/\1/p' || echo "")
  url=$(echo "$run_json" | sed -n 's/.*"htmlUrl": *"\([^"]*\)".*/\1/p' || echo "")
  wf=$(echo "$run_json" | sed -n 's/.*"workflowName": *"\([^"]*\)".*/\1/p' || echo "")
  created=$(echo "$run_json" | sed -n 's/.*"createdAt": *"\([^"]*\)".*/\1/p' || echo "")

  printf "Latest run: id=%s workflow=%s status=%s conclusion=%s url=%s created=%s\n" "$id" "$wf" "$status" "$conclusion" "$url" "$created"

  if [ "$status" != "in_progress" ] && [ "$status" != "queued" ]; then
    echo "Run finished: status=$status conclusion=$conclusion"
    echo "Fetching logs for run $id (may be large)..."
    gh run view "$id" --repo "$REPO" --log || true
    exit 0
  fi

  if [ "$id" != "$LAST_ID" ]; then
    echo "Observing run $id (workflow: $wf). Waiting for completion..."
    LAST_ID=$id
  fi

  if [ $SECONDS -ge $END ]; then
    echo "Timeout waiting for run to complete (after ${TIMEOUT}s) — last observed status: $status" >&2
    echo "Run URL: $url" >&2
    exit 2
  fi

  sleep 10
done
