# Agent instruction prompt — local assistant behaviour

When acting as an automated assistant/agent for this repository, follow these rules.

## 1) Review history
- Always read and consider the contents of `.local_state/chat_history.md` and `.local_state/todos.md` (if present) at the start of a session or before making edits. This preserves prior decisions and avoids regressions.

## 2) Save state
- Periodically save a succinct snapshot of the conversation and the current todo list to `.local_state/chat_history.md` and `.local_state/todos.md` respectively. Include timestamps and short summaries of decisions.

## 3) No secrets
- Never write API keys, tokens, or other secrets into `.local_state` files. If a secret is encountered, redact it and instruct the user to store it securely (environment variables or a secrets manager).

## 4) Minimal writes
- Keep `.local_state` lightweight — short summaries and structured todos are preferred over full raw logs.

## 5) Consistency
- When proposing changes to entrypoint or build scripts, explain trade-offs and add a short migration note to `.local_state` describing the changed behavior.

## 6) Manual opt-in for risky ops
- Runtime package installs or destructive operations (like runtime uninstalls) must be explicitly opt-in and clearly logged in `.local_state` when performed.

## 7) Periodic save cadence
- Save after any substantial change (edit/create files, run CI-affecting commands, or modify the todo list). Prefer human confirmation before saving very large logs.

This file is intended to be read by the local assistant and repository contributors; keep it up-to-date when agent behaviour changes.
