# Prompt files for this workspace

This folder contains workspace-scoped VS Code prompt files (`.prompt.md`) that can be run from the Chat view. The files follow the VS Code prompt-file format (see https://code.visualstudio.com/docs/copilot/customization/prompt-files).

Quick rules (workspace-specific)
- File extension: use `.prompt.md` (example: `agent_prompt.prompt.md`).
- Location: place workspace prompt files here: `.github/prompts/`.
- Header (YAML frontmatter) fields supported in this repo:
  - `description` (string) — short description of the prompt's purpose.
  - `mode` (ask | edit | agent) — chat mode used when running the prompt.
  - `model` (optional) — model to suggest. If not present, the currently selected model will be used.
  - `tools` (array) — names of tool sets allowed for this prompt (optional).

Body
- The body is free-form Markdown instructions. Use examples, expected input/output, and links to workspace files where helpful.
- You can reference workspace variables and inputs using VS Code variables (e.g. `${selection}`, `${fileBasename}`, `${input:varName}`).

Best practices for this repo
- Keep prompts concise and focused (single purpose). Prefer linking to `docs/` or `.github/copilot-instructions.md` for longer guidance.
- If a prompt requires repository-specific commands or environment variables (e.g. `PDFKB_*`), document those in the prompt body and link to `README.md` or `docs/parsers.md`.
- Avoid embedding secrets in prompt files.

Testing
- Open a `.prompt.md` file in VS Code and press the play button to run it in chat. Use the Chat view `Configure Chat > Prompt Files` to manage prompts.

Examples
- See `agent_prompt.prompt.md` for an example agent-mode prompt tailored to local assistant behaviour.
