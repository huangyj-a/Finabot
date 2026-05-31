# Repository Guidelines

## Project Structure & Module Organization

Finabot is a Python package for a Chinese-market financial assistant. Core source lives in `finabot/`, with the CLI entry point in `finabot/cli/commands.py` and package execution via `finabot/__main__.py`. Agent orchestration is under `finabot/agents/` and `finabot/graph/`; tool implementations, including AKShare wrappers, are in `finabot/tools/`. Tests live in `tests/`, and design notes or examples are in `docs/`. The `skills/` directory is excluded from packaging by `pyproject.toml`.

## Build, Test, and Development Commands

- `pip install -e .` installs the package in editable mode and registers the `finabot` console script.
- `finabot start` runs the interactive CLI.
- `finabot start --message "贵州茅台现在适合持有吗" --session cli:demo` sends a one-shot message through the CLI.
- `python -m finabot start ...` runs the same CLI through the module entry point.
- `finabot version` prints the installed package version.
- `pytest tests/` runs the test suite; use `pytest tests/test_akshare_tools.py::test_name` for a focused test.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation and clear, descriptive names. Follow existing module patterns: lowercase module names, `snake_case` functions and variables, and `PascalCase` classes. Keep tool outputs structured and JSON-friendly where existing tests assert contracts. Prefer small, focused functions over broad rewrites, and preserve Chinese prompts/documentation where they are part of runtime behavior.

## Testing Guidelines

Tests use `pytest`. Add tests under `tests/` with names matching `test_*.py` and functions named `test_*`. Existing AKShare tests stub external modules in `sys.modules`, so tests should remain offline and deterministic. When changing tool contracts, update or add assertions for returned JSON fields rather than relying on live network data.

## Commit & Pull Request Guidelines

Recent history uses short conventional-style prefixes such as `docs:` and `add:`. Keep commit messages concise and imperative, for example `docs: clarify session API` or `add: support fund lookup`. Pull requests should include a brief summary, testing performed, linked issue if applicable, and screenshots or CLI transcripts when user-facing output changes.

## Security & Configuration Tips

Configuration is loaded from `.env` via `python-dotenv`. Keep API keys out of commits; use provider-specific variables such as `ZAI_API_KEY` or fallback `ZHIPU_API_KEY`. Avoid adding tests or examples that call live financial APIs by default.

## Agent-Specific Notes

Sub-agents such as `market_analyst` and `researchers` are both graph nodes and tools. If adding another sub-agent, register it in both places and update routing in `finabot/graph/graph.py`. Preserve tool-call normalization fallbacks in `finabot/agents/nodes.py` unless replacing them with equivalent coverage.
