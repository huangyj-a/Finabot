# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Finabot is a Chinese-market financial assistant. A LangGraph supervisor routes user messages to sub-agents (`market_analyst`, `researchers`) or to AKShare-backed market-data tools, then synthesizes a final reply. Transport between channels and the agent core is an in-process async `MessageBus`. The codebase, prompts, and most documentation are in Chinese.

## Common commands

The project uses a `.venv/` checked into the repo root. Activate it before running anything (`.venv\Scripts\activate` on Windows bash, or call binaries via `.venv/Scripts/...`).

- Run interactively: `finabot start` (registered as a console script via `pyproject.toml`).
- One-shot message: `finabot start --message "贵州茅台现在适合持有吗" --session cli:demo`.
- Run as module: `python -m finabot start ...` (equivalent entry point).
- Show version: `finabot version`.
- Install in editable mode: `pip install -e .` (deps beyond `typer` / `python-dotenv` — `litellm`, `langgraph`, `langchain-core`, `akshare`, `pandas`, `pytest` — are already present in `.venv` but not declared in `pyproject.toml`).
- Run tests: `pytest tests/`. Single test: `pytest tests/test_akshare_tools.py::test_stock_a_history_returns_structured_json`. Tests install fake `litellm` and `akshare` modules into `sys.modules` at import time, so they run offline and never hit the network.

## Configuration

`.env` (loaded via `python-dotenv` in `cli/commands.py:start`) supplies LLM credentials:

- `LLM_PROVIDER` — legacy names (`zhipu`, `zhipuai`, `glm`) are normalized to `zai` by `finabot/agents/llm.py:resolve_provider`. Default: `zai`.
- `LLM_MODEL` — default `glm-4`.
- `<PROVIDER>_API_KEY` (e.g. `ZAI_API_KEY`) or fallback `ZHIPU_API_KEY`.

The full LiteLLM model string sent to `acompletion` is `{provider}/{model}`.

## Architecture (read these to understand the big picture)

The lifecycle of a single user message crosses several modules — reading any one in isolation is misleading.

1. **`finabot/cli/commands.py`** spawns an `Agent` and a `MessageBus`, then enters either `run_once` (for `--message`) or `run_interactive`. The agent's `run()` task is started concurrently and cancelled on exit.
2. **`finabot/bus/queue.py` + `bus/events.py`** — two `asyncio.Queue`s (inbound/outbound). `InboundMessage.session_key` defaults to `"{channel}:{chat_id}"`, which is how sessions are scoped.
3. **`finabot/agents/core.py:Agent.process`** — per-message coroutine. Fetches/creates per-session state (a dict with `messages` and `session_key`), appends the new `HumanMessage`, runs `self.graph.ainvoke(state)`, persists the final state, and publishes the last AI message to the outbound queue. Each message is its own asyncio task (`Agent.run` fans them out).
4. **`finabot/graph/graph.py:build_graph`** — LangGraph `StateGraph(AgentState)` with four nodes: `supervisor` (= `call_llm_node` from `agents/nodes.py`), `market_analyst`, `researchers`, `tool`. The supervisor is the entry point. Routing in `_internal_route_supervisor` inspects the **last AIMessage's `tool_calls`**:
   - empty → `END`
   - exactly one call to `market_analyst` or `researchers` → that sub-agent node (the sub-agent is wrapped so it consumes the latest human message, not the tool-call args)
   - anything else → the generic `tool` node
   All non-end nodes loop back to `supervisor`.
5. **`finabot/agents/nodes.py`** — `call_llm_node` calls LiteLLM via `litellm_glm_call`, then **normalizes tool calls** through three layers:
   - `normalize_tool_call` handles three input shapes (OpenAI-style `{"function": {...}}`, LangChain-style `{"name", "args"}`, and raw object attributes).
   - `extract_tool_calls_from_content` is a regex fallback that parses serialized `<tool_call><arg_key>...</arg_key><arg_value>...</arg_value>` markup that GLM sometimes emits as plain text instead of a structured tool call. If this fallback fires, the assistant text is cleared.
   - `format_tools` builds the OpenAI tool schema by reading each tool's `args_schema.model_json_schema()`; tools without a schema get a default `{expression: string}` shape.
6. **`finabot/agents/llm.py`** — wraps `litellm.acompletion`. `convert_messages` translates LangChain `BaseMessage`s into LiteLLM dicts, including reserializing prior `tool_calls` (re-emitting `id`, `function.name`, JSON-stringified `arguments`) so multi-turn tool conversations replay correctly. The Chinese system prompt in this file is the supervisor's single source of truth — it lists every tool by name and dictates output formatting (notably the **mandatory six-section format** for "is this stock worth holding" questions, which references `docs/examples1.md` as the density target).
7. **`finabot/tools/base.py:get_tools`** returns `[calculator, market_analyst, researchers, *get_akshare_tools()]`. The sub-agents (`market_analyst`, `researchers`) are registered both as graph nodes **and** as `@tool`s — the supervisor invokes them by emitting a tool call with their name, and the router sends control to the corresponding node. The calculator uses an AST whitelist (`_safe_eval`), not `eval()`.
8. **`finabot/tools/akshare_tools.py`** — wraps AKShare for A-share / HK / fund / index queries. Symbol resolution (`stock_a_lookup`) maps Chinese names to codes; higher-level tools (`stock_a_snapshot`, `stock_a_hold_analysis`, `stock_a_conclusion`) compose lower-level tools and return a single JSON payload with a `tool` field, used by tests as a contract.
9. **`finabot/agents/session.py:SessionManager`** — TTL-based (default 60 min) per-session history. `Agent.process` calls `cleanup_expired` on every message and discards the in-memory state for any expired key. `Agent.sessions` (the LangGraph state) and `SessionManager.sessions` (the chat log) are separate stores kept in parallel.

`finabot/api/`, `finabot/channels/`, `finabot/config/`, `finabot/dataflows/`, `finabot/providers/`, `finabot/utils/` exist as empty / stub directories — the only live channel today is the CLI.

## Things that have bitten people

- **Sub-agent vs tool dual registration.** `market_analyst` and `researchers` exist as `@tool` definitions *and* as graph nodes. If you add another sub-agent, register it in both places and add a branch in `_internal_route_supervisor`, or the supervisor will route it to the generic `tool` node and lose the conversational handoff.
- **Tool-call shape drift.** GLM models occasionally return tool calls as plain text instead of structured JSON. Don't delete the regex fallback in `extract_tool_calls_from_content` without a replacement — the e2e flow depends on it.
- **Message conversion.** When adding a new message type or tool-call field, update `convert_messages` in `agents/llm.py` — LiteLLM rejects payloads that mix LangChain-native shapes with the OpenAI `tool_calls` format.
- **Tests stub network deps.** `tests/test_akshare_tools.py` patches `sys.modules["litellm"]` and `sys.modules["akshare"]` with fakes at top of file. Anything importing those modules at collection time must tolerate the fakes (e.g. proxy helpers in `akshare_tools.py` are wrapped in `try/except` for this reason).
- **Stock-hold output contract.** When changing the supervisor system prompt, preserve the six-section structure for stock-hold questions (结论前置 / 核心判断 / 看多逻辑 / 看空 / 持仓策略 / 最后总结). `docs/examples1.md` is the reference density.
- **`docs/architecture.md`** is a useful overview but predates the supervisor / sub-agent split — when in doubt, read `graph/graph.py` and `agents/llm.py` over the doc.
