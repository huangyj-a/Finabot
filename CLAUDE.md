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
3. **`finabot/agents/core.py:Agent.process`** — per-message coroutine. Uses a `MemorySaver` checkpointer (`Agent.checkpointer`) compiled into the graph; session state (including `messages`) is keyed by `thread_id = "{channel}:{chat_id}"` and accumulates across turns automatically. `_process_locked` seeds the checkpointer from disk (`load_short_memory`) only when it has no checkpoint yet (fresh thread or process restart), streams the graph with `astream(subgraphs=True)` to push hold-pipeline step progress live, reads the authoritative final state via `aget_state`, then persists a curated user/assistant log via `save_short_memory` and publishes the last AI message. Each message is its own asyncio task (`Agent.run` fans them out). There is no manual `Agent.sessions` dict or `SessionManager` anymore.
4. **`finabot/graph/graph.py:build_graph`** — LangGraph `StateGraph(AgentState)` with nodes: `supervisor` (= `call_llm_node` from `agents/nodes.py`), `market_analyst`, `fundamental_analyst`, `news_analyst`, `researchers`, `hold_analysis_pipeline`, `tool`. The supervisor is the entry point. Routing in `_internal_make_route_supervisor` inspects the **last AIMessage's `tool_calls`**:
   - empty → `END`
   - exactly one call to a known sub-agent (`market_analyst`, `fundamental_analyst`, `news_analyst`, `researchers`, `hold_analysis_pipeline`) → that sub-agent node (sub-agents consume the latest human message, not the tool-call args)
   - anything else → the generic `tool` node
   All non-end nodes loop back to `supervisor`. The full multi空 debate (news → bull+bear → summary) is **folded into `hold_analysis_pipeline`** as a single supervisor hop — `bull_researcher` / `bear_researcher` / `summary_manager` are NOT separate supervisor-routed graph nodes anymore; they live inside the pipeline (`agents/hold_pipeline.py`) and are invoked directly there. `hold_analysis_pipeline` accepts a `debate_mode` flag to surface the step-by-step 新闻/看涨/看跌/结论稿件.
5. **`finabot/agents/nodes.py`** — `call_llm_node` calls LiteLLM via `litellm_glm_call`, then **normalizes tool calls** through three layers:
   - `normalize_tool_call` handles three input shapes (OpenAI-style `{"function": {...}}`, LangChain-style `{"name", "args"}`, and raw object attributes).
   - `extract_tool_calls_from_content` is a regex fallback that parses serialized `<tool_call><arg_key>...</arg_key><arg_value>...</arg_value>` markup that GLM sometimes emits as plain text instead of a structured tool call. If this fallback fires, the assistant text is cleared.
   - `format_tools` builds the OpenAI tool schema by reading each tool's `args_schema.model_json_schema()`; tools without a schema get a default `{expression: string}` shape.
6. **`finabot/agents/llm.py`** — wraps `litellm.acompletion`. `convert_messages` translates LangChain `BaseMessage`s into LiteLLM dicts, including reserializing prior `tool_calls` (re-emitting `id`, `function.name`, JSON-stringified `arguments`) so multi-turn tool conversations replay correctly. The Chinese system prompt in this file is the supervisor's single source of truth — it lists every tool by name and dictates output formatting (notably the **mandatory six-section format** for "is this stock worth holding" questions, which references `docs/examples1.md` as the density target). **Token streaming**: `litellm_glm_call(messages, ..., stream_label=...)` — when a token sink is installed via the contextvar in `finabot/agents/streaming.py` and the label is in the streamable set (`supervisor` / `fundamental_analyst` / `news_analyst`), the completion uses `stream=True` and forwards each text delta to the sink (`_internal_acompletion_stream` also reconstructs streaming tool-call fragments). Other labels stay whole-response so parallel `bull`/`bear` tokens don't interleave. `Agent._process_locked` installs the sink; `cli/commands.py` renders `stream: token` messages inline (typewriter).
7. **`finabot/tools/base.py:get_tools`** returns `[calculator, read_file, fundamental_analyst, market_analyst, news_analyst, researchers, hold_analysis_pipeline, get_stock_news_unified, *get_akshare_tools()]`. The supervisor-facing sub-agents (`market_analyst`, `fundamental_analyst`, `news_analyst`, `researchers`, `hold_analysis_pipeline`) are registered both as graph nodes **and** as `@tool`s. When the router sends control to the corresponding node the node wraps state; when a sub-agent call instead falls through to the generic `tool` node (e.g. mixed multi-tool batches), `call_tool_node` dispatches it through `_internal_invoke_sub_agent` (`agents/nodes.py`) with the **same** state context (reports, `akshare_cache`) and writes report increments back to state — both routes must behave identically. `bull_researcher` / `bear_researcher` / `summary_manager` are NOT in this list; they are invoked directly inside `hold_analysis_pipeline`, not as supervisor tools. The calculator uses an AST whitelist (`_safe_eval`, including power-size limits), not `eval()`. Debate-context helpers used by the pipeline's internal debate live in `agents/debate.py`.
8. **`finabot/tools/akshare_tools.py`** — wraps AKShare for A-share / HK / fund / index queries. Symbol resolution (`stock_a_lookup`) maps Chinese names to codes; higher-level tools (`stock_a_snapshot`, `stock_a_hold_analysis`, `stock_a_conclusion`) compose lower-level tools and return a single JSON payload with a `tool` field, used by tests as a contract.
9. **会话持久化（checkpointer）** — `Agent` 用 `MemorySaver`（`Agent.checkpointer`）编译进图，状态按 `thread_id = session_key` 存续并在轮次间自动累积。TTL（默认 60 分钟）由 `Agent._thread_last_used` 追踪、`Agent._cleanup_expired` 清理（`runtime.py:run_maintenance` 调用）。`save_short_memory` 仍把精选的 user/assistant 对话存成磁盘 JSON，用于进程重启后重新填充 checkpointer，也供 `build_memory_context` 构造提示词记忆。旧 `SessionManager` 与 `Agent.sessions` 双存储已删除。**跨轮次滚动摘要（`agents/rolling_summary.py`）**：长对话中段新增 ≥ `WINDOW`（默认 8）条且总条数 ≥ `MIN_MESSAGES`（默认 12）时，用 LLM 生成/合并摘要并持久化到 working memory，下一轮经 `build_memory_context` 注入提示词记忆（"跨轮次历史摘要"），保证中段被压缩器裁剪后模型仍有连续性。开关与阈值全部由环境变量控制（`FINABOT_ROLLING_SUMMARY` 及 `MIN_MESSAGES`/`WINDOW`/`TAIL_KEEP`/`MAX_CHARS`/`MAX_MSG_CHARS`，见 `.env.example`），`core.py` 调用时惰性读取；`litellm_glm_call(..., system_prompt=...)` 支持覆盖默认 supervisor 提示，供摘要器等独立角色使用。
10. **诊断与告警（`runtime.py`）** — `RuntimeService` 周期性写 `heartbeat.json`（bus 积压、会话/任务/锁、LLM 指标、每任务错误计数），`PeriodicTask` 捕获回调异常不致命。`DiagnosticMonitor`（`diagnostic_interval_seconds>0` 时注册为 `diagnostic` 周期任务）对连续两次 `snapshot()` 做差分判读：心跳停摆、任务错误增量、bus 积压、LLM 失败率/耗时，命中写一行 JSON 到 `diagnostic.log`；`error` 级 issue 额外触发告警送达——`notifier` 回调（`DiagnosticMonitor(runtime, notifier=...)`，同步/异步均可）或 `FINABOT_ALERT_WEBHOOK_URL` webhook（POST JSON，5s 超时）。告警送达全部尽力而为、失败绝不阻断主流程。接入示例见 `docs/diagnostic_alert.md`。

`finabot/api/`, `finabot/channels/`, `finabot/config/`, `finabot/dataflows/`, `finabot/providers/`, `finabot/utils/` exist as empty / stub directories — the only live channel today is the CLI.

## Things that have bitten people

- **Sub-agent vs tool dual registration.** The supervisor-facing sub-agents (`market_analyst`, `fundamental_analyst`, `news_analyst`, `researchers`, `hold_analysis_pipeline`) exist as `@tool` definitions *and* as graph nodes. If you add another supervisor sub-agent: register it in both places, add a branch in `_internal_make_route_supervisor`, AND add it to `_SUB_AGENT_NAMES` / `_internal_invoke_sub_agent` in `agents/nodes.py`. `bull_researcher` / `bear_researcher` / `summary_manager` are intentionally NOT supervisor tools — they run only inside `hold_analysis_pipeline`, so don't re-expose them as graph nodes or `@tool`s. `build_graph(single_agent=True)` 构建单 Agent 对照组（仅 supervisor+tool，无子代理节点；supervisor 用 `SINGLE_AGENT_SYSTEM_PROMPT`，`format_tools(single_agent=True)` 剔除子代理工具）。
- **Tool-call shape drift.** GLM models occasionally return tool calls as plain text instead of structured JSON. Don't delete the regex fallback in `extract_tool_calls_from_content` without a replacement — the e2e flow depends on it.
- **Message conversion.** When adding a new message type or tool-call field, update `convert_messages` in `agents/llm.py` — LiteLLM rejects payloads that mix LangChain-native shapes with the OpenAI `tool_calls` format.
- **Tests stub network deps.** `tests/test_akshare_tools.py` patches `sys.modules["litellm"]` and `sys.modules["akshare"]` with fakes at top of file. Anything importing those modules at collection time must tolerate the fakes (e.g. proxy helpers in `akshare_tools.py` are wrapped in `try/except` for this reason).
- **Stock-hold output contract.** When changing the supervisor system prompt, preserve the six-section structure for stock-hold questions (结论前置 / 核心判断 / 看多逻辑 / 看空 / 持仓策略 / 最后总结). `docs/examples1.md` is the reference density.
- **`docs/architecture.md`** is a useful overview but predates the supervisor / sub-agent split — when in doubt, read `graph/graph.py` and `agents/llm.py` over the doc.
