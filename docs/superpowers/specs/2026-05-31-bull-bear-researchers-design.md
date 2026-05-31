---
name: bull-bear-researchers
description: Design for bull and bear researcher agents integrated into finabot's LangGraph supervisor pattern
date: 2026-05-31
---

# Bull and Bear Researchers Design

## Overview

This design adds bull (看涨) and bear (看跌) researcher agents to finabot as separate graph nodes, enabling investment debate analysis. The researchers provide opposing perspectives on stocks and can reference each other's arguments within a session.

## Architecture

### Graph Integration

The bull and bear researchers integrate as separate nodes in the existing LangGraph supervisor pattern:

```
START → supervisor → [market_analyst | researchers | bull_researcher | bear_researcher | tool] → supervisor → END
```

**Key principles:**
- Bull and bear are independent graph nodes (not a combined debate node)
- Supervisor routes to them via tool calls (same pattern as market_analyst)
- Each researcher returns to supervisor after execution
- Supervisor controls debate flow and decides when to call each researcher

### State Management

**AgentState (unchanged):**
- `messages`: Sequence[BaseMessage]
- `session_key`: str

**Session State (extended):**
Session state managed by SessionManager will store an optional `debate_context` dict:

```python
debate_context = {
    "bull_arguments": [],      # List of bull researcher responses
    "bear_arguments": [],      # List of bear researcher responses
    "last_speaker": None,      # "bull" | "bear" | None
    "in_progress": None        # "bull" | "bear" | None (debate lock)
}
```

**Rationale for session state vs graph state:**
- Debate context is session-scoped, not message-scoped
- Naturally expires with session TTL (60 min default)
- Keeps AgentState simple and focused on message flow
- Avoids polluting graph state with debate-specific data

### RunnableConfig Integration

Session context is passed to tools via LangChain's `RunnableConfig` mechanism:

**In Agent.process():**
```python
config = {
    "configurable": {
        "session_key": session_key,
        "session_manager": self.session_manager
    }
}
result = await self.graph.ainvoke(state, config=config)
```

**In researcher tools:**
```python
@tool
async def bull_researcher(expression: str, config: RunnableConfig) -> str:
    session_manager = config["configurable"]["session_manager"]
    session_key = config["configurable"]["session_key"]
    # Access session state...
```

LangGraph automatically injects `config` parameter - no signature changes needed in tool registration.

## Component Design

### New Files

**finabot/agents/researchers/bull_researcher.py**
- System prompt defining bull analyst role (看涨分析师)
- `_internal_build_prompt()` - Creates ChatPromptTemplate
- `_internal_call_bull_researcher(expression, session_manager, session_key)` - Core logic
- `@tool async def bull_researcher(expression: str, config: RunnableConfig) -> str` - Tool wrapper

**finabot/agents/researchers/bear_researcher.py**
- System prompt defining bear analyst role (看跌分析师)
- `_internal_build_prompt()` - Creates ChatPromptTemplate
- `_internal_call_bear_researcher(expression, session_manager, session_key)` - Core logic
- `@tool async def bear_researcher(expression: str, config: RunnableConfig) -> str` - Tool wrapper

### Modified Files

**finabot/agents/researchers/__init__.py**
- Export `bull_researcher` and `bear_researcher` tools

**finabot/tools/base.py**
- Add `bull_researcher` and `bear_researcher` to `get_tools()` return list

**finabot/graph/graph.py**
- Add `_internal_bull_researcher_node` wrapper function
- Add `_internal_bear_researcher_node` wrapper function
- Add nodes: `g.add_node("bull_researcher", _internal_bull_researcher_node)`
- Add routing branches in `_internal_route_supervisor`:
  - `if tool_name == "bull_researcher": return "bull_researcher"`
  - `if tool_name == "bear_researcher": return "bear_researcher"`
- Add edges: `g.add_edge("bull_researcher", "supervisor")`

**finabot/agents/llm.py**
- Update supervisor system prompt to include bull_researcher and bear_researcher in tool list
- Add sequential execution constraint instruction (see Constraints section)

**finabot/agents/core.py**
- Pass RunnableConfig with session_manager and session_key when invoking graph

**finabot/agents/session.py**
- Extend SessionManager to store arbitrary session state (not just messages)
- Add methods: `get_state(session_key) -> dict`, `update_state(session_key, state: dict)`
- Store state alongside messages: `self.session_states: Dict[str, dict] = {}`

## Data Flow

### Initialization (Agent.process)

1. Agent receives InboundMessage
2. Fetches or creates session state from SessionManager
3. Appends HumanMessage to state["messages"]
4. Invokes graph with RunnableConfig:
   ```python
   config = {
       "configurable": {
           "session_key": session_key,
           "session_manager": self.session_manager
       }
   }
   result = await self.graph.ainvoke(state, config=config)
   ```

### Supervisor Routing

1. Supervisor (call_llm_node) receives user query
2. Analyzes query and decides to invoke bull_researcher or bear_researcher
3. Returns AIMessage with tool_call (e.g., `{"name": "bull_researcher", "args": {"expression": "..."}}`)
4. Router function `_internal_route_supervisor` detects tool_name and routes to corresponding node

### Tool Execution (Bull/Bear Researcher)

1. LangGraph's tool node executes the researcher tool
2. RunnableConfig is automatically injected by LangGraph
3. Researcher extracts context:
   ```python
   session_manager = config["configurable"]["session_manager"]
   session_key = config["configurable"]["session_key"]
   session_state = session_manager.get_state(session_key)
   ```
4. Initialize or retrieve debate_context:
   ```python
   debate_context = session_state.get("debate_context")
   if not debate_context:
       debate_context = {
           "bull_arguments": [],
           "bear_arguments": [],
           "last_speaker": None,
           "in_progress": None
       }
   ```
5. Check debate lock (see Constraints section):
   ```python
   if debate_context["in_progress"] and debate_context["in_progress"] != "bull":
       return "请等待空头研究员完成分析"
   debate_context["in_progress"] = "bull"
   ```
6. Build prompt with optional opponent context:
   ```python
   opponent_last = None
   if debate_context["last_speaker"] == "bear" and debate_context["bear_arguments"]:
       opponent_last = debate_context["bear_arguments"][-1]
   
   prompt = build_prompt(expression, opponent_last)
   ```
7. Call LLM via litellm_glm_call
8. Update debate_context:
   ```python
   debate_context["bull_arguments"].append(response.content)
   debate_context["last_speaker"] = "bull"
   debate_context["in_progress"] = None
   session_state["debate_context"] = debate_context
   session_manager.update_state(session_key, session_state)
   ```
9. Return response string as ToolMessage

### Return to Supervisor

1. Tool node returns ToolMessage to supervisor
2. Supervisor processes tool result and conversation history
3. Decides next action:
   - Call opposing researcher for counterargument
   - Call another tool for data
   - Generate final synthesis response
4. Final AIMessage published to outbound queue

### Session Cleanup

- debate_context persists in session state
- Expires with session TTL (default 60 min)
- SessionManager.cleanup_expired() removes old debate contexts automatically

## Prompt Design

### Bull Researcher Prompt (看涨分析师)

```python
_BULL_RESEARCHER_PROMPT = """
你是一位看涨分析师，负责为股票投资建立强有力的论证。

你的任务是构建基于证据的强有力案例，强调增长潜力、竞争优势和积极的市场指标。

请用中文回答，重点关注以下几个方面：

- 增长潜力：突出公司的市场机会、收入预测和可扩展性
- 竞争优势：强调独特产品、强势品牌或主导市场地位等因素
- 积极指标：使用财务健康状况、行业趋势和最新积极消息作为证据
- 反驳看跌观点：用具体数据和合理推理批判性分析看跌论点，全面解决担忧并说明为什么看涨观点更有说服力
- 参与讨论：以对话风格呈现你的论点，直接回应看跌分析师的观点并进行有效辩论

{opponent_context}

请确保所有回答都使用中文。
""".strip()
```

**Opponent context (conditional):**
```python
opponent_context = ""
if opponent_last_argument:
    opponent_context = f"""
对方分析师的最新观点：
{opponent_last_argument}

请针对以上观点进行回应和反驳。
"""
```

### Bear Researcher Prompt (看跌分析师)

```python
_BEAR_RESEARCHER_PROMPT = """
你是一位看跌分析师，负责论证投资风险和挑战。

你的目标是提出合理的论证，强调风险、挑战和负面指标。

请用中文回答，重点关注以下几个方面：

- 风险和挑战：突出市场饱和、财务不稳定或宏观经济威胁等可能阻碍股票表现的因素
- 竞争劣势：强调市场地位较弱、创新下降或来自竞争对手威胁等脆弱性
- 负面指标：使用财务数据、市场趋势或最近不利消息的证据来支持你的立场
- 反驳看涨观点：用具体数据和合理推理批判性分析看涨论点，揭露弱点或过度乐观的假设
- 参与讨论：以对话风格呈现你的论点，直接回应看涨分析师的观点并进行有效辩论

{opponent_context}

请确保所有回答都使用中文。
""".strip()
```

## Constraints and Error Handling

### Sequential Debate Enforcement

**Critical constraint:** Bull and bear researchers must execute sequentially, never in parallel.

**Problem:** If supervisor generates multiple tool_calls in one AIMessage (e.g., `[bull_researcher, bear_researcher]`), LangGraph executes them in parallel. Both read `last_speaker=None` and lose the debate dynamic.

**Solution (two-layer defense):**

**Layer 1: Supervisor prompt constraint (primary)**

Add to supervisor system prompt in `finabot/agents/llm.py`:

```python
重要规则：
- 对于投资辩论分析，你必须一次只调用一个研究员（bull_researcher 或 bear_researcher）
- 等待该研究员的回复后，再决定是否调用另一个研究员
- 绝不要在同一轮中同时调用多个研究员
- 如果需要多角度分析，请按顺序调用：先调用一个研究员，等待回复，再调用另一个
```

**Layer 2: Tool execution guard (fallback)**

In bull/bear researcher tool functions:

```python
debate_context = session_state.get("debate_context", {})
in_progress = debate_context.get("in_progress")

# Check if opponent is currently running
if in_progress and in_progress != "bull":  # For bull_researcher
    return "请等待空头研究员完成分析后再调用看涨研究员"

# Set lock
debate_context["in_progress"] = "bull"
session_state["debate_context"] = debate_context
session_manager.update_state(session_key, session_state)

try:
    # ... execute researcher logic ...
finally:
    # Release lock
    debate_context["in_progress"] = None
    session_state["debate_context"] = debate_context
    session_manager.update_state(session_key, session_state)
```

### Other Error Handling

**Missing session_manager in config:**
```python
if "configurable" not in config or "session_manager" not in config["configurable"]:
    return "系统错误：无法访问会话管理器"
```

**Session expired during debate:**
- If session_state is None, debate_context is None
- Researcher starts fresh with no opponent context
- No error - graceful degradation

**LLM call failure:**
- Standard litellm error handling
- Return error message to supervisor: "分析失败：{error_message}"
- Supervisor can retry or provide fallback response

**Invalid debate_context structure:**
```python
if not isinstance(debate_context, dict) or "bull_arguments" not in debate_context:
    # Reinitialize with clean structure
    debate_context = {
        "bull_arguments": [],
        "bear_arguments": [],
        "last_speaker": None,
        "in_progress": None
    }
```

## Testing Strategy

### Unit Tests

**Test debate_context initialization:**
- First call creates debate_context with empty arguments
- Subsequent calls reuse existing debate_context

**Test sequential execution:**
- Bull called first: last_speaker=None, no opponent context
- Bear called second: last_speaker="bull", receives bull's last argument
- Bull called third: last_speaker="bear", receives bear's last argument

**Test debate lock:**
- Parallel execution attempt returns error message
- Lock is released after execution completes
- Lock is released even if LLM call fails (finally block)

**Test session expiry:**
- Expired session results in fresh debate_context
- No errors when debate_context is missing

### Integration Tests

**Test full debate flow:**
1. User asks: "帮我深度对比一下寒武纪的利弊"
2. Supervisor calls bull_researcher
3. Bull provides bullish analysis
4. Supervisor calls bear_researcher
5. Bear references bull's points and provides bearish analysis
6. Supervisor synthesizes final response

**Test supervisor routing:**
- Verify supervisor generates single tool_call for researchers
- Verify routing logic correctly identifies bull/bear tool names
- Verify edges return to supervisor after execution

**Test RunnableConfig propagation:**
- Verify session_manager and session_key are accessible in tools
- Verify config is passed through graph execution

## Future Enhancements

### Phase 2: Data Integration

Integrate AKShare tools to provide researchers with market data:
- Call stock_a_snapshot before invoking researchers
- Pass market data in prompt context
- Add sentiment_report, news_report, fundamentals_report

### Phase 3: Memory Integration

Add memory system for past debate outcomes:
- Store successful/failed investment theses
- Reference past mistakes in current debates
- Learn from historical patterns

### Phase 4: Multi-Round Debate

Extend supervisor to orchestrate multi-round debates:
- Configurable debate rounds (e.g., 3 rounds)
- Debate termination conditions (convergence, time limit)
- Final synthesis node that summarizes debate

### Phase 5: Debate Visualization

Add debate history visualization:
- Format debate_context as structured output
- Display bull/bear arguments side-by-side
- Highlight key points of contention

## Implementation Checklist

- [ ] Extend `finabot/agents/session.py` SessionManager with state storage
- [ ] Create `finabot/agents/researchers/bull_researcher.py`
- [ ] Create `finabot/agents/researchers/bear_researcher.py`
- [ ] Update `finabot/agents/researchers/__init__.py` exports
- [ ] Add researchers to `finabot/tools/base.py:get_tools()`
- [ ] Add node wrappers in `finabot/graph/graph.py`
- [ ] Add routing branches in `_internal_route_supervisor`
- [ ] Add graph edges for bull/bear nodes
- [ ] Update supervisor prompt in `finabot/agents/llm.py`
- [ ] Add RunnableConfig passing in `finabot/agents/core.py`
- [ ] Write unit tests for debate_context logic
- [ ] Write integration tests for full debate flow
- [ ] Update documentation in `docs/architecture.md`

## Success Criteria

**Functional:**
- Bull and bear researchers can be invoked by supervisor
- Researchers reference opponent's last argument when available
- Debate context persists within session
- Sequential execution is enforced (no parallel debate)

**Quality:**
- Researchers provide substantive analysis in Chinese
- Debate feels conversational and responsive
- No crashes or state corruption
- Clean error messages for edge cases

**Performance:**
- No significant latency increase vs single researcher call
- Session state updates are efficient
- Debate context cleanup works correctly
