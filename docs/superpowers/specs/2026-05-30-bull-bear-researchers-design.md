# Bull/Bear Researchers Design

**Date:** 2026-05-30  
**Status:** Draft  
**Author:** Claude Code

## Overview

Add bull_researcher and bear_researcher agents to Finabot's multi-agent system. Each agent is implemented as a LangGraph subgraph with independent tool-calling capability (AKShare data + news). Supervisor routes investment judgment questions to both agents, receives opposing viewpoints, and synthesizes a final recommendation.

## Goals

1. Enable investment debate: bull constructs bullish arguments, bear constructs bearish arguments
2. Data-driven analysis: both agents call AKShare tools and news APIs to support their positions
3. Preserve existing agents: keep researchers (neutral research), market_analyst (market dynamics)
4. Single-round debate: bull and bear each analyze once independently, supervisor integrates

## Non-Goals

- Multi-round debate (bull → bear → bull counter → bear counter): deferred for complexity
- Bull/bear seeing each other's arguments: single-round means they work independently
- Replacing existing researchers: neutral research remains valuable for concept questions

## Architecture

### Graph Topology

```
User Message
  ↓
supervisor (existing)
  ↓ (conditional routing)
  ├─→ market_analyst (existing)
  ├─→ researchers (existing, neutral research)
  ├─→ bull_researcher (new subgraph)
  │    └─ internal loop: llm → tool? → tool_node → llm → ...
  ├─→ bear_researcher (new subgraph)
  │    └─ internal loop: llm → tool? → tool_node → llm → ...
  └─→ tool (existing, for supervisor's own use)
  ↓
All nodes return to supervisor
  ↓
END (supervisor outputs final answer)
```

### Responsibility Split

**Supervisor:**
- Routes concept/background questions → `researchers`
- Routes market analysis questions → `market_analyst`
- Routes investment judgment questions → `bull_researcher` + `bear_researcher` (may call both)
- Routes data queries → its own `tool` node
- Synthesizes bull/bear outputs into final recommendation

**bull_researcher subgraph:**
- Input: user question (e.g., "Is Moutai worth holding now?")
- Internal: calls AKShare tools + news_fetch, builds bullish case
- Output: bullish analysis text with data citations

**bear_researcher subgraph:**
- Input: same user question
- Internal: calls AKShare tools + news_fetch, builds bearish case
- Output: bearish analysis text with data citations

**researchers (existing):**
- Preserved for neutral research (e.g., "What is P/E ratio?")

### Key Design Decisions

1. **Bull/bear do not see each other's output:** single-round mode, independent analysis, supervisor integrates
2. **Subgraph isolation:** bull/bear tool-call history does not pollute supervisor's messages
3. **Tool sharing:** bull/bear subgraphs use the same tool list as supervisor (AKShare + news + calculator)

## Components

### 1. Bull Researcher Subgraph

**File:** `finabot/agents/researchers/bull_researcher.py`

**Structure:**
```python
def build_bull_researcher_subgraph():
    subgraph = StateGraph(AgentState)
    
    subgraph.add_node("llm", bull_llm_node)
    subgraph.add_node("tool", call_tool_node)  # reuse from agents/nodes.py
    
    subgraph.add_edge(START, "llm")
    subgraph.add_conditional_edges(
        "llm",
        should_continue_bull,
        {"tool": "tool", "end": END}
    )
    subgraph.add_edge("tool", "llm")
    
    return subgraph.compile(recursion_limit=5)
```

**bull_llm_node:**
- System prompt: bullish analyst role, emphasizes growth potential, competitive advantages, positive indicators
- Instructs LLM to call tools for real-time data rather than relying on memory
- Lists available tools: stock_a_lookup, stock_a_snapshot, stock_a_spot, stock_a_history, news_fetch, etc.
- Workflow guidance:
  1. If user provides stock name, call stock_a_lookup to get code
  2. Call stock_a_snapshot for latest data
  3. Call news_fetch for recent news
  4. Build bullish case with specific data citations

**should_continue_bull:**
- Checks if last message has `tool_calls`
- Returns `"tool"` if yes, `"end"` if no

**Tool execution:**
- Reuses `call_tool_node` from `agents/nodes.py`
- Reuses `format_tools()` for tool schema generation

### 2. Bear Researcher Subgraph

**File:** `finabot/agents/researchers/bear_researcher.py`

**Structure:** Symmetric to bull, only system prompt differs

**bear_llm_node:**
- System prompt: bearish analyst role, emphasizes risks, challenges, negative indicators
- Same tool list and workflow as bull
- Focus on downside risks and vulnerabilities

### 3. News Tool

**File:** `finabot/tools/news_tools.py`

```python
@tool
def news_fetch(keyword: str, days: int = 7) -> str:
    """Fetch recent news for a keyword (stock name or code)
    
    Args:
        keyword: search keyword
        days: fetch news from last N days, default 7
    
    Returns:
        JSON string with news list
    """
    try:
        df = ak.stock_news_em(symbol=keyword)
        
        if df.empty:
            return json.dumps({
                "tool": "news_fetch",
                "keyword": keyword,
                "count": 0,
                "news": []
            }, ensure_ascii=False)
        
        # Filter last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        if "发布时间" in df.columns:
            df["发布时间"] = pd.to_datetime(df["发布时间"])
            df = df[df["发布时间"] >= cutoff_date]
        
        # Take top 10
        df = df.head(10)
        
        news_list = []
        for _, row in df.iterrows():
            news_list.append({
                "标题": row.get("标题", ""),
                "内容": row.get("内容", "")[:200],  # truncate to 200 chars
                "发布时间": str(row.get("发布时间", "")),
                "来源": row.get("来源", "")
            })
        
        return json.dumps({
            "tool": "news_fetch",
            "keyword": keyword,
            "days": days,
            "count": len(news_list),
            "news": news_list
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "tool": "news_fetch",
            "keyword": keyword,
            "error": str(e)
        }, ensure_ascii=False)
```

**Design choices:**
- News source: AKShare's `stock_news_em` (East Money Finance)
- Content truncation: first 200 chars to avoid prompt bloat
- Time filter: default last 7 days, adjustable via parameter
- Error handling: returns JSON error instead of raising exception

### 4. Tool Registration

**File:** `finabot/tools/base.py`

```python
from finabot.tools.news_tools import news_fetch
from finabot.agents.researchers.bull_researcher import bull_researcher
from finabot.agents.researchers.bear_researcher import bear_researcher

def get_tools():
    return [
        calculator, 
        market_analyst, 
        researchers,
        bull_researcher,  # new: supervisor can call as tool
        bear_researcher,  # new: supervisor can call as tool
        news_fetch,       # new: news tool
        *get_akshare_tools()
    ]
```

**Note:** bull_researcher and bear_researcher need to be wrapped as `@tool` so supervisor can invoke them via tool_calls.

### 5. Supervisor Updates

**File:** `finabot/agents/llm.py`

**SYSTEM_PROMPT changes:**
- Add bull_researcher and bear_researcher to sub-agent list
- Add news_fetch to tool list
- Add routing principle: investment judgment questions → call both bull and bear, then synthesize
- Add synthesis instruction: when receiving both bull and bear analyses, final answer must:
  1. Summarize bullish arguments (2-3 points)
  2. Summarize bearish arguments (2-3 points)
  3. Provide comprehensive judgment and recommendation

### 6. Graph Routing Updates

**File:** `finabot/graph/graph.py`

**_internal_route_supervisor changes:**
```python
def _internal_route_supervisor(state: AgentState):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return "end"
    
    if len(tool_calls) == 1:
        tool_name = str(tool_calls[0].get("name") if isinstance(tool_calls[0], dict) 
                       else getattr(tool_calls[0], "name", ""))
        if tool_name == "market_analyst":
            return "market_analyst"
        if tool_name == "researchers":
            return "researchers"
        if tool_name == "bull_researcher":
            return "bull_researcher"
        if tool_name == "bear_researcher":
            return "bear_researcher"
    
    return "tool"
```

**build_graph changes:**
```python
def build_graph():
    g = StateGraph(AgentState)
    
    # Existing nodes
    g.add_node("supervisor", call_supervisor_node)
    g.add_node("market_analyst", _internal_market_analyst_node)
    g.add_node("researchers", _internal_researchers_node)
    g.add_node("tool", call_tool_node)
    
    # New nodes: bull/bear subgraphs
    g.add_node("bull_researcher", build_bull_researcher_subgraph())
    g.add_node("bear_researcher", build_bear_researcher_subgraph())
    
    g.add_edge(START, "supervisor")
    
    g.add_conditional_edges(
        "supervisor",
        _internal_route_supervisor,
        {
            "market_analyst": "market_analyst",
            "researchers": "researchers",
            "bull_researcher": "bull_researcher",
            "bear_researcher": "bear_researcher",
            "tool": "tool",
            "end": END,
        }
    )
    
    # All nodes return to supervisor
    g.add_edge("market_analyst", "supervisor")
    g.add_edge("researchers", "supervisor")
    g.add_edge("bull_researcher", "supervisor")
    g.add_edge("bear_researcher", "supervisor")
    g.add_edge("tool", "supervisor")
    
    return g.compile()
```

### 7. Wrapper Nodes for Graph Integration

**File:** `finabot/graph/graph.py`

```python
async def _internal_bull_researcher_node(state: AgentState):
    """Wrapper that extracts latest user message and passes to bull subgraph"""
    expression = _internal_latest_user_message(state)
    # Invoke bull subgraph with fresh state containing only the user question
    subgraph_state = {"messages": [HumanMessage(content=expression)]}
    result = await build_bull_researcher_subgraph().ainvoke(subgraph_state)
    # Extract final text from subgraph result
    final_message = result["messages"][-1]
    return {"messages": [AIMessage(content=str(final_message.content))]}

async def _internal_bear_researcher_node(state: AgentState):
    """Wrapper that extracts latest user message and passes to bear subgraph"""
    expression = _internal_latest_user_message(state)
    subgraph_state = {"messages": [HumanMessage(content=expression)]}
    result = await build_bear_researcher_subgraph().ainvoke(subgraph_state)
    final_message = result["messages"][-1]
    return {"messages": [AIMessage(content=str(final_message.content))]}
```

**Note:** These wrappers isolate subgraph state from supervisor state. Subgraph's internal tool-call history does not leak into supervisor's messages.

## Error Handling

### Tool Call Failures in Subgraph

- If news_fetch returns error, LLM should recognize and continue analysis based on other data
- If stock_a_snapshot fails, LLM should fall back to stock_a_spot + stock_a_individual_info
- System prompt instructs LLM to handle missing data gracefully

### Subgraph Recursion Limit

- Set `recursion_limit=5` when compiling subgraphs
- Prevents infinite tool-call loops
- If limit reached, subgraph returns whatever analysis it has so far

### Supervisor Missing Bull/Bear Output

- Supervisor system prompt includes fallback: "If bull_researcher or bear_researcher did not return results, provide conservative recommendation based on available information"

### Current researchers.py File Broken

- `finabot/agents/researchers/researchers.py` currently has syntax errors (undefined variables in f-string)
- Must be fixed before adding bull/bear: either remove the broken `_BEAR_RESEARCH_PROMPT` or fix variable references
- Recommended: delete lines 10-39 (the broken bear prompt), keep only the working researchers implementation

## Testing Strategy

### Unit Tests

**File:** `tests/test_bull_bear_researchers.py`

1. `test_bull_researcher_calls_tools`: mock litellm to return tool_calls, verify subgraph completes llm → tool → llm loop
2. `test_bear_researcher_calls_tools`: symmetric test for bear
3. `test_news_fetch_returns_json`: mock `ak.stock_news_em`, verify JSON structure (tool/keyword/count/news fields)
4. `test_supervisor_routes_to_bull_and_bear`: mock LLM to return tool_calls for bull/bear, verify routing logic

### Integration Tests

**File:** `tests/test_investment_debate_e2e.py`

1. End-to-end test: input "贵州茅台现在适合持有吗"
   - Expected flow:
     - supervisor → bull_researcher subgraph
       - bull llm (calls stock_a_lookup)
       - tool (returns 600519)
       - bull llm (calls stock_a_snapshot + news_fetch)
       - tool (returns data)
       - bull llm (generates bullish analysis)
     - supervisor → bear_researcher subgraph
       - (similar flow, generates bearish analysis)
     - supervisor → synthesizes output
   - Verify final output contains "看涨观点", "看跌观点", "综合判断"

### Manual Test Checklist

1. Concept question: "什么是市盈率" → should route to researchers
2. Market analysis: "A股大盘走势如何" → should route to market_analyst
3. Investment judgment: "贵州茅台适合持有吗" → should route to bull + bear
4. Data query: "查询贵州茅台最新价格" → should route to tool (stock_a_spot)
5. News in subgraph: bull/bear should be able to call news_fetch internally

## Implementation Order

1. Fix broken `researchers.py` (remove or fix lines 10-39)
2. Implement `news_fetch` tool in `tools/news_tools.py`
3. Implement `bull_researcher.py` subgraph
4. Implement `bear_researcher.py` subgraph
5. Update `tools/base.py` to register new tools
6. Update supervisor system prompt in `agents/llm.py`
7. Update graph routing in `graph/graph.py`
8. Write unit tests
9. Write integration tests
10. Manual testing with real questions

## Open Questions

None. All design decisions finalized with user approval.

## References

- `docs/reserchers.md`: original bull/bear implementation with multi-round debate and investment_debate_state
- `docs/architecture.md`: Finabot framework overview
- `docs/AKShare_dataapi.md`: AKShare data API reference
- `finabot/agents/llm.py`: supervisor system prompt
- `finabot/graph/graph.py`: current graph topology
- `finabot/agents/nodes.py`: tool execution and formatting logic
