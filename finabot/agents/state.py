from typing import Annotated, Any, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(dict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_key: str
    user_id: str
    memories: list[dict[str, Any]]
    market_report: str
    news_report: str
    bull_report: str
    bear_report: str
    fundamentals_report: str
    akshare_cache: dict[str, Any]
    debate_context: dict[str, Any]
    # 规则预路由（finabot.graph.router）写入：命中"持有+辩论"意图时为 True，
    # 供 hold_analysis_pipeline 节点读取（LLM 路由路径则来自 tool_call 参数）。
    debate_mode: bool
    # 评估/可追溯性字段（评估报告要求）：
    as_of: str | None
    # source_id -> {source, published_at, retrieved_at, url, priority, scope}
    evidence_registry: dict[str, dict[str, Any]]
    claims: list[dict[str, Any]]
    risk_flags: list[str]
    run_meta: dict[str, Any]
