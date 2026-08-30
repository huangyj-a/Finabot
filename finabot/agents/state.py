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
    # 评估/可追溯性字段（评估报告要求）：
    as_of: str | None
    # source_id -> {source, published_at, retrieved_at, url, priority, scope}
    evidence_registry: dict[str, dict[str, Any]]
    claims: list[dict[str, Any]]
    risk_flags: list[str]
    run_meta: dict[str, Any]
