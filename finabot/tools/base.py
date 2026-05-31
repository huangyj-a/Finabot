import ast
import os
import operator
from pathlib import Path

from langchain_core.tools import tool

from finabot.agents.analysts import market_analyst
from finabot.agents.analysts.news_analyst import news_analyst
from finabot.agents.hold_pipeline import run_hold_analysis_pipeline
from finabot.agents.managers.manager import summary_manager
from finabot.agents.researchers import bear_researcher, bull_researcher, researchers
from finabot.tools.akshare_tools import get_akshare_tools
from finabot.tools.news_tools import get_stock_news_unified


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _internal_skills_root() -> Path:
    configured = Path(os.getenv("FINABOT_SKILLS_DIR", "skills"))
    if not configured.is_absolute():
        configured = _REPO_ROOT / configured
    return configured.resolve()


def _internal_allowed_read_roots() -> list[Path]:
    context_cache = Path(os.getenv("FINABOT_CONTEXT_CACHE_DIR", ".finabot_context/tool_results"))
    if not context_cache.is_absolute():
        context_cache = _REPO_ROOT / context_cache
    return [_internal_skills_root(), context_cache.resolve()]


def _safe_eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        operator_fn = _ALLOWED_BINOPS.get(type(node.op))
        if operator_fn is None:
            raise ValueError("unsupported operator")
        return operator_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        operator_fn = _ALLOWED_UNARYOPS.get(type(node.op))
        if operator_fn is None:
            raise ValueError("unsupported operator")
        return operator_fn(_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """计算器工具，计算数学表达式"""
    try:
        parsed = ast.parse(expression, mode="eval")
        return str(_safe_eval(parsed))
    except Exception:
        return "计算错误"


@tool
def read_file(path: str) -> str:
    """按需读取技能或压缩上下文落盘文件内容。"""

    try:
        requested = Path(path)
        if not requested.is_absolute():
            requested = _REPO_ROOT / requested
        resolved = requested.resolve()
        allowed_roots = _internal_allowed_read_roots()
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            return "读取失败：只能读取 skills 或 .finabot_context 目录下的文件。"
        if not resolved.is_file():
            return "读取失败：文件不存在。"
        return resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return f"读取失败：{exc}"


@tool
async def hold_analysis_pipeline(expression: str) -> str:
    """单股持有分析流水线，共享AKShare缓存并一次性完成新闻、多空和总结分析。"""

    result = await run_hold_analysis_pipeline(expression, {"akshare_cache": {}})
    return result["summary_report"]


def get_tools():
    return [
        calculator,
        read_file,
        market_analyst,
        news_analyst,
        researchers,
        bull_researcher,
        bear_researcher,
        summary_manager,
        hold_analysis_pipeline,
        get_stock_news_unified,
        *get_akshare_tools(),
    ]
