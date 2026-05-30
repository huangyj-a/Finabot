import ast
import operator

from langchain_core.tools import tool

from finabot.agents.analysts import market_analyst
from finabot.agents.researchers import researchers
from finabot.tools.akshare_tools import get_akshare_tools


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


def get_tools():
    return [calculator, market_analyst, researchers, *get_akshare_tools()]