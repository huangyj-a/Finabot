from typing import Callable, Dict, Any
import json


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: List[Dict] = []

    def register(self, name: str, func: Callable, schema: Dict):
        self.tools[name] = func
        self.schemas.append({
            "type": "function",
            "function": schema
        })

    def get_schemas(self) -> List[Dict]:
        return self.schemas

    async def execute(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"Error: Tool {name} not found"
        return await self.tools[name](**args)


# 示例工具：计算器
async def calculator(expression: str) -> str:
    try:
        # 生产环境请使用安全的表达式解析，这里仅作演示
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {str(e)}"


# 注册工具
def create_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    calculator_schema = {
        "name": "calculator",
        "description": "Calculate mathematical expressions, e.g. 3*5+2",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to calculate"
                }
            },
            "required": ["expression"]
        }
    }
    registry.register("calculator", calculator, calculator_schema)
    return registry