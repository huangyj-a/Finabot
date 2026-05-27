from typing import List, Dict
from finabot.bus.events import InboundMessage


class ContextBuilder:
    def __init__(self, system_prompt: str = "You are a helpful AI assistant."):
        self.system_prompt = system_prompt

    def build(
        self,
        user_message: InboundMessage,
        history: List[Dict],
    ) -> List[Dict]:
        """
        构建发送给大模型的完整上下文：系统提示 + 历史对话 + 当前用户消息
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message.content})
        return messages