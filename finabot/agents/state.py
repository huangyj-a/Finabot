from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(dict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_key: str