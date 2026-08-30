"""Research sub-agents."""

from finabot.agents.researchers.bear_researcher import _internal_call_bear_researcher
from finabot.agents.researchers.bull_researcher import _internal_call_bull_researcher
from finabot.agents.researchers.researchers import researchers

__all__ = [
    "researchers",
    "_internal_call_bear_researcher",
    "_internal_call_bull_researcher",
]
