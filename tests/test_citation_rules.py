"""Prompt-level tests for unified citation requirements."""

from finabot.agents.analysts.news_analyst import _NEWS_ANALYST_PROMPT
from finabot.agents.llm import SYSTEM_PROMPT
from finabot.agents.managers.manager import _SUMMARY_MANAGER_PROMPT
from finabot.agents.researchers.bear_researcher import _BEAR_RESEARCHER_PROMPT
from finabot.agents.researchers.bull_researcher import _BULL_RESEARCHER_PROMPT


REQUIRED_CITATION_PHRASES = [
    "东方财富",
    "通达信",
    "Wind",
    "巨潮资讯网",
    "深交所互动易",
    "Omdia",
    "中国通信院",
    "来源/日期缺失",
]


def test_core_agent_prompts_share_unified_citation_rules():
    prompts = [
        _NEWS_ANALYST_PROMPT,
        _SUMMARY_MANAGER_PROMPT,
        _BULL_RESEARCHER_PROMPT,
        _BEAR_RESEARCHER_PROMPT,
        SYSTEM_PROMPT,
    ]

    for prompt in prompts:
        for phrase in REQUIRED_CITATION_PHRASES:
            assert phrase in prompt

