"""Finabot evaluation harness package (评估实操报告落地).

Modules:
- tasks: eval task schema + JSON loading
- graders: deterministic graders + one-vote-veto hard gates
- metrics: Pass@1 / Pass-all-N / severe-failure-rate etc.
- frozen_data: offline frozen-data fixture factory
- harness: trial runner that produces trace + report
- llm_judge: isolated LLM judges for news/bear/synthesis dimensions
"""

__all__ = [
    "tasks",
    "graders",
    "metrics",
    "frozen_data",
    "harness",
    "llm_judge",
    "ablation",
    "sources",
]