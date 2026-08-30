"""Tests for the clock abstraction (FINABOT_EVAL_AS_OF freeze)."""

import os
from datetime import datetime

from finabot.utils.clock import now


def test_clock_now_uses_wall_clock_by_default(monkeypatch):
    monkeypatch.delenv("FINABOT_EVAL_AS_OF", raising=False)
    before = datetime.now()
    result = now()
    after = datetime.now()
    assert before <= result <= after


def test_clock_now_respects_eval_as_of(monkeypatch):
    monkeypatch.setenv("FINABOT_EVAL_AS_OF", "2026-05-29T15:00:00+08:00")
    result = now()
    assert result.year == 2026
    assert result.month == 5
    assert result.day == 29


def test_clock_now_ignores_unparseable_as_of(monkeypatch):
    monkeypatch.setenv("FINABOT_EVAL_AS_OF", "not-a-date")
    before = datetime.now()
    result = now()
    after = datetime.now()
    assert before <= result <= after