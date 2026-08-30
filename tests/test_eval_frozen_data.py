"""Tests for the frozen-data layer."""

import json

import pandas as pd

from finabot.eval.frozen_data import FrozenData, install_frozen_akshare
from finabot.eval.tasks import load_task_by_id


def _load_frozen_t001():
    task = load_task_by_id("t001")
    assert task is not None
    return FrozenData(task)


def test_frozen_data_available_for_t001():
    frozen = _load_frozen_t001()
    assert frozen.available is True
    assert "stock_zh_a_hist" in frozen.snapshot


def test_get_akshare_frame_returns_dataframe():
    frozen = _load_frozen_t001()
    frame = frozen.get_akshare_frame("stock_zh_a_hist")
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) > 0
    # 列名兼容 eastmoney(收盘/日期) 与腾讯(close/date) 两种来源
    assert ("收盘" in frame.columns) or ("close" in frame.columns)


def test_get_akshare_frame_missing_returns_empty():
    frozen = _load_frozen_t001()
    frame = frozen.get_akshare_frame("stock_hk_index_daily_em")
    assert frame.empty


def test_install_frozen_akshare_intercepts_fetchers(monkeypatch):
    frozen = _load_frozen_t001()
    install_frozen_akshare(frozen, monkeypatch)

    import finabot.tools.akshare_tools as aktools

    frame = aktools.ak.stock_zh_a_hist(symbol="600519")
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) > 0


def test_frozen_data_serves_news_payload():
    task = load_task_by_id("t001")
    frozen = FrozenData(task)
    # t001 fixture 无新闻快照，应返回 None
    assert frozen.get_news_payload() is None