"""Tests for the frozen fixture snapshot builder."""

import json
import math

import pandas as pd

from finabot.eval.fixture_builder import assemble_snapshot, records_from_frame, write_snapshot


def test_records_from_frame_json_safe():
    df = pd.DataFrame([{"代码": "600519", "最新价": 1792.4, "异常": float("nan")}])
    records = records_from_frame(df)
    assert len(records) == 1
    assert records[0]["代码"] == "600519"
    # NaN 序列化为 None（有效 JSON），可 json.dumps 往返
    assert records[0]["异常"] is None
    json.dumps(records)


def test_records_from_frame_empty():
    assert records_from_frame(None) == []
    assert records_from_frame(pd.DataFrame()) == []


def test_records_from_frame_limits_rows():
    df = pd.DataFrame([{"i": i} for i in range(10)])
    assert len(records_from_frame(df, limit=3)) == 3


def test_assemble_snapshot_includes_meta_and_fetchers():
    snapshot = assemble_snapshot(
        {"retrieved_at": "2026-05-29"},
        {"stock_zh_a_hist": pd.DataFrame([{"收盘": 10.0}]), "stock_news": None},
    )
    assert snapshot["_meta"]["retrieved_at"] == "2026-05-29"
    assert snapshot["stock_zh_a_hist"][0]["收盘"] == 10.0
    assert snapshot["stock_news"] == []


def test_write_snapshot_roundtrip(tmp_path):
    path = tmp_path / "sub" / "snapshot.json"
    write_snapshot(path, {"_meta": {"a": 1}, "x": [{"v": 1.5}]})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["_meta"]["a"] == 1
    assert data["x"][0]["v"] == 1.5