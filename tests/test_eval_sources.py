"""Tests for the data source grading loader."""

from finabot.eval.sources import load_sources, source_level


def test_load_sources_has_three_levels():
    sources = load_sources()
    assert set(sources.keys()) == {"p0", "p1", "p2"}
    assert "证监会" in sources["p0"]
    assert "东方财富" in sources["p1"]
    assert "雪球" in sources["p2"]


def test_source_level_maps_original_source():
    assert source_level("巨潮资讯网公告") == "P0"
    assert source_level("东方财富个股新闻") == "P1"
    assert source_level("雪球用户讨论") == "P2"


def test_source_level_unknown_returns_none():
    assert source_level("") is None
    assert source_level("某个不存在的来源") is None