"""Sample real AKShare data into a frozen fixture snapshot.json.

评估报告"冻结数据离线套件"的内容生成器：给定股票/基金/指数与市场类型，
用与工具层相同的 AKShare fetcher 采样当前数据，写入
``eval/fixtures/<task_id>/snapshot.json`` 供 FrozenData 离线回放。

Usage:
  .venv/Scripts/python.exe scripts/gen_fixtures.py 贵州茅台 --market a --task-id t001
  .venv/Scripts/python.exe scripts/gen_fixtures.py 00700 --market hk --task-id t011
  .venv/Scripts/python.exe scripts/gen_fixtures.py 005827 --market fund --task-id t012
  .venv/Scripts/python.exe scripts/gen_fixtures.py sh000001 --market index --task-id t010
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime

from finabot.eval.fixture_builder import assemble_snapshot, records_from_frame, write_snapshot


@contextlib.contextmanager
def _no_proxy():
    """临时绕过系统代理（Windows 系统代理/环境代理都指向不可达代理时使用）。

    只对 requests 生效：把 get_environ_proxies 置空，使 requests 直连。
    """
    import requests.sessions as _sessions

    original = _sessions.get_environ_proxies
    _sessions.get_environ_proxies = lambda url, no_proxy=None: {}
    try:
        yield
    finally:
        _sessions.get_environ_proxies = original


def _resolve_code(ak, stock: str, market: str) -> str | None:
    """名称→代码；已是代码则直接返回。A股用 stock_info_a_code_name 解析。"""
    if stock.isdigit() and len(stock) == 6:
        return stock
    if market in {"a", "auto"}:
        try:
            lookup = ak.stock_info_a_code_name()
            for _, row in lookup.iterrows():
                name = str(row.get("name", "") or row.get("证券简称", "") or row.get("名称", "") or "")
                code = str(row.get("code", "") or row.get("证券代码", "") or row.get("代码", "") or "")
                if stock and (stock in name or name in stock) and code:
                    return code
        except Exception:
            pass
    return stock  # HK/基金/指数无法本地消歧时，直接用入参作为 symbol


def _samplers_a(ak, code: str) -> dict[str, object]:
    return {
        "stock_info_a_code_name": lambda: ak.stock_info_a_code_name(),
        "stock_zh_a_hist": lambda: ak.stock_zh_a_hist(symbol=code, start_date="20240101", end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq"),
        "stock_zh_a_spot_em": lambda: ak.stock_zh_a_spot_em(),
        "stock_individual_info_em": lambda: ak.stock_individual_info_em(symbol=code),
        "stock_value_em": lambda: ak.stock_value_em(symbol=code),
        "stock_financial_analysis_indicator_em": lambda: ak.stock_financial_analysis_indicator_em(symbol=code),
        "stock_individual_fund_flow": lambda: ak.stock_individual_fund_flow(stock=code),
        "stock_research_report_em": lambda: ak.stock_research_report_em(symbol=code),
        "stock_individual_notice_report": lambda: ak.stock_individual_notice_report(symbol=code),
    }


def _samplers_hk(ak, code: str) -> dict[str, object]:
    return {
        "stock_hk_spot_em": lambda: ak.stock_hk_spot_em(),
        "stock_hk_hist": lambda: ak.stock_hk_hist(symbol=code, start_date="20240101", end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq"),
        "stock_hk_index_spot_em": lambda: ak.stock_hk_index_spot_em(),
    }


def _samplers_fund(ak, code: str) -> dict[str, object]:
    return {
        "fund_open_fund_daily_em": lambda: ak.fund_open_fund_daily_em(),
        "fund_etf_spot_em": lambda: ak.fund_etf_spot_em(),
    }


def _samplers_index(ak, code: str) -> dict[str, object]:
    return {
        "stock_zh_index_spot_em": lambda: ak.stock_zh_index_spot_em(symbol="沪深重要指数"),
        "index_zh_a_hist": lambda: ak.index_zh_a_hist(symbol=code, period="daily", start_date="20240101", end_date=datetime.now().strftime("%Y%m%d")),
    }


_SAMPLERS = {
    "a": _samplers_a,
    "hk": _samplers_hk,
    "fund": _samplers_fund,
    "index": _samplers_index,
}


def sample_market(ak, market: str, code: str) -> dict[str, object]:
    samplers = _SAMPLERS.get(market, _samplers_a)
    fetched: dict[str, object] = {}
    for name, fn in samplers(ak, code).items():
        try:
            fetched[name] = records_from_frame(fn())
        except Exception:
            fetched[name] = []
    return fetched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", help="股票/基金/指数名称或代码")
    parser.add_argument("--market", default="a", choices=["a", "hk", "fund", "index", "auto"], help="市场类型（默认 a）")
    parser.add_argument("--task-id", default=None, help="输出到 eval/fixtures/<task-id>/snapshot.json")
    parser.add_argument("--out", default=None, help="输出 JSON 路径（覆盖 --task-id）")
    parser.add_argument("--no-proxy", action="store_true", help="绕过系统代理（系统代理不可达时使用）")
    args = parser.parse_args()

    import akshare as ak

    market = args.market
    if market == "auto":
        # 简单启发式：含 .HK → hk；5 位数字 → 基金；6 位数字且以 000/399 开头 → 指数；否则 a
        raw = args.stock.upper()
        if ".HK" in raw:
            market = "hk"
        elif raw.isdigit() and len(raw) == 5:
            market = "fund"
        elif raw.isdigit() and raw.startswith(("000", "399", "sh", "sz")):
            market = "index"
        else:
            market = "a"

    code = _resolve_code(ak, args.stock, market)
    if code is None:
        raise SystemExit(f"无法解析标的：{args.stock}")

    meta = {
        "fixture_version": "0.1",
        "task_id": args.task_id or "",
        "market": market,
        "resolved_symbol": code,
        "retrieved_at": datetime.now().isoformat(),
        "note": "由 scripts/gen_fixtures.py 采样生成；评估时请把任务 as_of 设为快照内 latest_trade_date。",
    }

    sample_ctx = _no_proxy() if args.no_proxy else contextlib.nullcontext()
    with sample_ctx:
        snapshot = assemble_snapshot(meta, sample_market(ak, market, code))

    if args.out:
        path = args.out
    elif args.task_id:
        path = f"eval/fixtures/{args.task_id}/snapshot.json"
    else:
        path = f"eval/fixtures/{code}/snapshot.json"

    write_snapshot(path, snapshot)
    print(f"已写入快照：{path}（market={market} symbol={code}，{len(snapshot) - 1} 个 fetcher）")


if __name__ == "__main__":
    main()