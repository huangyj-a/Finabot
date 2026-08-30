"""Sample real AKShare data into a frozen fixture snapshot.json.

评估报告"冻结数据离线套件"的内容生成器：给定股票名称/代码，用与
hold_analysis_pipeline 相同的 AKShare fetcher 采样当前数据，写入
``eval/fixtures/<task_id>/snapshot.json`` 供 FrozenData 离线回放。

Usage:
  .venv/Scripts/python.exe scripts/gen_fixtures.py 贵州茅台 --task-id t001
"""

from __future__ import annotations

import argparse
from datetime import datetime

from finabot.eval.fixture_builder import assemble_snapshot, records_from_frame, write_snapshot


def _resolve_code(ak, stock: str) -> str | None:
    """名称→代码；已是 6 位代码则直接返回。"""
    if stock.isdigit() and len(stock) == 6:
        return stock
    lookup = ak.stock_info_a_code_name()
    for _, row in lookup.iterrows():
        name = str(row.get("证券简称", "") or row.get("名称", "") or "")
        code = str(row.get("证券代码", "") or row.get("代码", "") or "")
        if stock in name and code:
            return code
    return None


def _samplers(ak, code: str) -> dict[str, object]:
    """与 hold pipeline 相同的 fetcher 采样（尽力而为，单个失败返回 []）。"""
    samplers = {
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
    fetched: dict[str, object] = {}
    for name, fn in samplers.items():
        try:
            fetched[name] = records_from_frame(fn())
        except Exception:
            fetched[name] = []
    return fetched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", help="股票名称或 6 位代码")
    parser.add_argument("--task-id", default=None, help="输出到 eval/fixtures/<task-id>/snapshot.json")
    parser.add_argument("--out", default=None, help="输出 JSON 路径（覆盖 --task-id）")
    args = parser.parse_args()

    import akshare as ak

    code = _resolve_code(ak, args.stock)
    if code is None:
        raise SystemExit(f"无法解析股票：{args.stock}")

    meta = {
        "fixture_version": "0.1",
        "task_id": args.task_id or "",
        "resolved_code": code,
        "retrieved_at": datetime.now().isoformat(),
        "note": "由 scripts/gen_fixtures.py 采样生成；评估时请把任务 as_of 设为 latest_trade_date。",
    }
    snapshot = assemble_snapshot(meta, _samplers(ak, code))

    if args.out:
        path = args.out
    elif args.task_id:
        path = f"eval/fixtures/{args.task_id}/snapshot.json"
    else:
        path = f"eval/fixtures/{code}/snapshot.json"

    write_snapshot(path, snapshot)
    print(f"已写入快照：{path}（股票 {code}，{len(snapshot) - 1} 个 fetcher）")


if __name__ == "__main__":
    main()