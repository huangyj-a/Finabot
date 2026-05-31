"""Unified local news tools."""

from __future__ import annotations

import json
import re
from datetime import datetime

from langchain_core.tools import tool


def _identify_stock_type(stock_code: str) -> str:
    code = str(stock_code or "").upper().strip()
    if re.match(r"^(00|30|60|68)\d{4}$", code) or re.match(r"^(SZ|SH)\d{6}$", code):
        return "A股"
    if re.match(r"^\d{4,5}(\.HK)?$", code):
        return "港股"
    if re.match(r"^[A-Z]{1,5}$", code) or ("." in code and not code.endswith(".HK")):
        return "美股"
    return "A股"


def _clean_stock_code(stock_code: str) -> str:
    return str(stock_code or "").upper().replace(".SH", "").replace(".SZ", "").replace(".HK", "").strip()


def _format_dataframe_news(df, source: str, max_news: int) -> str:
    if df is None or getattr(df, "empty", True):
        return ""
    rows = df.head(max(1, int(max_news or 10))).to_dict(orient="records")
    lines = [
        f"=== 📰 新闻数据来源: {source} ===",
        f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"新闻数量: {len(rows)} 条",
        "",
    ]
    for index, row in enumerate(rows, 1):
        title = row.get("新闻标题") or row.get("标题") or row.get("title") or row.get("名称") or "无标题"
        url = row.get("新闻链接") or row.get("链接") or row.get("url") or ""
        publish_time = row.get("发布时间") or row.get("时间") or row.get("日期") or row.get("publish_time") or "未知时间"
        content = row.get("新闻内容") or row.get("摘要") or row.get("内容") or row.get("summary") or ""
        lines.append(f"## {index}. {title}")
        lines.append(f"**时间**: {publish_time} | **来源**: {source}")
        if url:
            lines.append(f"**链接**: {url}")
        if content:
            lines.append(str(content)[:500])
        lines.append("---")
    return "\n".join(lines).strip()


def _get_a_share_news(stock_code: str, max_news: int) -> str:
    try:
        import akshare as ak

        clean_code = _clean_stock_code(stock_code)
        if hasattr(ak, "stock_news_em"):
            result = _format_dataframe_news(ak.stock_news_em(symbol=clean_code), "东方财富个股新闻", max_news)
            if result:
                return result
        if hasattr(ak, "stock_info_global_cls"):
            result = _format_dataframe_news(ak.stock_info_global_cls(symbol="全部"), "财联社电报", max_news)
            if result:
                return result
    except Exception as exc:
        return f"❌ A股新闻获取失败: {exc}"
    return "❌ 无法获取A股新闻数据，所有新闻源均不可用"


def _get_hk_or_us_news(stock_code: str, max_news: int, market: str) -> str:
    try:
        import akshare as ak

        if hasattr(ak, "stock_info_global_cls"):
            result = _format_dataframe_news(ak.stock_info_global_cls(symbol="全部"), f"{market}通用财经新闻", max_news)
            if result:
                return result
    except Exception as exc:
        return f"❌ {market}新闻获取失败: {exc}"
    return f"❌ 无法获取{market}新闻数据，所有新闻源均不可用"


@tool
def get_stock_news_unified(stock_code: str, max_news: int = 10, model_info: str = "") -> str:
    """统一新闻获取工具，按股票代码自动识别A股/港股/美股并返回格式化新闻。"""

    if not stock_code:
        return "❌ 错误: 未提供股票代码"
    stock_type = _identify_stock_type(stock_code)
    if stock_type == "A股":
        news = _get_a_share_news(stock_code, max_news)
    elif stock_type == "港股":
        news = _get_hk_or_us_news(stock_code, max_news, "港股")
    else:
        news = _get_hk_or_us_news(stock_code, max_news, "美股")
    return json.dumps(
        {
            "tool": "get_stock_news_unified",
            "stock_code": stock_code,
            "stock_type": stock_type,
            "fetch_time": datetime.now().isoformat(),
            "model_info": model_info,
            "news": news,
            "has_direct_news": not news.startswith("❌"),
        },
        ensure_ascii=False,
    )
