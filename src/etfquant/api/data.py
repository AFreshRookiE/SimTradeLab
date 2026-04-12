from __future__ import annotations

from typing import Any

from etfquant.core.config import DataConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = ["DataService"]

logger = get_logger("etfquant.api.data")


class DataService:
    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._bridge = DataBridge(config)

    def list_etfs(self, category: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
        codes = self._bridge.list_etf_codes(category)
        if search:
            codes = [c for c in codes if search.lower() in c.lower()]
        result = []
        for code in codes:
            info = self._bridge.classification.etf_map.get(code)
            result.append({
                "code": code,
                "name": info.name if info else "",
                "category": info.category if info else "",
                "tracking_index": info.tracking_index if info else "",
            })
        return result

    def get_etf_detail(self, code: str) -> dict[str, Any]:
        df = self._bridge.load_etf_daily(code)
        info = self._bridge.classification.etf_map.get(code)
        detail: dict[str, Any] = {
            "code": code,
            "name": info.name if info else "",
            "category": info.category if info else "",
            "tracking_index": info.tracking_index if info else "",
            "has_daily": not df.empty,
        }
        if not df.empty:
            close = df["close"].astype(float)
            ret = close.pct_change().dropna()
            detail.update({
                "start_date": str(df.index.min().date()),
                "end_date": str(df.index.max().date()),
                "rows": len(df),
                "latest_close": float(close.iloc[-1]),
                "avg_daily_return": float(ret.mean()),
                "daily_volatility": float(ret.std()),
                "high": float(close.max()),
                "low": float(close.min()),
            })
        nav_df = self._bridge.load_etf_nav(code)
        detail["has_nav"] = not nav_df.empty
        premium_df = self._bridge.load_etf_premium(code)
        detail["has_premium"] = not premium_df.empty
        return detail

    def get_coverage(self) -> dict[str, Any]:
        codes = self._bridge.list_etf_codes()
        categories: dict[str, int] = {}
        for code in codes:
            info = self._bridge.classification.etf_map.get(code)
            cat = info.category if info else "未分类"
            categories[cat] = categories.get(cat, 0) + 1
        sample_details = []
        for code in codes[:10]:
            df = self._bridge.load_etf_daily(code)
            if not df.empty:
                sample_details.append({
                    "code": code,
                    "start": str(df.index.min().date()),
                    "end": str(df.index.max().date()),
                    "rows": len(df),
                })
        return {
            "total_etf_count": len(codes),
            "categories": categories,
            "sample_details": sample_details,
        }

    def get_etf_chart_data(self, code: str) -> list[dict[str, Any]]:
        df = self._bridge.load_etf_daily(code)
        if df.empty:
            return []
        result = []
        for date, row in df.iterrows():
            item = {"date": str(date.date())}
            for col in ["open", "high", "low", "close", "volume"]:
                if col in row.index:
                    val = row[col]
                    item[col] = float(val) if val is not None else None
            result.append(item)
        return result

    def refresh_data(self) -> dict[str, Any]:
        self._bridge.clear_cache()
        codes = self._bridge.list_etf_codes()
        nav_count = 0
        premium_count = 0
        daily_count = 0
        for code in codes[:5]:
            df = self._bridge.load_etf_daily(code)
            if not df.empty:
                daily_count += 1
            nav_df = self._bridge.load_etf_nav(code)
            if not nav_df.empty:
                nav_count += 1
            premium_df = self._bridge.load_etf_premium(code)
            if not premium_df.empty:
                premium_count += 1
        return {
            "status": "ok",
            "message": f"数据缓存已刷新: {len(codes)} 只ETF, {daily_count} 有日线, {nav_count} 有净值, {premium_count} 有溢价",
            "total_etf": len(codes),
            "daily_count": daily_count,
            "nav_count": nav_count,
            "premium_count": premium_count,
        }
