from __future__ import annotations

import time
from typing import Any

import pandas as pd

from etfquant.core.config import DataConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = ["DataService"]

logger = get_logger("etfquant.api.data")


class DataService:
    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._bridge = DataBridge(config)
        self._golden_cross_cache: list[dict[str, Any]] = []
        self._golden_cross_cache_time: float = 0

    def list_etfs(self, category: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
        codes = self._bridge.list_etf_codes(category)
        result = []
        for code in codes:
            info = self._bridge.classification.etf_map.get(code)
            name = info.name if info else ""
            if search:
                q = search.lower()
                if q not in code.lower() and q not in name.lower():
                    continue
            result.append({
                "code": code,
                "name": name,
                "category": info.category if info else "",
                "tracking_index": info.tracking_index if info else "",
            })
        return result

    def get_golden_cross_etfs(self, top_n: int = 50, short_window: int = 5, long_window: int = 10, lookback: int = 5) -> list[dict[str, Any]]:
        if self._golden_cross_cache and (time.time() - self._golden_cross_cache_time) < 300:
            return self._golden_cross_cache[:top_n]

        signals: dict[str, tuple[str, float, int]] = {}
        codes = self._bridge.list_etf_codes()
        min_rows = long_window + lookback + 1
        for code in codes:
            df = self._bridge.load_etf_daily(code)
            if df.empty or len(df) < min_rows:
                continue
            close = df["close"].astype(float)
            tail = close.iloc[-(min_rows + 5):]
            ma_short = tail.rolling(short_window).mean()
            ma_long = tail.rolling(long_window).mean()
            diff = ma_short - ma_long
            if diff.isna().all():
                continue
            recent_diff = diff.iloc[-lookback:]
            prev_diff = diff.iloc[-lookback - 1]
            crossed = False
            cross_day = -1
            for i in range(len(recent_diff)):
                curr = recent_diff.iloc[i]
                prev = prev_diff if i == 0 else recent_diff.iloc[i - 1]
                if pd.isna(curr) or pd.isna(prev):
                    continue
                if prev <= 0 and curr > 0:
                    crossed = True
                    cross_day = i
                    break
            if not crossed:
                if recent_diff.iloc[-1] > 0 and prev_diff <= 0:
                    crossed = True
                    cross_day = 0
            latest_diff = diff.iloc[-1]
            if crossed and not pd.isna(latest_diff) and latest_diff > 0:
                strength = float(latest_diff / tail.iloc[-1] * 100)
                signal_label = f"MA{short_window}↑MA{long_window}"
                signals[code] = (signal_label, strength, cross_day)
        sorted_codes = sorted(codes, key=lambda c: (-signals[c][2] if c in signals else 999, -signals[c][1] if c in signals else -999))
        result = []
        for code in sorted_codes:
            info = self._bridge.classification.etf_map.get(code)
            if code in signals:
                sig_label, strength, _ = signals[code]
                result.append({
                    "code": code,
                    "name": info.name if info else "",
                    "category": info.category if info else "",
                    "tracking_index": info.tracking_index if info else "",
                    "signal": sig_label,
                    "strength": float(round(strength, 2)),
                })
            else:
                result.append({
                    "code": code,
                    "name": info.name if info else "",
                    "category": info.category if info else "",
                    "tracking_index": info.tracking_index if info else "",
                    "signal": "",
                    "strength": 0.0,
                })
        self._golden_cross_cache = result
        self._golden_cross_cache_time = time.time()
        return result[:top_n]

    def search_etfs(self, query: str, limit: int = 20) -> list[dict[str, str]]:
        if not query or len(query) < 1:
            return []
        q = query.lower()
        matches = []
        for code in self._bridge.list_etf_codes():
            info = self._bridge.classification.etf_map.get(code)
            name = info.name if info else ""
            if q in code.lower() or q in name.lower():
                matches.append({"code": code, "name": name, "label": f"{code} {name}"})
                if len(matches) >= limit:
                    break
        return matches

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
        adjust_df = self._bridge.load_etf_adjust_factor(code)
        detail["has_adjust"] = not adjust_df.empty
        return detail

    def get_etf_chart_data(self, code: str, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        df = self._bridge.load_etf_daily(code)
        if df.empty:
            return []
        df = df.copy()
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
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
        from etfquant.data.bridge import DataBridge
        DataBridge.clear_cache()
        self._golden_cross_cache = []
        self._golden_cross_cache_time = 0
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
