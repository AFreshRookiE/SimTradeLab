from __future__ import annotations

from nicegui import ui

from etfquant.ui.echart_theme import (
    DRAWDOWN_SERIES,
    ECHART_GRID,
    ECHART_XAXIS,
    ECHART_YAXIS,
    ECHART_YAXIS_PCT,
    METRICS_ITEMS,
    NAV_SERIES,
    TRADE_COLUMNS,
)


def format_trades(trades: list[dict]) -> list[dict]:
    result = []
    for t in trades:
        result.append({
            "date": t.get("date", ""),
            "code": t.get("code", ""),
            "action": t.get("action", ""),
            "price": f"{t.get('price', 0):.4f}",
            "shares": t.get("shares", ""),
            "amount": f"{t.get('amount', 0):,.2f}",
            "commission": f"{t.get('commission', 0):.2f}",
            "profit": f"{t.get('profit', 0):.2f}",
        })
    return result


def _format_value(data: dict, label: str, key: str | None, fmt: str | None) -> str:
    if key == "period":
        return f"{data.get('start_date', '')} ~ {data.get('end_date', '')}"
    val = data.get(key or label, 0)
    if fmt == "comma":
        return f"{val:,.0f}"
    if fmt == "pct_green":
        return f"{val:.2%}"
    if fmt == "pct_red":
        return f"{val:.2%}"
    if fmt == "pct":
        return f"{val:.2%}"
    if fmt == "f4":
        return f"{val:.4f}"
    if fmt == "int":
        return str(int(val))
    return str(val)


def _metric_color(label: str, fmt: str | None) -> str:
    if fmt == "pct_green":
        return "#3fb950"
    if fmt == "pct_red":
        return "#f85149"
    return "#c9d1d9"


def render_metrics(data: dict) -> None:
    with ui.card().classes("full-width q-mb-md"):
        ui.label("回测指标").classes("text-h6 q-mb-md").style("color: #58a6ff")
        with ui.row().classes("full-width wrap"):
            for label, key, fmt in METRICS_ITEMS:
                value = _format_value(data, label, key, fmt)
                color = _metric_color(label, fmt)
                with ui.card().classes("q-ma-xs q-pa-sm metric-card"):
                    ui.label(label).classes("text-caption").style("color: #8b949e")
                    ui.label(value).classes("text-body1 text-weight-bold").style(f"color: {color}")


def render_nav_chart(nav_data: list[dict]) -> None:
    if not nav_data:
        return
    with ui.card().classes("full-width q-mb-md"):
        ui.label("净值曲线").classes("text-h6 q-mb-sm").style("color: #58a6ff")
        dates = [d["date"] for d in nav_data]
        navs = [d["nav"] for d in nav_data]
        series = {**NAV_SERIES, "name": "净值", "data": navs}
        ui.echart({
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["净值"], "textStyle": {"color": "#8b949e"}},
            "grid": ECHART_GRID,
            "xAxis": {"type": "category", "data": dates, **ECHART_XAXIS},
            "yAxis": {"type": "value", **ECHART_YAXIS},
            "series": [series],
        }).classes("full-width")


def render_drawdown_chart(dd_data: list[dict]) -> None:
    if not dd_data:
        return
    with ui.card().classes("full-width q-mb-md"):
        ui.label("回撤图").classes("text-h6 q-mb-sm").style("color: #58a6ff")
        dates = [d["date"] for d in dd_data]
        dds = [d["drawdown"] * 100 for d in dd_data]
        series = {**DRAWDOWN_SERIES, "data": dds}
        ui.echart({
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "grid": ECHART_GRID,
            "xAxis": {"type": "category", "data": dates, **ECHART_XAXIS},
            "yAxis": {"type": "value", **ECHART_YAXIS_PCT},
            "series": [series],
        }).classes("full-width")


def render_trades_table(trades: list[dict], rows_per_page: int = 15) -> None:
    formatted = format_trades(trades)
    with ui.card().classes("full-width q-mb-md"):
        ui.label("交易明细").classes("text-h6 q-mb-sm").style("color: #58a6ff")
        ui.table(
            columns=TRADE_COLUMNS,
            rows=formatted,
            pagination={"rowsPerPage": rows_per_page},
        ).classes("full-width")
