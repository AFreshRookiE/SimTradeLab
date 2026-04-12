from __future__ import annotations

from typing import Any

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.ui.charts import render_drawdown_chart, render_metrics, render_nav_chart, render_trades_table


def create_result_page(config: ETFQuantConfig, shared_state: dict[str, Any] | None = None) -> None:
    shared = shared_state or {}
    ui.label("📊 结果展示").classes("text-h4 q-mb-md").style("color: #c9d1d9")

    svc = shared.get("last_backtest_svc")
    result = shared.get("last_backtest_result")

    if not result or not result.get("success"):
        ui.label("暂无回测结果，请先在「回测执行」页面运行回测").classes("text-body1").style("color: #8b949e")
        return

    render_metrics(result)

    if svc:
        nav_data = svc.get_nav_series()
        render_nav_chart(nav_data)

        dd_data = svc.get_drawdown_series()
        render_drawdown_chart(dd_data)

        trades = svc.get_trades()
        if trades:
            with ui.card().classes("full-width q-mb-md"):
                with ui.row().classes("full-width items-center justify-between"):
                    ui.label("交易明细").classes("text-h6").style("color: #58a6ff")
                    ui.button("💾 导出CSV", on_click=lambda: _export()).props("flat").style("color: #58a6ff")
                render_trades_table(trades, rows_per_page=20)

    def _export():
        from pathlib import Path
        if svc:
            path = str(Path(config.backtest.save_path) / "backtest_result.csv")
            svc.export_result(path)
            ui.notify(f"已导出: {path}", type="positive")
