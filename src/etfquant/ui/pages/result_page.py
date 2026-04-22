from __future__ import annotations

from typing import Any

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.api.backtest import BacktestService


_STRATEGY_LABELS = {
    "ma_cross": "MA均线交叉",
    "momentum": "动量策略",
    "mean_reversion": "均值回归",
    "ml_model": "ML模型策略",
}


def create_result_page(config: ETFQuantConfig, shared_state: dict[str, Any] | None = None) -> None:
    ui.label("📊 回测历史与对比分析").classes("text-h4 q-mb-md").style("color: #c9d1d9")
    svc = BacktestService(config.backtest, config.data)

    with ui.card().classes("full-width q-mb-md"):
        ui.label("回测历史记录").classes("text-h6 q-mb-md").style("color: #58a6ff")
        ui.markdown(
            "每次回测执行的结果会自动保存到历史记录中。勾选多条记录后点击'对比分析'，可横向比较不同策略/ETF的表现差异。"
        ).style("color: #8b949e; font-size: 13px; line-height: 1.6;")
        with ui.row().classes("full-width items-center q-mb-sm"):
            filter_code = ui.input(label="按ETF代码筛选", value="").classes("q-mr-md").style("min-width: 160px")
            filter_strategy = ui.select(
                label="按策略筛选",
                options={"": "全部", **_STRATEGY_LABELS},
                value="",
            ).classes("q-mr-md").style("min-width: 160px")
            ui.button("🔍 筛选", on_click=lambda: _refresh_history()).props("flat").style("color: #58a6ff")
            ui.button("🔄 刷新", on_click=lambda: _refresh_history()).props("flat").style("color: #58a6ff")
            ui.button("🗑 删除选中", on_click=lambda: _delete_selected()).props("flat").style("color: #f85149")
            ui.button("📊 对比分析", on_click=lambda: _compare_selected(), color="primary")
        history_table = ui.table(
            columns=[
                {"name": "timestamp", "label": "时间", "field": "timestamp", "align": "left", "sortable": True, "width": "140px"},
                {"name": "code", "label": "ETF代码", "field": "code", "align": "left", "sortable": True, "width": "100px"},
                {"name": "strategy_label", "label": "策略", "field": "strategy_label", "align": "center", "width": "100px"},
                {"name": "model_name", "label": "模型", "field": "model_name", "align": "center", "width": "120px"},
                {"name": "annual_return", "label": "年化收益", "field": "annual_return", "align": "right", "sortable": True, "width": "90px"},
                {"name": "sharpe_ratio", "label": "夏普比率", "field": "sharpe_ratio", "align": "right", "sortable": True, "width": "90px"},
                {"name": "max_drawdown", "label": "最大回撤", "field": "max_drawdown", "align": "right", "sortable": True, "width": "90px"},
                {"name": "win_rate", "label": "胜率", "field": "win_rate", "align": "right", "width": "70px"},
                {"name": "total_trades", "label": "交易次数", "field": "total_trades", "align": "center", "width": "80px"},
            ],
            rows=[],
            row_key="id",
            selection="multiple",
            pagination={"rowsPerPage": 15, "rowsPerPageOptions": [10, 15, 30, 50]},
        ).classes("full-width").props('resizable-columns separator="cell"')

    compare_container = ui.column().classes("full-width")

    def _refresh_history():
        code = filter_code.value.strip() or None
        strategy = filter_strategy.value or None
        history = svc.get_backtest_history(code=code, strategy_type=strategy, limit=200)
        rows = []
        for h in reversed(history):
            ts = h.get("timestamp", "")
            if "T" in ts:
                ts = ts.replace("T", " ")[:19]
            rows.append({
                "id": h.get("id", ""),
                "timestamp": ts,
                "code": h.get("code", ""),
                "strategy_label": _STRATEGY_LABELS.get(h.get("strategy_type", ""), h.get("strategy_type", "")),
                "model_name": h.get("model_name", ""),
                "annual_return": f"{h.get('annual_return', 0):.2%}",
                "sharpe_ratio": f"{h.get('sharpe_ratio', 0):.4f}",
                "max_drawdown": f"{h.get('max_drawdown', 0):.2%}",
                "win_rate": f"{h.get('win_rate', 0):.2%}",
                "total_trades": h.get("total_trades", 0),
            })
        history_table.rows = rows

    def _delete_selected():
        selected = history_table.selected
        if not selected:
            ui.notify("请先勾选要删除的记录", type="warning")
            return
        count = 0
        for row in selected:
            entry_id = row["id"] if isinstance(row, dict) else row
            if svc.delete_backtest_history(entry_id):
                count += 1
        ui.notify(f"已删除 {count} 条记录", type="positive")
        history_table.selected = []
        _refresh_history()

    def _compare_selected():
        selected = history_table.selected
        if not selected or len(selected) < 2:
            ui.notify("请至少勾选2条记录进行对比", type="warning")
            return
        entry_ids = [r["id"] if isinstance(r, dict) else r for r in selected]
        records = svc.get_backtest_comparison(entry_ids)
        if not records:
            ui.notify("未找到对比数据", type="negative")
            return

        compare_container.clear()
        with compare_container:
            with ui.card().classes("full-width q-mb-md"):
                ui.label("📊 对比分析").classes("text-h6 q-mb-md").style("color: #58a6ff")

                compare_cols = [
                    {"name": "metric", "label": "指标", "field": "metric", "align": "left", "width": "120px"},
                ]
                for r in records:
                    label = f"{r.get('code', '')} | {_STRATEGY_LABELS.get(r.get('strategy_type', ''), r.get('strategy_type', ''))}"
                    if r.get("model_name"):
                        label += f" | {r['model_name']}"
                    compare_cols.append({"name": r["id"], "label": label, "field": r["id"], "align": "right", "width": "150px"})

                metrics_list = [
                    ("年化收益率", "annual_return", "pct"),
                    ("夏普比率", "sharpe_ratio", "f4"),
                    ("索提诺比率", "sortino_ratio", "f4"),
                    ("卡尔玛比率", "calmar_ratio", "f4"),
                    ("最大回撤", "max_drawdown", "pct"),
                    ("胜率", "win_rate", "pct"),
                    ("盈亏比", "profit_loss_ratio", "f4"),
                    ("总收益率", "total_return", "pct"),
                    ("交易次数", "total_trades", "int"),
                    ("回测区间", "period", "str"),
                ]

                compare_rows = []
                for metric_name, metric_key, fmt in metrics_list:
                    row = {"metric": metric_name}
                    best_val = None
                    best_id = None
                    if fmt in ("pct", "f4") and metric_key not in ("max_drawdown",):
                        for r in records:
                            v = r.get(metric_key, 0)
                            if best_val is None or v > best_val:
                                best_val = v
                                best_id = r["id"]
                    for r in records:
                        v = r.get(metric_key, 0)
                        if metric_key == "period":
                            row[r["id"]] = f"{r.get('start_date', '')[:10]} ~ {r.get('end_date', '')[:10]}"
                        elif fmt == "pct":
                            row[r["id"]] = f"{v:.2%}"
                        elif fmt == "f4":
                            row[r["id"]] = f"{v:.4f}"
                        elif fmt == "int":
                            row[r["id"]] = str(int(v))
                        else:
                            row[r["id"]] = str(v)
                    compare_rows.append(row)

                ui.table(
                    columns=compare_cols,
                    rows=compare_rows,
                    row_key="metric",
                ).classes("full-width").props('separator="cell"')

                with ui.expansion("💡 对比结论", icon="lightbulb").classes("full-width q-mt-md").style("border: 1px solid #30363d; border-radius: 8px;"):
                    best_sharpe = max(records, key=lambda r: r.get("sharpe_ratio", 0))
                    best_return = max(records, key=lambda r: r.get("annual_return", 0))
                    min_dd = min(records, key=lambda r: abs(r.get("max_drawdown", 0)))
                    best_sharpe_label = f"{best_sharpe.get('code', '')} ({_STRATEGY_LABELS.get(best_sharpe.get('strategy_type', ''), '')})"
                    best_return_label = f"{best_return.get('code', '')} ({_STRATEGY_LABELS.get(best_return.get('strategy_type', ''), '')})"
                    min_dd_label = f"{min_dd.get('code', '')} ({_STRATEGY_LABELS.get(min_dd.get('strategy_type', ''), '')})"
                    ui.markdown(f"""
**对比结论：**

- 🏆 **夏普比率最高**: {best_sharpe_label}，夏普 = {best_sharpe.get('sharpe_ratio', 0):.4f}
- 💰 **年化收益最高**: {best_return_label}，年化 = {best_return.get('annual_return', 0):.2%}
- 🛡 **回撤最小**: {min_dd_label}，最大回撤 = {min_dd.get('max_drawdown', 0):.2%}

**实盘建议**：优先选择夏普比率最高的组合（风险调整后收益最优），同时关注最大回撤是否在可接受范围内。
""").style("color: #c9d1d9; font-size: 13px; line-height: 1.6;")

    _refresh_history()
