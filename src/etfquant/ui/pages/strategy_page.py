from __future__ import annotations

import asyncio
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.api.strategy import StrategyService
from etfquant.ui.charts import render_drawdown_chart, render_metrics, render_nav_chart, render_trades_table


def _prev_trading_day(offset: int = 1) -> str:
    today = date.today()
    if today.weekday() == 5:
        delta = 1 + offset - 1
    elif today.weekday() == 6:
        delta = 2 + offset - 1
    elif today.weekday() == 0:
        delta = 3 + offset - 1
    else:
        delta = offset
    prev = today - timedelta(days=delta)
    return prev.strftime("%Y%m%d")


def _load_etf_options(config: ETFQuantConfig) -> list[dict[str, str]]:
    from etfquant.data.bridge import DataBridge
    try:
        bridge = DataBridge(config.data)
        codes = bridge.list_etf_codes()
        names = bridge.get_etf_names()
        options = []
        for code in codes:
            name = names.get(code, "")
            label = f"{code} {name}" if name else code
            options.append({"label": label, "value": code})
        return options
    except Exception:
        return []


class _StrategyBacktestHistory:
    """策略编辑页的回测历史存储（内存 + 文件持久化）。"""

    def __init__(self, config: ETFQuantConfig) -> None:
        self._path = Path(config.backtest.save_path) / "strategy_backtest_history.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._items = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._items = []

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._items, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, record: dict[str, Any]) -> None:
        record["id"] = f"bt_{int(time.time() * 1000)}"
        record["timestamp"] = time.strftime("%m/%d %H:%M:%S")
        self._items.insert(0, record)
        self._items = self._items[:200]
        self._save()

    def list(self) -> list[dict[str, Any]]:
        return list(self._items)

    def get(self, bt_id: str) -> dict[str, Any] | None:
        for item in self._items:
            if item.get("id") == bt_id:
                return item
        return None


def create_strategy_page(config: ETFQuantConfig) -> None:
    ui.label("📝 策略编辑").classes("text-h4 q-mb-md").style("color: #c9d1d9")
    svc = StrategyService(config.backtest)
    history = _StrategyBacktestHistory(config)

    with ui.splitter(value=22).classes("full-width").style("height: calc(100vh - 140px)") as splitter:
        with splitter.before:
            ui.label("策略列表").classes("text-subtitle1 q-mb-sm").style("color: #58a6ff")
            strategy_list = ui.column().classes("full-width").style("overflow-y: auto; max-height: calc(100vh - 360px);")

            ui.separator().classes("q-my-md")
            ui.label("模板策略").classes("text-subtitle1 q-mb-sm").style("color: #58a6ff")
            template_list = ui.column().classes("full-width")

        with splitter.after:
            with ui.splitter(value=65, horizontal=True).classes("full-width").style("height: 100%") as inner_splitter:
                with inner_splitter.before:
                    with ui.column().classes("full-width q-pa-md").style("height: 100%; overflow-y: auto;"):
                        with ui.row().classes("full-width items-center justify-between q-mb-sm"):
                            name_input = ui.input(value="my_strategy", placeholder="策略名称（字母数字下划线）").props("dense outlined").classes("q-mr-md").style("max-width: 280px")
                            ui.button("💾 保存", on_click=lambda: _save(), color="primary").props("flat dense").style("color: #58a6ff")
                            ui.button("🗑 删除", on_click=lambda: _confirm_delete(), color="negative").props("flat dense").style("color: #f85149")
                            status_label = ui.label("").classes("text-body2 q-ml-md").style("color: #8b949e")

                        with ui.expansion("⚙️ 回测参数", icon="tune").classes("full-width q-mb-sm").style("border: 1px solid #30363d; border-radius: 8px;"):
                            with ui.row().classes("full-width wrap items-center"):
                                etf_options = _load_etf_options(config)
                                etf_labels = [o["label"] for o in etf_options] if etf_options else [config.backtest.benchmark]
                                default_label = next((o["label"] for o in etf_options if o["value"] == config.backtest.benchmark), etf_labels[0] if etf_labels else config.backtest.benchmark)
                                code_select = ui.select(
                                    label="回测ETF",
                                    options=etf_labels,
                                    value=default_label,
                                    with_input=True,
                                    new_value_mode="add",
                                ).classes("q-mr-md q-mb-sm").style("min-width: 220px")
                                start_date_input = ui.input(label="开始日期", value="20160101", placeholder="YYYYMMDD").props("dense outlined").classes("q-mr-md q-mb-sm").style("min-width: 140px")
                                end_date_input = ui.input(label="结束日期", value=_prev_trading_day(), placeholder="YYYYMMDD").props("dense outlined").classes("q-mr-md q-mb-sm").style("min-width: 140px")
                                capital_input = ui.number(label="初始资金", value=config.backtest.initial_capital, min=1000, step=10000).props("dense outlined").classes("q-mr-md q-mb-sm").style("min-width: 140px")
                                commission_input = ui.number(label="佣金率", value=config.backtest.commission_rate, format="%.5f", step=0.0001).props("dense outlined").classes("q-mr-md q-mb-sm").style("min-width: 120px")
                                slippage_input = ui.number(label="滑点率", value=config.backtest.slippage_rate, format="%.5f", step=0.0001).props("dense outlined").classes("q-mr-md q-mb-sm").style("min-width: 120px")

                        with ui.row().classes("full-width items-center q-mb-sm"):
                            ui.button("▶ 运行回测", on_click=lambda: _run_backtest(), color="positive").props("flat dense").style("color: #3fb950")
                            run_status = ui.label("").classes("text-body2 q-ml-md").style("color: #8b949e")

                        code_editor = ui.codemirror(
                            value="",
                            language="Python",
                            theme="githubDark",
                            line_wrapping=True,
                        ).classes("full-width").style("min-height: 360px")

                        with ui.expansion("📖 策略编写说明", icon="help_outline").classes("full-width q-mt-sm").style("border: 1px solid #30363d; border-radius: 8px;"):
                            ui.markdown(
                                "**策略结构：**\n\n"
                                "```python\n"
                                "def initialize(context, data):\n"
                                "    context.fast_period = 5  # 设置参数\n\n"
                                "def handle_data(context, data):\n"
                                "    # data.close / data.open / data.high / data.low\n"
                                "    # data.volume / data.nav / data.premium_rate\n"
                                "    # context.position (当前持仓股数)\n"
                                "    # context.xxx (initialize中设置的自定义参数)\n"
                                "    order_target_percent(context, 0.95)  # 买入95%\n"
                                "    order_target_percent(context, 0)     # 清仓\n"
                                "```\n\n"
                                "**可用数据：** `data.close`, `data.open`, `data.high`, `data.low`, "
                                "`data.volume`, `data.nav`, `data.premium_rate`（均为 pandas Series）\n\n"
                                "**可用函数：** `order_target_percent(context, percent)` — 调整仓位至指定比例\n\n"
                                "**context 属性：** `context.position` 当前持仓股数；可在 `initialize` 中设置任意自定义参数"
                            ).style("color: #8b949e; font-size: 12px; line-height: 1.6;")

                        result_container = ui.column().classes("full-width q-mt-md")

                with inner_splitter.after:
                    with ui.column().classes("full-width q-pa-sm").style("height: 100%; overflow-y: auto;"):
                        ui.label("📜 回测历史").classes("text-subtitle1 q-mb-sm").style("color: #58a6ff")
                        history_list = ui.column().classes("full-width")

                        ui.separator().classes("q-my-md")
                        ui.label("🖥 终端日志").classes("text-subtitle1 q-mb-sm").style("color: #58a6ff")
                        log_container = ui.column().classes("full-width").style("flex: 1; min-height: 300px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 8px; overflow-y: auto;")

    current_strategy_id = {"value": ""}
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        log_lines.append(line)
        if len(log_lines) > 200:
            log_lines.pop(0)
        log_container.clear()
        with log_container:
            for l in log_lines[-50:]:
                ui.label(l).classes("text-caption").style("color: #8b949e; font-family: monospace; line-height: 1.4;")

    def _refresh_history():
        history_list.clear()
        with history_list:
            items = history.list()
            if not items:
                ui.label("暂无回测记录").classes("text-body2").style("color: #8b949e")
            for item in items:
                ann = item.get("annual_return", 0)
                ann_color = "#3fb950" if ann >= 0 else "#f85149"
                with ui.card().classes("q-mb-xs q-pa-xs").style("background: #161b22; border: 1px solid #30363d; cursor: pointer;"):
                    with ui.row().classes("full-width items-center justify-between"):
                        with ui.column():
                            ui.label(f"{item.get('timestamp', '')}  {item.get('strategy_name', '自定义策略')}").classes("text-caption").style("color: #8b949e")
                            ui.label(f"{item.get('code', '')}  {item.get('start_date', '')}~{item.get('end_date', '')}").classes("text-caption").style("color: #8b949e")
                        with ui.column().classes("items-end"):
                            ui.label(f"{ann:.2%}").classes("text-body2 text-weight-bold").style(f"color: {ann_color}")
                            ui.label(f"夏普 {item.get('sharpe_ratio', 0):.2f}  回撤 {item.get('max_drawdown', 0):.2%}").classes("text-caption").style("color: #8b949e")
                    ui.button("查看详情", on_click=lambda i=item: _show_history_result(i)).props("flat dense").classes("full-width").style("color: #58a6ff")

    def _show_history_result(item: dict[str, Any]) -> None:
        result_container.clear()
        with result_container:
            ui.separator().classes("q-mb-md")
            ui.label(f"📊 历史回测结果 — {item.get('timestamp', '')}").classes("text-h6 q-mb-md").style("color: #58a6ff")
            render_metrics(item)
            nav_data = item.get("nav_series", [])
            if nav_data:
                render_nav_chart(nav_data)
            dd_data = item.get("drawdown_series", [])
            if dd_data:
                render_drawdown_chart(dd_data)
            trades = item.get("trades", [])
            if trades:
                render_trades_table(trades)

    def _refresh_list():
        strategy_list.clear()
        with strategy_list:
            strategies = svc.list_strategies()
            if not strategies:
                ui.label("暂无策略").classes("text-body2").style("color: #8b949e")
            for s in strategies:
                ui.button(
                    s["name"],
                    on_click=lambda sid=s["id"]: _load_strategy(sid),
                ).classes("full-width q-mb-xs").props("flat dense align=left").style("color: #c9d1d9")

        template_list.clear()
        with template_list:
            templates = svc.list_templates()
            for t in templates:
                ui.button(
                    f"📋 {t['name']}",
                    on_click=lambda tid=t["id"]: _load_template(tid),
                ).classes("full-width q-mb-xs").props("flat dense align=left").style("color: #8b949e")

    def _load_strategy(sid: str):
        result = svc.get_strategy(sid)
        if result:
            current_strategy_id["value"] = sid
            name_input.value = sid
            code_editor.value = result["content"]
            status_label.text = f"已加载: {sid}"
            status_label.style("color: #3fb950")

    def _load_template(tid: str):
        code = svc.get_template(tid)
        current_strategy_id["value"] = ""
        name_input.value = tid
        code_editor.value = code
        status_label.text = f"已加载模板: {tid}"
        status_label.style("color: #58a6ff")

    def _save():
        sid = name_input.value.strip()
        if not sid:
            ui.notify("请输入策略名称", type="warning")
            return
        try:
            svc.save_strategy(sid, code_editor.value)
        except ValueError as e:
            ui.notify(str(e), type="negative")
            return
        current_strategy_id["value"] = sid
        status_label.text = f"已保存: {sid}"
        status_label.style("color: #3fb950")
        _refresh_list()
        ui.notify(f"策略已保存: {sid}", type="positive")

    def _confirm_delete():
        sid = current_strategy_id["value"] or name_input.value.strip()
        if not sid:
            ui.notify("请先选择或输入要删除的策略名称", type="warning")
            return
        with ui.dialog() as dialog, ui.card():
            ui.label(f"确定删除策略「{sid}」？").classes("text-body1 q-mb-md").style("color: #c9d1d9")
            ui.label("删除后无法恢复").classes("text-body2 q-mb-md").style("color: #f85149")
            with ui.row():
                ui.button("取消", on_click=dialog.close).props("flat")
                ui.button("确认删除", on_click=lambda: [_do_delete(sid), dialog.close()], color="negative")

    def _do_delete(sid: str):
        try:
            deleted = svc.delete_strategy(sid)
        except ValueError as e:
            ui.notify(str(e), type="negative")
            return
        if not deleted:
            ui.notify(f"策略不存在: {sid}", type="warning")
            return
        current_strategy_id["value"] = ""
        code_editor.value = ""
        status_label.text = f"已删除: {sid}"
        status_label.style("color: #f85149")
        _refresh_list()
        ui.notify(f"策略已删除: {sid}", type="positive")

    def _extract_code(val: str) -> str:
        if not val:
            return ""
        return val.split()[0] if " " in val else val

    async def _run_backtest():
        code = _extract_code(str(code_select.value)).strip()
        if not code:
            ui.notify("请选择或输入ETF代码", type="warning")
            return
        strategy_code = code_editor.value
        if not strategy_code.strip():
            ui.notify("策略代码为空", type="warning")
            return
        run_status.text = f"⏳ 正在回测 {code}..."
        run_status.style("color: #58a6ff")
        _log(f"开始回测: {code}")
        try:
            from etfquant.api.backtest import BacktestService
            bt_svc = BacktestService(config.backtest, config.data, config.ml)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: bt_svc.run_backtest(
                    code=code,
                    strategy_type="custom",
                    strategy_code=strategy_code,
                    initial_capital=capital_input.value,
                    commission_rate=commission_input.value,
                    slippage_rate=slippage_input.value,
                    start_date=start_date_input.value.strip() or None,
                    end_date=end_date_input.value.strip() or None,
                ),
            )
            if result.get("success"):
                ann = result.get("annual_return", 0)
                sharpe = result.get("sharpe_ratio", 0)
                dd = result.get("max_drawdown", 0)
                trades_count = result.get("total_trades", 0)
                run_status.text = f"✅ 年化{ann:.2%} 夏普{sharpe:.4f} 最大回撤{dd:.2%}"
                run_status.style("color: #3fb950")
                ui.notify(f"回测完成: 年化{ann:.2%}", type="positive")
                _log(f"回测完成: 年化{ann:.2%} 夏普{sharpe:.4f} 交易{trades_count}笔")

                record = dict(result)
                record["code"] = code
                record["strategy_name"] = current_strategy_id["value"] or name_input.value.strip() or "自定义策略"
                record["nav_series"] = bt_svc.get_nav_series()
                record["drawdown_series"] = bt_svc.get_drawdown_series()
                record["trades"] = bt_svc.get_trades()
                history.add(record)
                _refresh_history()

                result_container.clear()
                with result_container:
                    ui.separator().classes("q-mb-md")
                    ui.label("📊 回测结果").classes("text-h6 q-mb-md").style("color: #58a6ff")
                    render_metrics(result)
                    render_nav_chart(bt_svc.get_nav_series())
                    render_drawdown_chart(bt_svc.get_drawdown_series())
                    trades = bt_svc.get_trades()
                    if trades:
                        render_trades_table(trades)
            else:
                err = result.get("error", "未知错误")
                run_status.text = f"❌ {err}"
                run_status.style("color: #f85149")
                ui.notify(err, type="negative")
                _log(f"回测失败: {err}")
        except Exception as e:
            run_status.text = f"❌ 异常: {e}"
            run_status.style("color: #f85149")
            _log(f"回测异常: {e}")

    _refresh_list()
    _refresh_history()
    _load_template("ma_cross")
