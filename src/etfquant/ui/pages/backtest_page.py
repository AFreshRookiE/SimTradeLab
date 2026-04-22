from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.api.backtest import BacktestService
from etfquant.api.strategy import StrategyService
from etfquant.ui.charts import render_drawdown_chart, render_metrics, render_nav_chart, render_trades_table


def _prev_trading_day() -> str:
    today = date.today()
    if today.weekday() == 5:
        offset = 1
    elif today.weekday() == 6:
        offset = 2
    elif today.weekday() == 0:
        offset = 3
    else:
        offset = 1
    prev = today - timedelta(days=offset)
    return prev.strftime("%Y%m%d")


def _discover_models(config: ETFQuantConfig) -> dict[str, str]:
    from pathlib import Path
    model_dir = Path(config.ml.save_path)
    options: dict[str, str] = {"": "（无）"}
    if model_dir.exists():
        for f in model_dir.glob("*.ptp"):
            options[str(f)] = f"{f.stem} ({f.stat().st_size / 1024:.0f}KB)"
    return options


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


def create_backtest_page(config: ETFQuantConfig, shared_state: dict[str, Any] | None = None) -> None:
    shared = shared_state or {}
    ui.label("📈 回测执行").classes("text-h4 q-mb-md").style("color: #c9d1d9")
    svc = BacktestService(config.backtest, config.data, config.ml)

    ui.markdown(
        "**使用流程**：① 选择标的与策略 → ② 调整参数 → ③ 点击执行按钮\n\n"
        "三个执行按钮的区别：\n"
        "- **运行回测**：用当前策略和参数，对指定的一支ETF做回测\n"
        "- **端到端训练+回测**：从因子挖掘→模型训练→回测，全自动流程（适合首次使用）\n"
        "- **全市场Top5**：用当前策略扫描全市场ETF，找出表现最好的5支"
    ).style("color: #8b949e; font-size: 13px; line-height: 1.8; margin-bottom: 12px;")

    with ui.card().classes("full-width q-mb-md"):
        with ui.row().classes("full-width items-center"):
            ui.label("① 选择标的与策略").classes("text-h6").style("color: #58a6ff")
            ui.icon("help_outline", size="18px").classes("q-ml-xs cursor-pointer").on("click", lambda: param_help.toggle())
        with ui.row().classes("full-width q-mt-sm"):
            etf_options = _load_etf_options(config)
            etf_labels = [o["label"] for o in etf_options] if etf_options else [config.backtest.benchmark]
            default_label = next((o["label"] for o in etf_options if o["value"] == config.backtest.benchmark), etf_labels[0] if etf_labels else config.backtest.benchmark)
            code_input = ui.select(
                label="ETF代码",
                options=etf_labels,
                value=default_label,
                with_input=True,
                new_value_mode="add",
            ).classes("q-mr-md").style("min-width: 280px")
            strategy_select = ui.select(
                label="策略类型",
                options={
                    "ma_cross": "MA均线交叉",
                    "momentum": "动量策略",
                    "mean_reversion": "均值回归",
                    "etf_premium": "ETF折溢价套利",
                    "ml_model": "ML模型策略",
                    "custom": "📝 自定义策略",
                },
                value="ma_cross",
                on_change=lambda: _on_strategy_change(),
            ).classes("q-mr-md").style("min-width: 160px")
            strategy_svc = StrategyService(config.backtest)
            user_strategies = {s["id"]: s["name"] for s in strategy_svc.list_strategies()}
            custom_strategy_select = ui.select(
                label="选择自定义策略",
                options=user_strategies,
                value=None,
                with_input=False,
            ).classes("q-mr-md").style("min-width: 200px")
            custom_strategy_select.set_visibility(False)
            model_path_input = ui.select(
                label="ML模型（仅ML模型策略需要）",
                options=_discover_models(config),
                value=None,
                with_input=True,
            ).style("min-width: 280px")

    with ui.card().classes("full-width q-mb-md"):
        with ui.row().classes("full-width items-center"):
            ui.label("② 调整参数").classes("text-h6").style("color: #58a6ff")
        ui.label("交易参数").classes("text-subtitle2 q-mt-sm q-mb-xs").style("color: #f0883e")
        with ui.row().classes("full-width"):
            capital_input = ui.number(label="初始资金", value=20000, format="%.0f", step=10000, min=1000).classes("q-mr-md").style("min-width: 140px")
            t_plus_1_switch = ui.switch(text="T+1", value=config.backtest.t_plus_1).classes("q-mr-md")
            commission_input = ui.number(label="手续费率", value=config.backtest.commission_rate, format="%.4f", step=0.0001, min=0.0).classes("q-mr-md").style("min-width: 140px")
            slippage_input = ui.number(label="滑点率", value=config.backtest.slippage_rate, format="%.4f", step=0.0001, min=0.0).classes("q-mr-md").style("min-width: 140px")
        with ui.row().classes("full-width q-mt-sm"):
            start_date_input = ui.input(label="回测起始日", value="20200101", placeholder="YYYYMMDD").classes("q-mr-md").style("min-width: 140px")
            end_date_input = ui.input(label="回测结束日", value=_prev_trading_day(), placeholder="YYYYMMDD").classes("q-mr-md").style("min-width: 140px")

        ui.separator().classes("q-my-md")
        ui.label("策略专属参数").classes("text-subtitle2 q-mb-xs").style("color: #f0883e")
        with ui.row().classes("full-width"):
            ma_short_input = ui.number(label="短期均线天数", value=5, min=2, max=60, step=1).classes("q-mr-md").style("min-width: 160px")
            ma_long_input = ui.number(label="长期均线天数", value=20, min=5, max=120, step=1).classes("q-mr-md").style("min-width: 160px")
            momentum_lookback = ui.number(label="动量回看天数", value=20, min=5, max=60, step=1).classes("q-mr-md").style("min-width: 160px")
        with ui.row().classes("full-width q-mt-sm"):
            mr_lookback = ui.number(label="均值回归回看天数", value=20, min=5, max=60, step=1).classes("q-mr-md").style("min-width: 160px")
            mr_entry_z = ui.number(label="入场Z-score", value=-2.0, step=0.1).classes("q-mr-md").style("min-width: 160px")
            mr_exit_z = ui.number(label="出场Z-score", value=0.0, step=0.1).classes("q-mr-md").style("min-width: 160px")
        with ui.row().classes("full-width q-mt-sm"):
            premium_threshold = ui.number(label="折溢价阈值", value=0.005, format="%.4f", step=0.001, min=0.0).classes("q-mr-md").style("min-width: 160px")

        with ui.expansion("📖 参数说明", icon="help_outline").classes("full-width q-mt-sm").style("border: 1px solid #30363d; border-radius: 8px;") as param_help:
            ui.markdown(
                "**交易参数：**\n\n"
                "| 参数 | 说明 | 建议范围 | 实际可取值范围 |\n"
                "|------|------|----------|----------------|\n"
                "| 初始资金 | 回测起始资金 | 1万~100万 | >=1000 |\n"
                "| T+1 | 是否启用T+1约束 | 建议开启（A股规则） | 开/关 |\n"
                "| 手续费率 | 单边交易佣金费率 | 0.01%~0.05% | 0~1% |\n"
                "| 滑点率 | 模拟成交价偏差 | 0.05%~0.2% | 0~1% |\n"
                "| 回测区间 | 数据截取范围 | 3~5年 | 有数据的任意区间 |\n\n"
                "**策略专属参数：**\n\n"
                "| 参数 | 适用策略 | 说明 | 建议范围 | 实际可取值范围 |\n"
                "|------|----------|------|----------|----------------|\n"
                "| 短期均线天数 | MA均线交叉 | 短期MA周期 | 3~10 | 2~60 |\n"
                "| 长期均线天数 | MA均线交叉 | 长期MA周期 | 15~30 | 5~120 |\n"
                "| 动量回看天数 | 动量策略 | 计算N日收益率 | 10~30 | 5~60 |\n"
                "| 均值回归回看天数 | 均值回归 | 计算均值和标准差的窗口 | 15~30 | 5~60 |\n"
                "| 入场Z-score | 均值回归 | Z低于此值买入 | -2.5~-1.5 | -4.0~0.0 |\n"
                "| 出场Z-score | 均值回归 | Z高于此值卖出 | -0.5~0.5 | -3.0~3.0 |\n"
                "| 折溢价阈值 | ETF折溢价 | 溢价率偏离阈值 | 0.3%~1% | 0~5% |\n\n"
                "**T+1说明**：A股ETF实行T+1交易制度——今天买入的份额明天才能卖出。开启后回测更接近真实交易，但短线策略收益会降低。\n\n"
                "**手续费率说明**：场内ETF基础佣金费率通常为0.03%（万分之3），买卖均收，最低5元。0.03%=0.0003。\n\n"
                "**滑点率说明**：模拟实际成交价与预期价的偏差，包含买卖价差和冲击成本。流动性好的ETF滑点小，流动性差的大。\n\n"
                "**策略逻辑：**\n"
                "- **MA均线交叉**：短期均线上穿长期均线买入，下穿卖出\n"
                "- **动量策略**：N日收益率为正买入，为负卖出\n"
                "- **均值回归**：Z-score低于入场阈值买入，回归至出场阈值卖出\n"
                "- **ETF折溢价**：溢价率低于-阈值买入，高于+阈值卖出\n"
                "- **ML模型策略**：用训练好的机器学习模型预测涨跌\n"
                "- **📝 自定义策略**：加载策略编辑页保存的用户策略代码执行回测"
            ).style("color: #8b949e; font-size: 12px; line-height: 1.6;")

    with ui.card().classes("full-width q-mb-md"):
        ui.label("③ 执行").classes("text-h6 q-mb-sm").style("color: #58a6ff")
        with ui.row().classes("full-width q-mb-sm"):
            ui.button("🚀 运行回测", on_click=lambda: _run(), color="primary").props("size=lg")
            ui.button("🔄 端到端训练+回测", on_click=lambda: _run_pipeline(), color="secondary").props("size=lg")
            ui.button("🏆 全市场Top5", on_click=lambda: _run_top_etfs(), color="accent").props("size=lg")
        with ui.row().classes("full-width items-center"):
            progress = ui.linear_progress(value=0).classes("col q-mr-md")
            run_status = ui.label("").classes("text-body2").style("color: #8b949e")

    pipeline_stages_row = ui.row().classes("q-mb-md q-gutter-xs")
    result_container = ui.column().classes("full-width")

    def _on_strategy_change():
        is_custom = strategy_select.value == "custom"
        is_ml = strategy_select.value == "ml_model"
        custom_strategy_select.set_visibility(is_custom)
        model_path_input.set_visibility(is_ml)
        if is_custom:
            user_strategies = {s["id"]: s["name"] for s in strategy_svc.list_strategies()}
            custom_strategy_select.options = user_strategies
            if user_strategies:
                custom_strategy_select.value = next(iter(user_strategies))

    def _extract_code(val: str) -> str:
        if not val:
            return ""
        return val.split()[0] if " " in val else val

    async def _run():
        code = _extract_code(str(code_input.value)).strip()
        if not code:
            ui.notify("请选择ETF代码", type="warning")
            return

        strategy_code = None
        if strategy_select.value == "custom":
            sid = custom_strategy_select.value
            if not sid:
                ui.notify("请选择自定义策略", type="warning")
                return
            strategy_data = strategy_svc.get_strategy(sid)
            if not strategy_data:
                ui.notify(f"策略不存在: {sid}", type="warning")
                return
            strategy_code = strategy_data["content"]

        progress.value = 0.1
        run_status.text = f"⏳ 正在回测 {code}..."
        run_status.style("color: #58a6ff")
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: svc.run_backtest(
                    code=code,
                    strategy_type=strategy_select.value,
                    model_path=model_path_input.value or None,
                    strategy_code=strategy_code,
                    initial_capital=capital_input.value,
                    t_plus_1=t_plus_1_switch.value,
                    commission_rate=commission_input.value,
                    slippage_rate=slippage_input.value,
                    start_date=start_date_input.value.strip() or None,
                    end_date=end_date_input.value.strip() or None,
                    ma_short=int(ma_short_input.value),
                    ma_long=int(ma_long_input.value),
                    momentum_lookback=int(momentum_lookback.value),
                    mr_lookback=int(mr_lookback.value),
                    mr_entry_z=mr_entry_z.value,
                    mr_exit_z=mr_exit_z.value,
                    premium_threshold=premium_threshold.value,
                )
            )
            progress.value = 1.0
            if result.get("success"):
                shared["last_backtest_svc"] = svc
                shared["last_backtest_result"] = result
                svc.save_backtest_result(result, code, strategy_select.value, model_path_input.value or None)
                run_status.text = f"✅ 回测完成: {code} 年化{result.get('annual_return', 0):.2%} 夏普{result.get('sharpe_ratio', 0):.4f}"
                run_status.style("color: #3fb950")
            else:
                run_status.text = f"❌ 回测失败: {result.get('error', '未知错误')}"
                run_status.style("color: #f85149")
        except Exception as e:
            progress.value = 0
            run_status.text = f"❌ 异常: {e}"
            run_status.style("color: #f85149")
            return

        result_container.clear()
        with result_container:
            if not result.get("success"):
                ui.label(f"❌ 回测失败: {result.get('error', '未知错误')}").style("color: #f85149")
                return
            render_metrics(result)
            render_nav_chart(svc.get_nav_series())
            render_drawdown_chart(svc.get_drawdown_series())
            trades = svc.get_trades()
            if trades:
                with ui.card().classes("full-width q-mb-md"):
                    with ui.row().classes("full-width items-center justify-between"):
                        ui.label("交易明细").classes("text-h6").style("color: #58a6ff")
                        ui.button("💾 导出CSV", on_click=lambda: _export_csv()).props("flat").style("color: #58a6ff")
                    render_trades_table(trades)

    async def _run_pipeline():
        code = _extract_code(str(code_input.value)).strip() or config.backtest.benchmark
        run_status.text = f"⏳ 端到端训练+回测: {code}..."
        run_status.style("color: #58a6ff")
        progress.value = 0.05

        pipeline_stages_row.clear()
        with pipeline_stages_row:
            for lbl in ["数据检查", "因子生成", "模型训练", "回测验证"]:
                ui.label(lbl).classes("stage-badge stage-pending")

        try:
            from etfquant.pipeline.bus import PipelineBus
            bus = PipelineBus(config)
            future = bus.run_full_pipeline([code])
            result = await asyncio.get_event_loop().run_in_executor(None, future.result)

            pipeline_stages_row.clear()
            with pipeline_stages_row:
                stage_labels = ["数据检查", "因子生成", "模型训练", "回测验证"]
                stage_keys = ["data_check", "alpha_generate", "ml_train", "backtest"]
                for lbl, key in zip(stage_labels, stage_keys):
                    s = result.stages.get(key)
                    if s and s.status.value == "completed":
                        ui.label(lbl).classes("stage-badge stage-done")
                    elif s and s.status.value == "failed":
                        ui.label(lbl).classes("stage-badge stage-failed")
                    else:
                        ui.label(lbl).classes("stage-badge stage-pending")

            progress.value = 1.0
            if result.success:
                run_status.text = f"✅ 端到端完成! 耗时{result.total_elapsed:.1f}s, 有效因子{result.alpha_count}个"
                run_status.style("color: #3fb950")
                shared["last_pipeline_result"] = result
                if result.backtest_summary:
                    result_container.clear()
                    with result_container:
                        render_metrics(result.backtest_summary)
            else:
                run_status.text = "❌ 端到端执行失败"
                run_status.style("color: #f85149")
        except Exception as e:
            progress.value = 0
            run_status.text = f"❌ 异常: {e}"
            run_status.style("color: #f85149")

    async def _run_top_etfs():
        model_path = model_path_input.value or None
        strategy = strategy_select.value
        if strategy == "ml_model" and not model_path:
            ui.notify("ML模型策略需要选择模型", type="warning")
            return
        if strategy == "custom" and not custom_strategy_select.value:
            ui.notify("请选择自定义策略", type="warning")
            return

        strategy_code = None
        if strategy == "custom":
            sid = custom_strategy_select.value
            strategy_data = strategy_svc.get_strategy(sid)
            if not strategy_data:
                ui.notify(f"策略不存在: {sid}", type="warning")
                return
            strategy_code = strategy_data["content"]

        progress.value = 0.05
        run_status.text = "⏳ 正在扫描全市场ETF..."
        run_status.style("color: #58a6ff")

        def _scan_all():
            from etfquant.data.bridge import DataBridge
            bridge = DataBridge(config.data)
            etf_list = bridge.list_etf_codes()
            results = []
            for i, code in enumerate(etf_list[:200]):
                try:
                    r = svc.run_backtest(
                        code=code,
                        strategy_type=strategy,
                        model_path=model_path,
                        strategy_code=strategy_code,
                        initial_capital=capital_input.value,
                        t_plus_1=t_plus_1_switch.value,
                        commission_rate=commission_input.value,
                        slippage_rate=slippage_input.value,
                        start_date=start_date_input.value.strip() or None,
                        end_date=end_date_input.value.strip() or None,
                        ma_short=int(ma_short_input.value),
                        ma_long=int(ma_long_input.value),
                        momentum_lookback=int(momentum_lookback.value),
                    )
                    if r.get("success") and r.get("total_trades", 0) > 0:
                        results.append({
                            "code": code,
                            "annual_return": r.get("annual_return", 0),
                            "sharpe_ratio": r.get("sharpe_ratio", 0),
                            "max_drawdown": r.get("max_drawdown", 0),
                            "win_rate": r.get("win_rate", 0),
                            "total_trades": r.get("total_trades", 0),
                            "total_return": r.get("total_return", 0),
                            "calmar_ratio": r.get("calmar_ratio", 0),
                        })
                except Exception:
                    pass
            results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
            return results[:5]

        try:
            top5 = await asyncio.get_event_loop().run_in_executor(None, _scan_all)
            progress.value = 1.0
            if not top5:
                run_status.text = "⚠️ 未找到有效回测结果（可能策略未产生交易信号）"
                run_status.style("color: #d29922")
                return
            run_status.text = f"✅ 扫描完成，找到 {len(top5)} 个表现优秀的ETF"
            run_status.style("color: #3fb950")
            result_container.clear()
            with result_container:
                with ui.card().classes("full-width q-mb-md"):
                    ui.label("🏆 全市场回测Top5 ETF").classes("text-h6 q-mb-md").style("color: #58a6ff")
                    ui.markdown("基于所选策略在全市场ETF中回测，按夏普比率排序取前5名。这些ETF与当前策略最匹配，可作为实盘候选标的。").style("color: #8b949e; font-size: 13px;")
                    ui.table(
                        columns=[
                            {"name": "rank", "label": "排名", "field": "rank", "align": "center", "width": "60px"},
                            {"name": "code", "label": "ETF代码", "field": "code", "align": "left", "sortable": True, "width": "100px"},
                            {"name": "annual_return", "label": "年化收益", "field": "annual_return", "align": "right", "sortable": True, "width": "100px"},
                            {"name": "sharpe_ratio", "label": "夏普比率", "field": "sharpe_ratio", "align": "right", "sortable": True, "width": "100px"},
                            {"name": "max_drawdown", "label": "最大回撤", "field": "max_drawdown", "align": "right", "sortable": True, "width": "100px"},
                            {"name": "win_rate", "label": "胜率", "field": "win_rate", "align": "right", "width": "80px"},
                            {"name": "total_trades", "label": "交易次数", "field": "total_trades", "align": "center", "width": "80px"},
                            {"name": "calmar_ratio", "label": "卡尔玛比率", "field": "calmar_ratio", "align": "right", "width": "100px"},
                        ],
                        rows=[{**r, "rank": i + 1, "annual_return": f"{r['annual_return']:.2%}", "sharpe_ratio": f"{r['sharpe_ratio']:.4f}", "max_drawdown": f"{r['max_drawdown']:.2%}", "win_rate": f"{r['win_rate']:.2%}", "calmar_ratio": f"{r['calmar_ratio']:.4f}"} for i, r in enumerate(top5)],
                        row_key="code",
                    ).classes("full-width").props('separator="cell"')
                    with ui.expansion("💡 实盘操作建议", icon="lightbulb").classes("full-width q-mt-md").style("border: 1px solid #30363d; border-radius: 8px;"):
                        best = top5[0] if top5 else {}
                        ui.markdown(f"""
**基于回测结果的实盘操作建议：**

1. **优先关注Top1 ETF** ({best.get('code', '')})：夏普比率 {best.get('sharpe_ratio', 0):.4f}，年化收益 {best.get('annual_return', 0):.2%}
2. **分散配置**：建议从Top3中各选1支，等权配置降低单一标的风险
3. **关注回撤**：最大回撤超过20%的标的需谨慎，建议设置止损线
4. **交易频率**：交易次数过少（<5次）的策略可能缺乏统计显著性
5. **模型时效**：ML模型策略需定期重新训练，建议每周更新因子和模型

**风险提示**：回测收益不代表未来表现，实盘需考虑滑点、流动性、冲击成本等因素。
""").style("color: #c9d1d9; font-size: 13px; line-height: 1.6;")
        except Exception as e:
            progress.value = 0
            run_status.text = f"❌ 异常: {e}"
            run_status.style("color: #f85149")

    def _export_csv():
        from pathlib import Path
        path = str(Path(config.backtest.save_path) / "backtest_result.csv")
        svc.export_result(path)
        ui.notify(f"已导出: {path}", type="positive")
