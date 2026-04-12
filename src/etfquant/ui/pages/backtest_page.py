from __future__ import annotations

from typing import Any

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.api.backtest import BacktestService
from etfquant.ui.charts import render_drawdown_chart, render_metrics, render_nav_chart, render_trades_table


def _discover_models(config: ETFQuantConfig) -> dict[str, str]:
    from pathlib import Path
    model_dir = Path(config.ml.save_path)
    options: dict[str, str] = {"": "（无）"}
    if model_dir.exists():
        for f in model_dir.glob("*.ptp"):
            options[str(f)] = f"{f.stem} ({f.stat().st_size / 1024:.0f}KB)"
    return options


def create_backtest_page(config: ETFQuantConfig, shared_state: dict[str, Any] | None = None) -> None:
    shared = shared_state or {}
    ui.label("📈 回测执行").classes("text-h4 q-mb-md").style("color: #c9d1d9")
    svc = BacktestService(config.backtest, config.data)

    with ui.card().classes("full-width q-mb-md"):
        ui.label("回测参数配置").classes("text-h6 q-mb-md").style("color: #58a6ff")
        with ui.row().classes("full-width"):
            code_input = ui.input(label="ETF代码", value=config.backtest.benchmark).classes("q-mr-md")
            strategy_select = ui.select(
                label="策略类型",
                options={"ma": "MA均线策略", "momentum": "动量策略", "mean_reversion": "均值回归", "ml_model": "ML模型策略"},
                value="ma",
            ).classes("q-mr-md")
            capital_input = ui.number(label="初始资金", value=config.backtest.initial_capital, format="%.0f").classes("q-mr-md")
        with ui.row().classes("full-width q-mt-md"):
            t_plus_1_switch = ui.switch(text="T+1", value=config.backtest.t_plus_1).classes("q-mr-md")
            commission_input = ui.number(label="手续费率", value=config.backtest.commission_rate, format="%.4f").classes("q-mr-md")
            slippage_input = ui.number(label="滑点率", value=config.backtest.slippage_rate, format="%.4f").classes("q-mr-md")
        model_path_input = ui.select(
            label="ML模型",
            options=_discover_models(config),
            value=None,
            with_input=True,
        ).classes("full-width q-mt-md")

    with ui.row().classes("q-mb-md items-center"):
        ui.button("🚀 一键运行", on_click=lambda: _run(), color="primary").props("size=lg")
        ui.button("⚡ 全流程", on_click=lambda: _run_pipeline(), color="secondary").props("size=lg")
        progress = ui.linear_progress(value=0).classes("col q-mx-md")
        run_status = ui.label("").classes("text-body2").style("color: #8b949e")

    pipeline_stages_row = ui.row().classes("q-mb-md q-gutter-xs")
    result_container = ui.column().classes("full-width")

    async def _run():
        progress.value = 0.1
        run_status.text = "⏳ 正在运行回测..."
        run_status.style("color: #58a6ff")
        code = code_input.value.strip()
        if not code:
            ui.notify("请输入ETF代码", type="warning")
            return

        result = svc.run_backtest(
            code=code,
            strategy_type=strategy_select.value,
            model_path=model_path_input.value or None,
            initial_capital=capital_input.value,
            t_plus_1=t_plus_1_switch.value,
            commission_rate=commission_input.value,
            slippage_rate=slippage_input.value,
        )
        progress.value = 1.0
        run_status.text = "✅ 回测完成"
        run_status.style("color: #3fb950")

        if result.get("success"):
            shared["last_backtest_svc"] = svc
            shared["last_backtest_result"] = result

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
        import asyncio
        code = code_input.value.strip() or config.backtest.benchmark
        run_status.text = f"⏳ 全流程运行: {code}..."
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
                run_status.text = f"✅ 全流程完成! 耗时{result.total_elapsed:.1f}s, 有效因子{result.alpha_count}个"
                run_status.style("color: #3fb950")
                shared["last_pipeline_result"] = result
                if result.backtest_summary:
                    result_container.clear()
                    with result_container:
                        render_metrics(result.backtest_summary)
            else:
                run_status.text = "❌ 全流程执行失败"
                run_status.style("color: #f85149")
        except Exception as e:
            progress.value = 0
            run_status.text = f"❌ 异常: {e}"
            run_status.style("color: #f85149")

    def _export_csv():
        from pathlib import Path
        path = str(Path(config.backtest.save_path) / "backtest_result.csv")
        svc.export_result(path)
        ui.notify(f"已导出: {path}", type="positive")
