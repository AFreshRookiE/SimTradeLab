from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from nicegui import app, ui

from etfquant.core.config import ETFQuantConfig, load_config
from etfquant.ui.pages.data_page import create_data_page
from etfquant.ui.pages.factor_page import create_factor_page
from etfquant.ui.pages.strategy_page import create_strategy_page
from etfquant.ui.pages.backtest_page import create_backtest_page
from etfquant.ui.pages.result_page import create_result_page

_CSS = """
body { background-color: #0d1117 !important; color: #c9d1d9 !important; }
.q-drawer { background-color: #161b22 !important; border-right: 1px solid #30363d !important; }
.q-header { background-color: #161b22 !important; border-bottom: 1px solid #30363d !important; }
.q-footer { background-color: #161b22 !important; border-top: 1px solid #30363d !important; }
.q-card { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; color: #c9d1d9 !important; }
.q-table { background-color: #0d1117 !important; color: #c9d1d9 !important; }
.q-table__container { border: 1px solid #30363d !important; }
.q-table thead th { color: #8b949e !important; border-bottom: 1px solid #30363d !important; }
.q-table tbody td { color: #c9d1d9 !important; border-bottom: 1px solid #21262d !important; }
.q-table__bottom { color: #8b949e !important; border-top: 1px solid #30363d !important; }
.q-field__control { background-color: #0d1117 !important; color: #c9d1d9 !important; }
.q-field__label { color: #8b949e !important; }
.q-field--outlined .q-field__control { border-color: #30363d !important; }
.q-field--outlined .q-field__control:hover { border-color: #58a6ff !important; }
.q-field--focused .q-field__control { border-color: #58a6ff !important; }
.q-tab-panel { background-color: transparent !important; }
.q-tabs { background-color: #161b22 !important; }
.q-tab { color: #8b949e !important; }
.q-tab--active { color: #58a6ff !important; }
.q-separator { background-color: #30363d !important; }
.q-btn { border-radius: 6px !important; }
.q-select__dropdown-icon { color: #8b949e !important; }
.q-menu { background-color: #161b22 !important; border: 1px solid #30363d !important; }
.q-item { color: #c9d1d9 !important; }
.q-item--active { color: #58a6ff !important; background-color: #1f2937 !important; }
.q-splitter__panel { color: #c9d1d9 !important; }
.q-splitter__separator { background-color: #30363d !important; }
.q-linear-progress__track { background-color: #21262d !important; }
.q-linear-progress__bar { background-color: #58a6ff !important; }
.q-input .q-field__native { color: #c9d1d9 !important; }
.q-textarea .q-field__native { color: #c9d1d9 !important; }
.q-select .q-field__native { color: #c9d1d9 !important; }
.q-switch__inner { color: #8b949e !important; }
.q-switch--active .q-switch__inner { color: #58a6ff !important; }
.q-notification { background-color: #161b22 !important; border: 1px solid #30363d !important; color: #c9d1d9 !important; }
.nav-btn { color: #8b949e !important; font-size: 13px !important; padding: 6px 12px !important; border-radius: 6px !important; transition: all 0.2s !important; }
.nav-btn:hover { color: #58a6ff !important; background-color: #1f2937 !important; }
.home-card { background: linear-gradient(135deg, #161b22 0%, #1a2332 100%) !important; border: 1px solid #30363d !important; border-radius: 12px !important; padding: 24px !important; cursor: pointer !important; transition: all 0.3s !important; min-width: 160px !important; }
.home-card:hover { border-color: #58a6ff !important; transform: translateY(-2px); box-shadow: 0 8px 24px rgba(88,166,255,0.15) !important; }
.metric-card { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; padding: 12px 16px !important; }
.metric-card .label { color: #8b949e !important; font-size: 12px !important; }
.metric-card .value { color: #c9d1d9 !important; font-size: 18px !important; font-weight: 600 !important; }
.stage-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 4px; }
.stage-pending { background-color: #21262d; color: #8b949e; }
.stage-running { background-color: #1f3a5f; color: #58a6ff; }
.stage-done { background-color: #1a3a2a; color: #3fb950; }
.stage-failed { background-color: #3d1a1a; color: #f85149; }
"""

_NAV_ITEMS = [
    ("/data", "📊", "数据管理"),
    ("/factor", "🧬", "因子管理"),
    ("/strategy", "📝", "策略编辑"),
    ("/backtest", "📈", "回测执行"),
    ("/result", "📊", "结果展示"),
]

_shared_state: dict[str, Any] = {}


def _create_nav(config: ETFQuantConfig) -> None:
    with ui.header().classes("items-center justify-between q-px-lg q-py-xs").style("height: 48px"):
        with ui.row().classes("items-center"):
            ui.html('<span style="font-size:20px;font-weight:700;color:#58a6ff">ETFQuant</span><span style="font-size:20px;font-weight:300;color:#8b949e">Desk</span>')
            ui.label("|").classes("text-grey-7 q-mx-sm")
            ui.label("ETF专属量化分析平台").classes("text-caption text-grey-6")
        with ui.row().classes("items-center q-gutter-xs"):
            for path, icon, label in _NAV_ITEMS:
                ui.button(f"{icon} {label}", on_click=lambda p=path: ui.navigate.to(p)).classes("nav-btn").props("flat dense no-caps")

    with ui.footer().classes("text-center q-py-xs").style("height: 28px"):
        ui.label("ETFQuantDesk v0.2.0 | ETF Data → Alpha → ML → Backtest").classes("text-caption text-grey-7")


def main() -> None:
    config = load_config(str(Path("config/etfquant.yaml")))
    ui.add_css(_CSS, shared=True)
    if Path("static").exists():
        app.add_static_files("/static", "static")

    @ui.page("/")
    def index():
        _create_nav(config)
        with ui.column().classes("q-pa-xl items-center justify-center").style("min-height: calc(100vh - 76px)"):
            ui.html('<div style="text-align:center;margin-bottom:32px"><h1 style="font-size:36px;font-weight:700;color:#58a6ff;margin:0">🚀 ETFQuantDesk</h1><p style="font-size:16px;color:#8b949e;margin-top:8px">ETF专属量化分析桌面软件</p><p style="font-size:14px;color:#6e7681;margin-top:4px">全流程: 数据 → 因子生成 → 机器学习 → 回测验证</p></div>')

            with ui.row().classes("q-gutter-lg q-mb-xl"):
                for path, icon, label in _NAV_ITEMS:
                    with ui.column().classes("home-card items-center").on("click", lambda p=path: ui.navigate.to(p)):
                        ui.label(icon).style("font-size: 36px")
                        ui.label(label).style("color: #c9d1d9; font-size: 14px; font-weight: 500; margin-top: 8px")

            with ui.card().classes("q-mt-lg").style("max-width: 640px; width: 100%"):
                ui.label("⚡ 一键全流程").classes("text-subtitle1 q-mb-md").style("color: #58a6ff; font-weight: 600")
                ui.label("数据检查 → 因子生成 → 模型训练 → 回测验证，一键完成").classes("text-body2 q-mb-md").style("color: #8b949e")
                with ui.row().classes("full-width items-center"):
                    code_input = ui.input(value=config.backtest.benchmark, placeholder="ETF代码").classes("col q-mr-md").props("dense outlined")
                    ui.button("🚀 一键运行", on_click=lambda: _run_pipeline(code_input.value.strip()), color="primary").props("dense")
                pipeline_status = ui.label("").classes("text-body2 q-mt-sm")
                pipeline_stages = ui.row().classes("q-mt-sm q-gutter-xs")

            async def _run_pipeline(code: str):
                if not code:
                    code = config.backtest.benchmark
                pipeline_status.text = f"⏳ 正在运行全流程: {code}..."
                pipeline_status.style("color: #58a6ff")
                pipeline_stages.clear()
                with pipeline_stages:
                    for s in ["数据检查", "因子生成", "模型训练", "回测验证"]:
                        ui.label(s).classes("stage-badge stage-pending")
                try:
                    from etfquant.pipeline.bus import PipelineBus
                    bus = PipelineBus(config)
                    future = bus.run_full_pipeline([code])
                    result = await asyncio.get_event_loop().run_in_executor(None, future.result)
                    pipeline_stages.clear()
                    with pipeline_stages:
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
                    if result.success:
                        pipeline_status.text = f"✅ 完成! 耗时{result.total_elapsed:.1f}s, 有效因子{result.alpha_count}个"
                        pipeline_status.style("color: #3fb950")
                        _shared_state["last_pipeline_result"] = result
                    else:
                        pipeline_status.text = "❌ 执行失败，请查看日志"
                        pipeline_status.style("color: #f85149")
                except Exception as e:
                    pipeline_status.text = f"❌ 异常: {e}"
                    pipeline_status.style("color: #f85149")

    @ui.page("/data")
    def data_page():
        _create_nav(config)
        with ui.column().classes("q-pa-lg full-width"):
            create_data_page(config)

    @ui.page("/factor")
    def factor_page():
        _create_nav(config)
        with ui.column().classes("q-pa-lg full-width"):
            create_factor_page(config)

    @ui.page("/strategy")
    def strategy_page():
        _create_nav(config)
        with ui.column().classes("q-pa-lg full-width"):
            create_strategy_page(config)

    @ui.page("/backtest")
    def backtest_page():
        _create_nav(config)
        with ui.column().classes("q-pa-lg full-width"):
            create_backtest_page(config, _shared_state)

    @ui.page("/result")
    def result_page():
        _create_nav(config)
        with ui.column().classes("q-pa-lg full-width"):
            create_result_page(config, _shared_state)

    ui.run(
        host=config.ui.host,
        port=config.ui.port,
        title=config.ui.title,
        reload=False,
        dark=True,
    )


if __name__ == "__main__":
    main()
