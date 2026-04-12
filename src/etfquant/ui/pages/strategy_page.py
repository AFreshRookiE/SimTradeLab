from __future__ import annotations

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.api.strategy import StrategyService


def create_strategy_page(config: ETFQuantConfig) -> None:
    ui.label("📝 策略编辑").classes("text-h4 q-mb-md").style("color: #c9d1d9")
    svc = StrategyService()

    with ui.splitter(value=22).classes("full-width").style("height: calc(100vh - 140px)") as splitter:
        with splitter.before:
            ui.label("策略列表").classes("text-subtitle1 q-mb-sm").style("color: #58a6ff")
            strategy_list = ui.column().classes("full-width")

            ui.separator().classes("q-my-md")
            ui.label("模板策略").classes("text-subtitle1 q-mb-sm").style("color: #58a6ff")
            template_list = ui.column().classes("full-width")

        with splitter.after:
            with ui.column().classes("full-width q-pa-md").style("height: 100%"):
                with ui.row().classes("full-width items-center justify-between q-mb-sm"):
                    name_input = ui.input(value="my_strategy", placeholder="策略名称").props("dense outlined").classes("q-mr-md").style("max-width: 250px")
                    ui.button("💾 保存", on_click=lambda: _save(), color="primary").props("flat dense").style("color: #58a6ff")
                    ui.button("🗑 删除", on_click=lambda: _delete(), color="negative").props("flat dense").style("color: #f85149")
                    status_label = ui.label("").classes("text-body2 q-ml-md").style("color: #8b949e")

                code_editor = ui.codemirror(
                    value="",
                    language="Python",
                    theme="githubDark",
                    line_wrapping=True,
                ).classes("full-width").style("flex: 1; min-height: 500px")

    current_strategy_id = {"value": ""}

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
        svc.save_strategy(sid, code_editor.value)
        current_strategy_id["value"] = sid
        status_label.text = f"已保存: {sid}"
        status_label.style("color: #3fb950")
        _refresh_list()
        ui.notify(f"策略已保存: {sid}", type="positive")

    def _delete():
        sid = current_strategy_id["value"] or name_input.value.strip()
        if not sid:
            return
        svc.delete_strategy(sid)
        current_strategy_id["value"] = ""
        code_editor.value = ""
        status_label.text = f"已删除: {sid}"
        status_label.style("color: #f85149")
        _refresh_list()
        ui.notify(f"策略已删除: {sid}", type="positive")

    _refresh_list()
    _load_template("ma_cross")
