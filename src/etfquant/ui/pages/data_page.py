from __future__ import annotations

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.api.data import DataService
from etfquant.ui.echart_theme import ECHART_GRID, ECHART_XAXIS, ECHART_YAXIS, NAV_SERIES


def create_data_page(config: ETFQuantConfig) -> None:
    ui.label("📊 数据管理").classes("text-h4 q-mb-md").style("color: #c9d1d9")
    svc = DataService(config.data)

    with ui.tabs().classes("full-width") as tabs:
        ui.tab("etf_list", label="ETF列表")
        ui.tab("coverage", label="数据覆盖范围")

    with ui.tab_panels(tabs, value="etf_list").classes("full-width"):
        with ui.tab_panel("etf_list"):
            with ui.row().classes("full-width q-mb-md items-end"):
                category_select = ui.select(
                    label="分类筛选",
                    options=["全部"] + list(svc._bridge.classification.categories.keys()),
                    value="全部",
                ).classes("q-mr-md")
                search_input = ui.input(label="搜索", placeholder="代码或名称...").classes("q-mr-md")
                ui.button("🔍 查询", on_click=lambda: _refresh()).classes("q-mr-md")
                ui.button("🔄 一键更新", on_click=lambda: _update_data(), color="secondary")
                update_status = ui.label("").classes("text-body2 q-ml-md").style("color: #8b949e")

            etf_table = ui.table(
                columns=[
                    {"name": "code", "label": "代码", "field": "code", "sortable": True, "align": "left"},
                    {"name": "name", "label": "名称", "field": "name", "sortable": True, "align": "left"},
                    {"name": "category", "label": "分类", "field": "category", "sortable": True, "align": "center"},
                    {"name": "tracking_index", "label": "跟踪指数", "field": "tracking_index", "align": "left"},
                ],
                rows=[],
                row_key="code",
                pagination={"rowsPerPage": 25},
            ).classes("full-width")

            detail_card = ui.card().classes("full-width q-mt-md")
            with detail_card:
                ui.label("ETF 详情").classes("text-h6 q-mb-sm").style("color: #58a6ff")
                detail_label = ui.label("请选择ETF查看详情").classes("text-body1").style("color: #8b949e")

            chart_card = ui.card().classes("full-width q-mt-md")
            with chart_card:
                ui.label("K线走势").classes("text-h6 q-mb-sm").style("color: #58a6ff")
                chart = ui.echart({}).classes("full-width")

            def _refresh():
                category = category_select.value
                search = search_input.value.strip()
                etfs = svc.list_etfs(category if category != "全部" else None, search or None)
                etf_table.rows = etfs

            def _update_data():
                update_status.text = "⏳ 正在刷新数据..."
                update_status.style("color: #58a6ff")
                result = svc.refresh_data()
                update_status.text = f"✅ {result['message']}"
                update_status.style("color: #3fb950")
                ui.notify(result["message"], type="positive")
                _refresh()

            async def _on_row_select(e):
                if not e.args:
                    return
                code = e.args.get("code", "")
                if not code:
                    return
                detail = svc.get_etf_detail(code)
                lines = [f"代码: {detail['code']}", f"名称: {detail['name']}", f"分类: {detail['category']}"]
                if detail.get("has_daily"):
                    lines += [
                        f"数据区间: {detail['start_date']} ~ {detail['end_date']}",
                        f"数据条数: {detail['rows']}",
                        f"最新收盘: {detail['latest_close']:.4f}",
                        f"日均收益率: {detail['avg_daily_return']:.4%}",
                        f"日波动率: {detail['daily_volatility']:.4%}",
                    ]
                detail_label.text = "\n".join(lines)

                chart_data = svc.get_etf_chart_data(code)
                if chart_data:
                    dates = [d["date"] for d in chart_data]
                    closes = [d.get("close", 0) for d in chart_data]
                    series = {**NAV_SERIES, "data": closes}
                    chart._props['options'] = {
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "axis"},
                        "grid": ECHART_GRID,
                        "xAxis": {"type": "category", "data": dates, **ECHART_XAXIS},
                        "yAxis": {"type": "value", **ECHART_YAXIS},
                        "series": [series],
                    }
                    chart.update()

            etf_table.on("rowClick", _on_row_select)
            _refresh()

        with ui.tab_panel("coverage"):
            coverage = svc.get_coverage()
            ui.label(f"ETF总数: {coverage['total_etf_count']}").classes("text-h6 q-mb-md").style("color: #58a6ff")

            with ui.row().classes("full-width q-mb-lg"):
                for cat, count in coverage["categories"].items():
                    with ui.card().classes("q-mr-md q-mb-md metric-card"):
                        ui.label(cat).classes("text-subtitle2").style("color: #8b949e")
                        ui.label(str(count)).classes("text-h4").style("color: #58a6ff; font-weight: 600")

            ui.label("数据样例").classes("text-subtitle1 q-mb-sm").style("color: #58a6ff")
            ui.table(
                columns=[
                    {"name": "code", "label": "代码", "field": "code", "align": "left"},
                    {"name": "start", "label": "起始日期", "field": "start", "align": "left"},
                    {"name": "end", "label": "结束日期", "field": "end", "align": "left"},
                    {"name": "rows", "label": "数据条数", "field": "rows", "align": "right"},
                ],
                rows=coverage["sample_details"],
            ).classes("full-width")
