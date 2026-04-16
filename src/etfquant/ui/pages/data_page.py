from __future__ import annotations

from nicegui import ui

from etfquant.core.config import ETFQuantConfig
from etfquant.api.data import DataService
from etfquant.ui.echart_theme import ECHART_XAXIS, ECHART_YAXIS, KLINE_COLORS


def _calc_ma(data: list[float | None], n: int) -> list[float | None]:
    result: list[float | None] = []
    for i in range(len(data)):
        if i < n - 1:
            result.append(None)
        else:
            window = [v for v in data[i - n + 1:i + 1] if v is not None]
            if len(window) < n:
                result.append(None)
            else:
                result.append(round(sum(window) / n, 4))
    return result


def _calc_ema(data: list[float], n: int) -> list[float]:
    if not data:
        return []
    k = 2 / (n + 1)
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(data[i] * k + result[-1] * (1 - k))
    return result


def _calc_macd(closes: list[float], short: int = 12, long: int = 26, signal: int = 9):
    ema_short = _calc_ema(closes, short)
    ema_long = _calc_ema(closes, long)
    dif = [round(s - l, 4) for s, l in zip(ema_short, ema_long)]
    dea = _calc_ema(dif, signal)
    dea = [round(d, 4) for d in dea]
    macd = [round(2 * (d - e), 4) for d, e in zip(dif, dea)]
    return dif, dea, macd


def create_data_page(config: ETFQuantConfig) -> None:
    ui.label("📊 数据管理").classes("text-h4 q-mb-lg").style("color: #c9d1d9")
    svc = DataService(config.data)

    with ui.column().classes("full-width"):
        with ui.card().classes("full-width q-mb-md").style("background: linear-gradient(135deg, #161b22 0%, #1a2332 100%) !important; border: 1px solid #30363d !important; border-radius: 12px !important;"):
            with ui.row().classes("full-width items-center q-pa-md").style("position: relative"):
                category_select = ui.select(
                    label="分类筛选",
                    options=["全部"] + list(svc._bridge.classification.categories.keys()),
                    value="全部",
                    on_change=lambda: _refresh(),
                ).props("outlined").style("min-width: 180px; margin-right: 16px")
                search_input = ui.input(
                    label="搜索ETF",
                    placeholder="输入代码或名称...",
                ).props("outlined clearable debounce=300").style("min-width: 280px; flex: 1; margin-right: 16px").on(
                    "update:modelValue", lambda e: _on_search_input(e.args if isinstance(e.args, str) else "")
                )
                ui.button("🔍 查询", on_click=lambda: _refresh()).classes("q-mr-md").props("dense")
                ui.button("🔄 一键更新", on_click=lambda: _update_data(), color="secondary").props("dense")
                update_status = ui.label("").classes("text-body2 q-ml-md").style("color: #8b949e; min-width: 200px")

                search_dropdown = ui.column().classes("q-pa-xs").style(
                    "position: absolute; top: 100%; left: 180px; z-index: 9999; "
                    "max-height: 240px; overflow-y: auto; background-color: #161b22; "
                    "border: 1px solid #30363d; border-radius: 6px; min-width: 320px; "
                    "box-shadow: 0 8px 24px rgba(0,0,0,0.6)"
                )

        with ui.row().classes("full-width").style("display: flex; flex-wrap: nowrap; gap: 16px;"):
            with ui.column().style("flex: 0 0 45%; max-width: 45%;"):
                with ui.card().classes("full-width q-mb-md").style("border-radius: 12px !important;"):
                    hot_label = ui.label("✨ 金叉信号 ETF").classes("text-subtitle2 q-pa-sm q-mb-none").style("color: #f0883e")
                    etf_table = ui.table(
                        columns=[
                            {"name": "code", "label": "代码", "field": "code", "sortable": True, "align": "left", "width": "120px"},
                            {"name": "name", "label": "名称", "field": "name", "sortable": True, "align": "left", "width": "200px"},
                            {"name": "signal", "label": "信号", "field": "signal", "sortable": True, "align": "center", "width": "100px"},
                            {"name": "strength", "label": "强度", "field": "strength", "sortable": True, "align": "right", "width": "80px"},
                        ],
                        rows=[],
                        row_key="code",
                        pagination={"rowsPerPage": 5, "rowsPerPageOptions": [5, 10, 20, 50, 100]},
                    ).classes("full-width").on("rowClick", lambda e: _on_row_click(e), [[], ["code"], None])

                with ui.card().classes("full-width").style("border-radius: 12px !important;"):
                    ui.label("ETF 详情").classes("text-h6 q-mb-sm").style("color: #58a6ff")
                    detail_label = ui.label("点击ETF查看详情").classes("text-body1").style("color: #8b949e; line-height: 1.6")

            with ui.column().style("flex: 1 1 55%; min-width: 0;"):
                with ui.card().classes("full-width").style("border-radius: 12px !important;"):
                    with ui.row().classes("full-width items-center q-pb-sm"):
                        ui.label("K线走势").classes("text-h6").style("color: #58a6ff")
                        ui.space()
                        start_date_input = ui.input(value="20200101", placeholder="起始 YYYYMMDD").props(
                            "dense outlined dark"
                        ).style("width: 140px; font-size: 13px; margin-right: 8px")
                        ui.label("~").style("color: #8b949e; font-size: 13px; margin-right: 8px")
                        end_date_input = ui.input(value="20261231", placeholder="结束 YYYYMMDD").props(
                            "dense outlined dark"
                        ).style("width: 140px; font-size: 13px; margin-right: 8px")
                        ui.button("🔍 查询", on_click=lambda: _update_chart(), color="primary").props("dense").style("margin-right: 4px")
                        ui.button("🔄 重置", on_click=lambda: _reset_dates(), color="secondary").props("flat dense round").style("color: #8b949e")
                    chart = ui.echart({}).classes("full-width").style("height: 480px")

        _current_code = {"value": ""}

        def _on_row_click(e):
            if not e.args:
                return
            args = e.args
            if isinstance(args, list) and len(args) >= 2:
                row = args[1]
            elif isinstance(args, dict):
                row = args
            else:
                return
            if isinstance(row, dict):
                code = row.get("code", "")
            elif isinstance(row, str):
                code = row
            else:
                return
            if code:
                _show_etf(code)

        def _show_etf(code: str):
            _current_code["value"] = code
            detail = svc.get_etf_detail(code)
            category = detail['category'] if detail['category'] else "未分类"
            lines = [f"代码: {detail['code']}", f"名称: {detail['name']}", f"分类: {category}"]
            if detail.get("has_daily"):
                lines += [
                    f"数据区间: {detail['start_date']} ~ {detail['end_date']}",
                    f"数据条数: {detail['rows']}",
                    f"最新收盘: {detail['latest_close']:.4f}",
                    f"日均收益率: {detail['avg_daily_return']:.4%}",
                    f"日波动率: {detail['daily_volatility']:.4%}",
                ]
                start_date_input.value = detail["start_date"].replace("-", "")
                end_date_input.value = detail["end_date"].replace("-", "")
            data_sources = []
            if detail.get("has_nav"):
                data_sources.append("净值")
            if detail.get("has_premium"):
                data_sources.append("溢价")
            if detail.get("has_adjust"):
                data_sources.append("复权因子")
            if data_sources:
                lines.append(f"数据源: {', '.join(data_sources)}")
            detail_label.text = "\n".join(lines)
            _update_chart()

        def _on_search_input(query: str):
            search_dropdown.clear()
            if not query:
                return
            matches = svc.search_etfs(query, limit=15)
            if not matches:
                with search_dropdown:
                    ui.label("无匹配结果").classes("text-caption q-pa-sm").style("color: #8b949e")
                return
            with search_dropdown:
                for m in matches:
                    code = m["code"]
                    name = m["name"]
                    label_text = f"{code}  {name}"
                    ui.button(
                        label_text,
                        on_click=lambda c=code: _pick_search(c),
                    ).classes("full-width q-mb-xs").props("flat dense align=left no-caps").style(
                        "color: #c9d1d9; font-size: 13px; text-align: left"
                    )

        def _pick_search(code: str):
            search_input.value = ""
            search_dropdown.clear()
            _show_etf(code)

        def _refresh():
            category = category_select.value
            search_text = search_input.value.strip() if search_input.value else ""
            if search_text:
                hot_label.text = "🔍 搜索结果"
                etfs = svc.list_etfs(category if category != "全部" else None, search_text or None)
                for row in etfs:
                    row["signal"] = ""
                    row["strength"] = ""
            else:
                hot_label.text = "✨ 金叉信号 ETF"
                etfs = svc.get_golden_cross_etfs(top_n=50)
            etf_table.rows = etfs

        def _update_data():
            update_status.text = "⏳ 正在刷新数据..."
            update_status.style("color: #58a6ff")
            result = svc.refresh_data()
            update_status.text = f"✅ {result['message']}"
            update_status.style("color: #3fb950")
            ui.notify(result["message"], type="positive")
            _refresh()

        def _reset_dates():
            code = _current_code["value"]
            if not code:
                return
            detail = svc.get_etf_detail(code)
            if detail.get("has_daily"):
                start_date_input.value = detail["start_date"].replace("-", "")
                end_date_input.value = detail["end_date"].replace("-", "")
                _update_chart()

        def _parse_date(val: str) -> str | None:
            if not val or len(val) < 8:
                return None
            clean = val.strip().replace("-", "").replace("/", "")
            if len(clean) == 8 and clean.isdigit():
                return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"
            return None

        def _update_chart():
            code = _current_code["value"]
            if not code:
                ui.notify("请先选择一只ETF", type="warning")
                return
            raw_start = start_date_input.value or ""
            raw_end = end_date_input.value or ""
            start = _parse_date(raw_start)
            end = _parse_date(raw_end)
            chart_data = svc.get_etf_chart_data(code, start_date=start, end_date=end)
            if not chart_data:
                ui.notify(f"该日期范围内无数据 (输入: {raw_start}~{raw_end})", type="info")
                return

            dates = [d["date"] for d in chart_data]
            closes = [d.get("close", 0) for d in chart_data]
            opens = [d.get("open") for d in chart_data]
            highs = [d.get("high") for d in chart_data]
            lows = [d.get("low") for d in chart_data]

            ohlc_data = []
            for i in range(len(dates)):
                o = opens[i] if opens[i] is not None else closes[i]
                h = highs[i] if highs[i] is not None else closes[i]
                l = lows[i] if lows[i] is not None else closes[i]
                c = closes[i]
                ohlc_data.append([o, c, l, h])

            ma5 = _calc_ma(closes, 5)
            ma10 = _calc_ma(closes, 10)
            ma20 = _calc_ma(closes, 20)
            ma60 = _calc_ma(closes, 60)

            dif, dea, macd = _calc_macd(closes)

            macd_colors = []
            for v in macd:
                if v is not None and v >= 0:
                    macd_colors.append(KLINE_COLORS["macd_up"])
                else:
                    macd_colors.append(KLINE_COLORS["macd_down"])

            chart._props['options'] = {
                "backgroundColor": "transparent",
                "animation": False,
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {"type": "cross", "crossStyle": {"color": "#555"}},
                },
                "legend": {
                    "data": ["K线", "MA5", "MA10", "MA20", "MA60"],
                    "textStyle": {"color": "#8b949e"},
                    "top": 0,
                },
                "grid": [
                    {"left": "4%", "right": "2%", "top": "8%", "height": "55%"},
                    {"left": "4%", "right": "2%", "top": "72%", "height": "18%"},
                ],
                "xAxis": [
                    {"type": "category", "data": dates, "gridIndex": 0, **ECHART_XAXIS, "axisLabel": {"color": "#8b949e", "fontSize": 10}},
                    {"type": "category", "data": dates, "gridIndex": 1, **ECHART_XAXIS, "axisLabel": {"show": False}},
                ],
                "yAxis": [
                    {"type": "value", "gridIndex": 0, "scale": True, **ECHART_YAXIS},
                    {"type": "value", "gridIndex": 1, "scale": True, **ECHART_YAXIS, "axisLabel": {"show": False}, "splitLine": {"show": False}},
                ],
                "dataZoom": [
                    {"type": "inside", "xAxisIndex": [0, 1], "start": 70, "end": 100},
                    {"type": "slider", "xAxisIndex": [0, 1], "bottom": "2%", "height": 16, "borderColor": "#30363d", "backgroundColor": "#161b22", "fillerColor": "rgba(88,166,255,0.15)", "handleStyle": {"color": "#58a6ff"}, "textStyle": {"color": "#8b949e"}},
                ],
                "series": [
                    {
                        "name": "K线",
                        "type": "candlestick",
                        "data": ohlc_data,
                        "xAxisIndex": 0,
                        "yAxisIndex": 0,
                        "itemStyle": {
                            "color": KLINE_COLORS["up"],
                            "color0": KLINE_COLORS["down"],
                            "borderColor": KLINE_COLORS["up"],
                            "borderColor0": KLINE_COLORS["down"],
                        },
                    },
                    {
                        "name": "MA5",
                        "type": "line",
                        "data": ma5,
                        "xAxisIndex": 0,
                        "yAxisIndex": 0,
                        "smooth": True,
                        "showSymbol": False,
                        "lineStyle": {"width": 1, "color": KLINE_COLORS["ma5"]},
                        "itemStyle": {"color": KLINE_COLORS["ma5"]},
                    },
                    {
                        "name": "MA10",
                        "type": "line",
                        "data": ma10,
                        "xAxisIndex": 0,
                        "yAxisIndex": 0,
                        "smooth": True,
                        "showSymbol": False,
                        "lineStyle": {"width": 1, "color": KLINE_COLORS["ma10"]},
                        "itemStyle": {"color": KLINE_COLORS["ma10"]},
                    },
                    {
                        "name": "MA20",
                        "type": "line",
                        "data": ma20,
                        "xAxisIndex": 0,
                        "yAxisIndex": 0,
                        "smooth": True,
                        "showSymbol": False,
                        "lineStyle": {"width": 1, "color": KLINE_COLORS["ma20"]},
                        "itemStyle": {"color": KLINE_COLORS["ma20"]},
                    },
                    {
                        "name": "MA60",
                        "type": "line",
                        "data": ma60,
                        "xAxisIndex": 0,
                        "yAxisIndex": 0,
                        "smooth": True,
                        "showSymbol": False,
                        "lineStyle": {"width": 1, "color": KLINE_COLORS["ma60"]},
                        "itemStyle": {"color": KLINE_COLORS["ma60"]},
                    },
                    {
                        "name": "DIF",
                        "type": "line",
                        "data": dif,
                        "xAxisIndex": 1,
                        "yAxisIndex": 1,
                        "showSymbol": False,
                        "lineStyle": {"width": 1.5, "color": KLINE_COLORS["dif"]},
                        "itemStyle": {"color": KLINE_COLORS["dif"]},
                    },
                    {
                        "name": "DEA",
                        "type": "line",
                        "data": dea,
                        "xAxisIndex": 1,
                        "yAxisIndex": 1,
                        "showSymbol": False,
                        "lineStyle": {"width": 1.5, "color": KLINE_COLORS["dea"]},
                        "itemStyle": {"color": KLINE_COLORS["dea"]},
                    },
                    {
                        "name": "MACD",
                        "type": "bar",
                        "data": macd,
                        "xAxisIndex": 1,
                        "yAxisIndex": 1,
                        "itemStyle": {
                            "color": macd_colors,
                        },
                    },
                ],
            }
            chart.update()

        _refresh()
