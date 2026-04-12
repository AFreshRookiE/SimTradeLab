from __future__ import annotations

ECHART_GRID = {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True}
ECHART_XAXIS = {"axisLabel": {"color": "#8b949e"}, "axisLine": {"lineStyle": {"color": "#30363d"}}}
ECHART_YAXIS = {"scale": True, "axisLabel": {"color": "#8b949e"}, "splitLine": {"lineStyle": {"color": "#21262d"}}, "axisLine": {"lineStyle": {"color": "#30363d"}}}
ECHART_YAXIS_PCT = {"axisLabel": {"color": "#8b949e", "formatter": "{value}%"}, "splitLine": {"lineStyle": {"color": "#21262d"}}, "axisLine": {"lineStyle": {"color": "#30363d"}}}

NAV_SERIES = {
    "type": "line",
    "lineStyle": {"color": "#58a6ff", "width": 2},
    "itemStyle": {"color": "#58a6ff"},
    "showSymbol": False,
    "areaStyle": {
        "color": {
            "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
            "colorStops": [
                {"offset": 0, "color": "rgba(88,166,255,0.3)"},
                {"offset": 1, "color": "rgba(88,166,255,0.02)"},
            ],
        }
    },
}

DRAWDOWN_SERIES = {
    "type": "line",
    "lineStyle": {"color": "#f85149", "width": 2},
    "itemStyle": {"color": "#f85149"},
    "showSymbol": False,
    "areaStyle": {
        "color": {
            "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
            "colorStops": [
                {"offset": 0, "color": "rgba(248,81,73,0.3)"},
                {"offset": 1, "color": "rgba(248,81,73,0.02)"},
            ],
        }
    },
}

METRICS_ITEMS = [
    ("策略名称", "strategy_name", None),
    ("回测区间", None, "period"),
    ("初始资金", "initial_capital", "comma"),
    ("期末资金", "final_capital", "comma"),
    ("总收益率", "total_return", "pct_green"),
    ("年化收益率", "annual_return", "pct_green"),
    ("最大回撤", "max_drawdown", "pct_red"),
    ("夏普比率", "sharpe_ratio", "f4"),
    ("索提诺比率", "sortino_ratio", "f4"),
    ("卡尔玛比率", "calmar_ratio", "f4"),
    ("胜率", "win_rate", "pct"),
    ("盈亏比", "profit_loss_ratio", "f4"),
    ("总交易次数", "total_trades", "int"),
]

TRADE_COLUMNS = [
    {"name": "date", "label": "日期", "field": "date", "align": "left"},
    {"name": "code", "label": "代码", "field": "code", "align": "left"},
    {"name": "action", "label": "方向", "field": "action", "align": "center"},
    {"name": "price", "label": "价格", "field": "price"},
    {"name": "shares", "label": "数量", "field": "shares"},
    {"name": "amount", "label": "金额", "field": "amount"},
    {"name": "commission", "label": "手续费", "field": "commission"},
    {"name": "profit", "label": "盈亏", "field": "profit"},
]
