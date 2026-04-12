from __future__ import annotations

from pathlib import Path
from typing import Any

from etfquant.core.config import BacktestConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge
from etfquant.backtest.engine import ETFBacktester, BacktestResult

__all__ = ["BacktestService"]

logger = get_logger("etfquant.api.backtest")

_STRATEGY_TEMPLATES: dict[str, str] = {
    "ma_cross": '''def initialize(context):
    context.fast_period = 5
    context.slow_period = 20

def handle_data(context, data):
    fast_ma = data.close.rolling(context.fast_period).mean()
    slow_ma = data.close.rolling(context.slow_period).mean()
    if fast_ma.iloc[-1] > slow_ma.iloc[-1] and context.position == 0:
        order_target_percent(context, 0.95)
    elif fast_ma.iloc[-1] < slow_ma.iloc[-1] and context.position > 0:
        order_target_percent(context, 0)
''',
    "momentum": '''def initialize(context):
    context.lookback = 20

def handle_data(context, data):
    ret = data.close.pct_change(context.lookback).iloc[-1]
    if ret > 0 and context.position == 0:
        order_target_percent(context, 0.95)
    elif ret < 0 and context.position > 0:
        order_target_percent(context, 0)
''',
    "mean_reversion": '''def initialize(context):
    context.lookback = 20
    context.entry_z = -2.0
    context.exit_z = 0.0

def handle_data(context, data):
    mean = data.close.rolling(context.lookback).mean()
    std = data.close.rolling(context.lookback).std()
    z = (data.close.iloc[-1] - mean.iloc[-1]) / std.iloc[-1]
    if z < context.entry_z and context.position == 0:
        order_target_percent(context, 0.95)
    elif z > context.exit_z and context.position > 0:
        order_target_percent(context, 0)
''',
    "etf_premium": '''def initialize(context):
    context.premium_threshold = 0.005

def handle_data(context, data):
    premium = data.premium_rate.iloc[-1] if "premium_rate" in data else 0
    if premium < -context.premium_threshold and context.position == 0:
        order_target_percent(context, 0.95)
    elif premium > context.premium_threshold and context.position > 0:
        order_target_percent(context, 0)
''',
}


class BacktestService:
    def __init__(self, config: BacktestConfig, data_config: Any) -> None:
        self._config = config
        self._data_config = data_config
        self._last_result: BacktestResult | None = None

    def list_strategies(self) -> list[dict[str, str]]:
        return [
            {"id": k, "name": v_name, "description": _STRATEGY_DESC.get(k, "")}
            for k, v_name in [
                ("ma_cross", "MA均线交叉"),
                ("momentum", "动量策略"),
                ("mean_reversion", "均值回归"),
                ("etf_premium", "ETF折溢价套利"),
            ]
        ]

    def get_strategy_template(self, strategy_id: str) -> str:
        return _STRATEGY_TEMPLATES.get(strategy_id, "")

    def run_backtest(
        self,
        code: str,
        strategy_type: str = "ma",
        model_path: str | None = None,
        initial_capital: float | None = None,
        t_plus_1: bool | None = None,
        commission_rate: float | None = None,
        slippage_rate: float | None = None,
    ) -> dict[str, Any]:
        import pandas as pd

        config = self._config.model_copy(update={
            "initial_capital": initial_capital or self._config.initial_capital,
            "t_plus_1": t_plus_1 if t_plus_1 is not None else self._config.t_plus_1,
            "commission_rate": commission_rate or self._config.commission_rate,
            "slippage_rate": slippage_rate or self._config.slippage_rate,
        })

        bridge = DataBridge(self._data_config)
        backtester = ETFBacktester(bridge, config)
        price_df = bridge.load_etf_daily(code)

        if price_df.empty:
            return {"success": False, "error": f"无数据: {code}"}

        if strategy_type == "ml_model" and model_path:
            from etfquant.ml.trainer import ModelPackage
            model_pkg = ModelPackage.load(model_path)
            result = backtester.run_model_backtest(model_pkg, code, "ML模型策略")
        else:
            signals = pd.DataFrame(index=price_df.index)
            signals["signal"] = 0
            close = price_df["close"].astype(float)

            if strategy_type == "ma":
                ma_short = close.rolling(5).mean()
                ma_long = close.rolling(20).mean()
                signals.loc[ma_short > ma_long, "signal"] = 1
                signals.loc[ma_short < ma_long, "signal"] = -1
                result = backtester.run_signal_backtest(signals, code, "MA5/MA20均线策略")
            elif strategy_type == "momentum":
                ret_20 = close.pct_change(20)
                signals.loc[ret_20 > 0, "signal"] = 1
                signals.loc[ret_20 < 0, "signal"] = -1
                result = backtester.run_signal_backtest(signals, code, "20日动量策略")
            elif strategy_type == "mean_reversion":
                mean = close.rolling(20).mean()
                std = close.rolling(20).std()
                z = (close - mean) / std
                signals.loc[z < -2, "signal"] = 1
                signals.loc[z > 0, "signal"] = -1
                result = backtester.run_signal_backtest(signals, code, "均值回归策略")
            else:
                ma_short = close.rolling(5).mean()
                ma_long = close.rolling(20).mean()
                signals.loc[ma_short > ma_long, "signal"] = 1
                signals.loc[ma_short < ma_long, "signal"] = -1
                result = backtester.run_signal_backtest(signals, code, "MA5/MA20均线策略")

        self._last_result = result
        return self._result_to_dict(result)

    def get_last_result(self) -> dict[str, Any] | None:
        if self._last_result is None:
            return None
        return self._result_to_dict(self._last_result)

    def get_nav_series(self) -> list[dict[str, Any]]:
        if self._last_result is None or self._last_result.nav_series.empty:
            return []
        return [
            {"date": str(d.date()), "nav": float(v)}
            for d, v in self._last_result.nav_series.items()
        ]

    def get_drawdown_series(self) -> list[dict[str, Any]]:
        if self._last_result is None or self._last_result.drawdown_series.empty:
            return []
        return [
            {"date": str(d.date()), "drawdown": float(v)}
            for d, v in self._last_result.drawdown_series.items()
        ]

    def get_trades(self) -> list[dict[str, Any]]:
        if self._last_result is None:
            return []
        return [
            {
                "date": t.date, "code": t.code, "action": t.action,
                "price": t.price, "shares": t.shares,
                "amount": t.amount, "commission": t.commission, "profit": t.profit,
            }
            for t in self._last_result.trades
        ]

    def export_result(self, path: str) -> str:
        import pandas as pd
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._last_result is None:
            return ""
        trades = self.get_trades()
        if trades:
            df = pd.DataFrame(trades)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def _result_to_dict(self, result: BacktestResult) -> dict[str, Any]:
        return {
            "success": True,
            "strategy_name": result.strategy_name,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "win_rate": result.win_rate,
            "profit_loss_ratio": result.profit_loss_ratio,
            "total_trades": result.total_trades,
            "nav_count": len(result.nav_series),
        }


_STRATEGY_DESC: dict[str, str] = {
    "ma_cross": "双均线交叉策略：短期均线上穿长期均线买入，下穿卖出",
    "momentum": "动量策略：N日收益率为正买入，为负卖出",
    "mean_reversion": "均值回归策略：Z-score低于阈值买入，回归卖出",
    "etf_premium": "ETF折溢价策略：溢价率低于阈值买入，高于阈值卖出",
}
