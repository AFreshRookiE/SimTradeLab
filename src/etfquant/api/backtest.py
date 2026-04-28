from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from etfquant.core.config import BacktestConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge
from etfquant.backtest.engine import ETFBacktester, BacktestResult

__all__ = ["BacktestService"]

logger = get_logger("etfquant.api.backtest")

_BACKTEST_HISTORY_FILE = "output/backtest/backtest_history.json"
_TOP5_RESULT_FILE = "output/backtest/top5_result.json"

_STRATEGY_DESC: dict[str, str] = {
    "ma_cross": "双均线交叉策略：短期均线上穿长期均线买入，下穿卖出",
    "momentum": "动量策略：N日收益率为正买入，为负卖出",
    "mean_reversion": "均值回归策略：Z-score低于阈值买入，回归卖出",
    "etf_premium": "ETF折溢价策略：溢价率低于阈值买入，高于阈值卖出",
}

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
    def __init__(self, config: BacktestConfig, data_config: Any, ml_config: Any = None) -> None:
        self._config = config
        self._data_config = data_config
        self._ml_config = ml_config
        self._last_result: BacktestResult | None = None

    def list_strategies(self) -> list[dict[str, str]]:
        return [
            {"id": k, "name": v_name, "description": _STRATEGY_DESC.get(k, "")}
            for k, v_name in [
                ("ma_cross", "MA均线交叉"),
                ("momentum", "动量策略"),
                ("mean_reversion", "均值回归"),
                ("etf_premium", "ETF折溢价套利"),
                ("ml_model", "ML模型策略"),
            ]
        ]

    def get_strategy_template(self, strategy_id: str) -> str:
        return _STRATEGY_TEMPLATES.get(strategy_id, "")

    def run_backtest(
        self,
        code: str,
        strategy_type: str = "ma",
        model_path: str | None = None,
        strategy_code: str | None = None,
        initial_capital: float | None = None,
        t_plus_1: bool | None = None,
        commission_rate: float | None = None,
        slippage_rate: float | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        ma_short: int = 5,
        ma_long: int = 20,
        momentum_lookback: int = 20,
        mr_lookback: int = 20,
        mr_entry_z: float = -2.0,
        mr_exit_z: float = 0.0,
        premium_threshold: float = 0.005,
    ) -> dict[str, Any]:
        import pandas as pd

        config = self._config.model_copy(update={
            "initial_capital": initial_capital if initial_capital is not None else self._config.initial_capital,
            "t_plus_1": t_plus_1 if t_plus_1 is not None else self._config.t_plus_1,
            "commission_rate": commission_rate if commission_rate is not None else self._config.commission_rate,
            "slippage_rate": slippage_rate if slippage_rate is not None else self._config.slippage_rate,
        })

        bridge = DataBridge(self._data_config)
        backtester = ETFBacktester(bridge, config, self._ml_config)
        price_df = bridge.load_etf_daily(code)

        if price_df.empty:
            return {"success": False, "error": f"无数据: {code}"}

        if start_date:
            try:
                sd = pd.Timestamp(start_date)
                price_df = price_df[price_df.index >= sd]
            except Exception:
                pass
        if end_date:
            try:
                ed = pd.Timestamp(end_date)
                price_df = price_df[price_df.index <= ed]
            except Exception:
                pass
        if price_df.empty:
            return {"success": False, "error": f"回测区间内无数据: {code} {start_date}~{end_date}"}

        if strategy_type == "ml_model" and model_path:
            from etfquant.ml.trainer import ModelPackage
            model_pkg = ModelPackage.load(model_path)
            result = backtester.run_model_backtest(model_pkg, code, "ML模型策略")
        elif strategy_type == "custom" and strategy_code:
            result = backtester.run_strategy_backtest(strategy_code, code, "自定义策略")
        else:
            signals = pd.DataFrame(index=price_df.index)
            signals["signal"] = 0
            close = price_df["close"].astype(float)

            if strategy_type in ("ma", "ma_cross"):
                ma_s = close.rolling(ma_short).mean()
                ma_l = close.rolling(ma_long).mean()
                signals.loc[ma_s > ma_l, "signal"] = 1
                signals.loc[ma_s < ma_l, "signal"] = -1
                result = backtester.run_signal_backtest(signals, code, f"MA{ma_short}/MA{ma_long}均线策略")
            elif strategy_type == "momentum":
                ret_n = close.pct_change(momentum_lookback)
                signals.loc[ret_n > 0, "signal"] = 1
                signals.loc[ret_n < 0, "signal"] = -1
                result = backtester.run_signal_backtest(signals, code, f"{momentum_lookback}日动量策略")
            elif strategy_type == "mean_reversion":
                mean = close.rolling(mr_lookback).mean()
                std = close.rolling(mr_lookback).std()
                z = (close - mean) / std
                signals.loc[z < mr_entry_z, "signal"] = 1
                signals.loc[z > mr_exit_z, "signal"] = -1
                result = backtester.run_signal_backtest(signals, code, f"均值回归(Z<{mr_entry_z})策略")
            elif strategy_type == "etf_premium":
                if "premium_rate" in price_df.columns:
                    premium = price_df["premium_rate"].astype(float)
                    signals.loc[premium < -premium_threshold, "signal"] = 1
                    signals.loc[premium > premium_threshold, "signal"] = -1
                    result = backtester.run_signal_backtest(signals, code, f"ETF折溢价(阈值{premium_threshold:.4f})策略")
                else:
                    ma_s = close.rolling(ma_short).mean()
                    ma_l = close.rolling(ma_long).mean()
                    signals.loc[ma_s > ma_l, "signal"] = 1
                    signals.loc[ma_s < ma_l, "signal"] = -1
                    result = backtester.run_signal_backtest(signals, code, f"MA{ma_short}/MA{ma_long}均线策略(折溢价数据不可用)")
            else:
                ma_s = close.rolling(ma_short).mean()
                ma_l = close.rolling(ma_long).mean()
                signals.loc[ma_s > ma_l, "signal"] = 1
                signals.loc[ma_s < ma_l, "signal"] = -1
                result = backtester.run_signal_backtest(signals, code, f"MA{ma_short}/MA{ma_long}均线策略")

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

    def save_backtest_result(self, result: dict[str, Any], code: str, strategy_type: str, model_path: str | None = None) -> str:
        if not result.get("success"):
            return ""
        history_path = Path(_BACKTEST_HISTORY_FILE)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        entry = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S") + f"_{code}",
            "timestamp": datetime.now().isoformat(),
            "code": code,
            "strategy_type": strategy_type,
            "model_name": Path(model_path).stem if model_path else "",
            **{k: v for k, v in result.items() if k != "success"},
        }
        history.append(entry)
        if len(history) > 500:
            history = history[-500:]
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info("回测结果已保存: %s %s", code, strategy_type)
        return entry["id"]

    def get_backtest_history(self, code: str | None = None, strategy_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        history_path = Path(_BACKTEST_HISTORY_FILE)
        if not history_path.exists():
            return []
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return []
        if code:
            history = [h for h in history if h.get("code") == code]
        if strategy_type:
            history = [h for h in history if h.get("strategy_type") == strategy_type]
        return history[-limit:]

    def delete_backtest_history(self, entry_id: str) -> bool:
        history_path = Path(_BACKTEST_HISTORY_FILE)
        if not history_path.exists():
            return False
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return False
        new_history = [h for h in history if h.get("id") != entry_id]
        if len(new_history) == len(history):
            return False
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(new_history, f, ensure_ascii=False, indent=2)
        return True

    def get_backtest_comparison(self, entry_ids: list[str]) -> list[dict[str, Any]]:
        history_path = Path(_BACKTEST_HISTORY_FILE)
        if not history_path.exists():
            return []
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return []
        return [h for h in history if h.get("id") in entry_ids]

    def get_latest_backtest(self) -> dict[str, Any] | None:
        history_path = Path(_BACKTEST_HISTORY_FILE)
        if not history_path.exists():
            return None
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            return history[-1] if history else None
        except Exception:
            return None

    def save_top5_result(self, top5: list[dict[str, Any]], strategy_type: str, strategy_params: dict[str, Any] | None = None) -> None:
        top5_path = Path(_TOP5_RESULT_FILE)
        top5_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.now().isoformat(),
            "strategy_type": strategy_type,
            "strategy_params": strategy_params or {},
            "top5": top5,
        }
        with open(top5_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Top5结果已保存: 策略=%s, Top1=%s", strategy_type, top5[0]["code"] if top5 else "")

    def get_top5_result(self) -> dict[str, Any] | None:
        top5_path = Path(_TOP5_RESULT_FILE)
        if not top5_path.exists():
            return None
        try:
            with open(top5_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
