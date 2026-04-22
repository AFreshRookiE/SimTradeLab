from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etfquant.core.config import BacktestConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = ["ETFBacktester", "BacktestResult", "TradeRecord", "ETFMarketProfile", "StrategyContext"]

logger = get_logger("etfquant.backtest")


@dataclass
class TradeRecord:
    date: str
    code: str
    action: str
    price: float
    shares: int
    amount: float
    commission: float
    profit: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    total_trades: int = 0
    trades: list[TradeRecord] = field(default_factory=list)
    nav_series: pd.Series = field(default_factory=pd.Series)
    benchmark_series: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "策略名称": self.strategy_name,
            "回测区间": f"{self.start_date} ~ {self.end_date}",
            "初始资金": f"{self.initial_capital:,.2f}",
            "期末资金": f"{self.final_capital:,.2f}",
            "总收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annual_return:.2%}",
            "最大回撤": f"{self.max_drawdown:.2%}",
            "夏普比率": f"{self.sharpe_ratio:.4f}",
            "索提诺比率": f"{self.sortino_ratio:.4f}",
            "卡尔玛比率": f"{self.calmar_ratio:.4f}",
            "胜率": f"{self.win_rate:.2%}",
            "盈亏比": f"{self.profit_loss_ratio:.4f}",
            "总交易次数": self.total_trades,
        }


@dataclass
class ETFMarketProfile:
    t_plus_1: bool = False
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    min_trade_unit: int = 100
    price_limit_pct: float = 0.10
    is_cross_border: bool = False
    is_bond: bool = False
    is_commodity: bool = False

    @classmethod
    def from_code(cls, code: str, config: BacktestConfig) -> ETFMarketProfile:
        profile = cls(
            t_plus_1=config.t_plus_1,
            commission_rate=config.commission_rate,
            slippage_rate=config.slippage_rate,
            min_trade_unit=config.min_trade_unit,
            price_limit_pct=config.price_limit_pct,
        )
        if code.startswith("51") and int(code[2:4]) >= 30:
            profile.is_cross_border = True
            profile.t_plus_1 = False
            profile.price_limit_pct = 0.20
        if code.startswith("511"):
            profile.is_bond = True
            profile.t_plus_1 = False
            profile.commission_rate = 0.0001
        if code.startswith("518"):
            profile.is_commodity = True
            profile.t_plus_1 = True
            profile.price_limit_pct = 0.10
        return profile


class ETFBacktester:
    def __init__(self, data_bridge: DataBridge, config: BacktestConfig, ml_config: Any = None) -> None:
        self._bridge = data_bridge
        self._config = config
        self._ml_config = ml_config

    def run_strategy_backtest(
        self,
        strategy_code: str,
        code: str,
        strategy_name: str = "自定义策略",
    ) -> BacktestResult:
        price_df = self._bridge.load_etf_daily(code)
        if price_df.empty:
            logger.error("无法加载ETF数据: %s", code)
            return BacktestResult()

        context = StrategyContext()
        data = _StrategyData(price_df, self._bridge, code)

        safe_ns: dict[str, Any] = {"__builtins__": {}, "order_target_percent": order_target_percent}
        try:
            exec(strategy_code, safe_ns)
        except Exception as exc:
            logger.error("策略代码编译失败: %s", exc)
            return BacktestResult()

        init_fn = safe_ns.get("initialize")
        handle_fn = safe_ns.get("handle_data")

        if handle_fn is None:
            logger.error("策略代码缺少 handle_data 函数")
            return BacktestResult()

        if init_fn is not None:
            try:
                import inspect
                sig = inspect.signature(init_fn)
                n_params = len(sig.parameters)
                if n_params >= 2:
                    init_fn(context, data)
                else:
                    init_fn(context)
            except Exception as exc:
                logger.error("initialize() 执行失败: %s", exc)

        signals = pd.DataFrame(index=price_df.index)
        signals["signal"] = 0

        for i in range(len(price_df)):
            data._set_position(i)
            try:
                handle_fn(context, data)
            except Exception as exc:
                if i < 5:
                    logger.warning("handle_data() 第%d天执行异常: %s", i, exc)
            if context._pending_signal is not None:
                signals.iloc[i, signals.columns.get_loc("signal")] = context._pending_signal
                if context._pending_signal == 1:
                    context.position = 1
                elif context._pending_signal == -1:
                    context.position = 0
                context._pending_signal = None

        return self.run_signal_backtest(signals, code, strategy_name)

    def run_signal_backtest(
        self,
        signals: pd.DataFrame,
        code: str,
        strategy_name: str = "ETF信号策略",
    ) -> BacktestResult:
        price_df = self._bridge.load_etf_daily(code)
        if price_df.empty:
            logger.error("无法加载ETF数据: %s", code)
            return BacktestResult()

        profile = ETFMarketProfile.from_code(code, self._config)
        merged = pd.merge(
            signals,
            price_df[["open", "close", "high", "low"]],
            left_index=True,
            right_index=True,
            how="inner",
        )
        if merged.empty:
            return BacktestResult()

        capital = self._config.initial_capital
        position = 0
        nav_list: list[float] = []
        trades: list[TradeRecord] = []
        dates: list[str] = []

        for date, row in merged.iterrows():
            signal = row.get("signal", 0)
            open_price = float(row["open"])
            close_price = float(row["close"])

            if signal > 0 and position == 0:
                exec_price = open_price * (1 + profile.slippage_rate)
                shares = int(capital * 0.95 / (exec_price * profile.min_trade_unit)) * profile.min_trade_unit
                if shares > 0:
                    amount = shares * exec_price
                    commission = amount * profile.commission_rate
                    capital -= amount + commission
                    position = shares
                    trades.append(TradeRecord(
                        date=str(date), code=code, action="BUY",
                        price=exec_price, shares=shares, amount=amount, commission=commission,
                    ))

            elif signal < 0 and position > 0:
                exec_price = open_price * (1 - profile.slippage_rate)
                amount = position * exec_price
                commission = amount * profile.commission_rate
                profit = amount - position * (trades[-1].price if trades else exec_price)
                capital += amount - commission
                trades.append(TradeRecord(
                    date=str(date), code=code, action="SELL",
                    price=exec_price, shares=position, amount=amount, commission=commission, profit=profit,
                ))
                position = 0

            current_nav = capital + position * close_price
            nav_list.append(current_nav)
            dates.append(str(date))

        if position > 0:
            last_price = float(merged.iloc[-1]["close"])
            capital += position * last_price
            position = 0

        nav_series = pd.Series(nav_list, index=pd.to_datetime(dates), name="nav")
        return self._calc_metrics(nav_series, trades, strategy_name, merged, code)

    def run_model_backtest(
        self,
        model_package: Any,
        code: str,
        strategy_name: str = "ETF模型策略",
        buy_threshold: float = 0.0,
        sell_threshold: float = 0.0,
    ) -> BacktestResult:
        from etfquant.ml.trainer import FeatureEngineer, ETFDataSource

        ml_cfg = self._ml_config
        if ml_cfg is None:
            from etfquant.core.config import MLConfig
            ml_cfg = MLConfig()

        ds = ETFDataSource(self._bridge, ml_cfg)
        fe = FeatureEngineer(ds, ml_cfg, factor_expressions=model_package.factor_expressions)
        features = fe.build_features(code)
        if features is None or features.empty:
            return BacktestResult()

        feature_cols = model_package.feature_names
        X = features[feature_cols].dropna()
        predictions = model_package.predict(X)

        signals = pd.DataFrame(index=X.index)
        signals["signal"] = 0
        signals.loc[predictions > buy_threshold, "signal"] = 1
        signals.loc[predictions < sell_threshold, "signal"] = -1

        return self.run_signal_backtest(signals, code, strategy_name)

    def _calc_metrics(
        self,
        nav_series: pd.Series,
        trades: list[TradeRecord],
        strategy_name: str,
        price_df: pd.DataFrame,
        code: str,
    ) -> BacktestResult:
        if nav_series.empty:
            return BacktestResult()

        initial = self._config.initial_capital
        final = nav_series.iloc[-1]
        total_return = (final - initial) / initial

        daily_returns = nav_series.pct_change().dropna()
        trading_days = len(nav_series)
        annual_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

        running_max = nav_series.cummax()
        drawdown = (nav_series - running_max) / running_max
        max_drawdown = drawdown.min()

        sharpe = 0.0
        sortino = 0.0
        if daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * (252**0.5)
            downside = daily_returns[daily_returns < 0]
            downside_std = downside.std() if len(downside) > 0 else daily_returns.std()
            sortino = daily_returns.mean() / downside_std * (252**0.5)

        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        win_trades = [t for t in trades if t.action == "SELL" and t.profit > 0]
        lose_trades = [t for t in trades if t.action == "SELL" and t.profit <= 0]
        sell_trades = [t for t in trades if t.action == "SELL"]
        win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0.0
        avg_win = np.mean([t.profit for t in win_trades]) if win_trades else 0.0
        avg_lose = abs(np.mean([t.profit for t in lose_trades])) if lose_trades else 1.0
        pl_ratio = avg_win / avg_lose if avg_lose > 0 else 0.0

        benchmark_df = self._bridge.load_etf_daily(self._config.benchmark)
        benchmark_series = pd.Series(dtype=float)
        if not benchmark_df.empty and "close" in benchmark_df.columns:
            bm = benchmark_df["close"].reindex(nav_series.index, method="ffill").dropna()
            if not bm.empty:
                benchmark_series = bm / bm.iloc[0] * initial

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=str(nav_series.index[0]),
            end_date=str(nav_series.index[-1]),
            initial_capital=initial,
            final_capital=final,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_loss_ratio=pl_ratio,
            total_trades=len(sell_trades),
            trades=trades,
            nav_series=nav_series,
            benchmark_series=benchmark_series,
            drawdown_series=drawdown,
        )


class StrategyContext:
    """策略上下文，供用户策略代码中的 initialize/handle_data 使用。"""

    def __init__(self) -> None:
        self.position: int = 0
        self._pending_signal: int | None = None
        self._params: dict[str, Any] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name == "position":
            object.__setattr__(self, name, value)
        else:
            self._params[name] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._params[name]
        except KeyError:
            raise AttributeError(f"策略上下文无属性: {name}")


class _StrategyData:
    """策略数据接口，提供 price/NAV 等数据给 handle_data。

    支持 data.close.iloc[-1] 获取当前日期的值（自动按 _pos 切片）。
    """

    def __init__(self, price_df: pd.DataFrame, bridge: DataBridge, code: str) -> None:
        self._price_df = price_df
        self._bridge = bridge
        self._code = code
        self._pos = 0
        self._close = price_df.get("close", pd.Series(dtype=float))
        self._open = price_df.get("open", pd.Series(dtype=float))
        self._high = price_df.get("high", pd.Series(dtype=float))
        self._low = price_df.get("low", pd.Series(dtype=float))
        self._volume = price_df.get("volume", pd.Series(dtype=float))
        self._amount = price_df.get("amount", pd.Series(dtype=float))

        nav_df = bridge.load_etf_nav(code)
        nav_col = None
        for c in ("nav", "单位净值", "nav_per_unit"):
            if c in nav_df.columns:
                nav_col = c
                break
        if nav_col is not None:
            self._nav = nav_df[nav_col].astype(float).reindex(price_df.index, method="ffill")
        else:
            self._nav = pd.Series(0.0, index=price_df.index)

        premium_df = bridge.load_etf_premium(code)
        if not premium_df.empty and "premium_rate" in premium_df.columns:
            self._premium_rate = premium_df["premium_rate"].astype(float).reindex(price_df.index, method="ffill")
        else:
            close = self._close.astype(float)
            nav = self._nav
            self._premium_rate = (close - nav) / nav.replace(0, np.nan)

        self._update_slices()

    def _update_slices(self) -> None:
        """按当前位置切片，让 data.close.iloc[-1] 返回当前日期的值。"""
        end = self._pos + 1
        self.close = self._close.iloc[:end]
        self.open = self._open.iloc[:end]
        self.high = self._high.iloc[:end]
        self.low = self._low.iloc[:end]
        self.volume = self._volume.iloc[:end]
        self.amount = self._amount.iloc[:end]
        self.nav = self._nav.iloc[:end]
        self.premium_rate = self._premium_rate.iloc[:end]

    def _set_position(self, idx: int) -> None:
        self._pos = idx
        self._update_slices()

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item) and isinstance(getattr(self, item, None), pd.Series)


def order_target_percent(context: StrategyContext, percent: float) -> None:
    if percent > 0:
        context._pending_signal = 1
    elif percent == 0 and context.position > 0:
        context._pending_signal = -1
    else:
        context._pending_signal = None
