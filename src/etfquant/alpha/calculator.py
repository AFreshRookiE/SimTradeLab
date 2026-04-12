from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from etfquant.core.config import AlphaConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = [
    "AlphaCalculator",
    "ETFAlphaCalculator",
    "AlphaFactor",
    "AlphaPool",
    "PresetFactors",
    "calculate_ic",
    "calculate_rank_ic",
]

logger = get_logger("etfquant.alpha")


def calculate_ic(predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 10:
        return 0.0, 1.0
    ic, p_value = stats.pearsonr(predictions[mask], actuals[mask])
    return float(ic), float(p_value)


def calculate_rank_ic(predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 10:
        return 0.0, 1.0
    ric, p_value = stats.spearmanr(predictions[mask], actuals[mask])
    return float(ric), float(p_value)


@dataclass
class AlphaFactor:
    name: str
    expression: str
    description: str = ""
    ic: float = 0.0
    rank_ic: float = 0.0
    icir: float = 0.0
    is_valid: bool = False
    category: str = "custom"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "description": self.description,
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "icir": self.icir,
            "is_valid": self.is_valid,
            "category": self.category,
            "params": self.params,
        }


class AlphaCalculator(ABC):
    @abstractmethod
    def calc_single_IC_ret(self, expr: str) -> float:
        pass

    @abstractmethod
    def calc_single_rIC_ret(self, expr: str) -> float:
        pass

    @abstractmethod
    def calc_single_all_ret(self, expr: str) -> tuple[float, float]:
        pass

    @abstractmethod
    def calc_mutual_IC(self, expr1: str, expr2: str) -> float:
        pass


class ETFAlphaCalculator(AlphaCalculator):
    def __init__(self, data_bridge: DataBridge, config: AlphaConfig) -> None:
        self._bridge = data_bridge
        self._config = config
        self._returns_cache: dict[str, pd.Series] = {}
        self._factor_cache: dict[str, pd.Series] = {}

    def _get_forward_returns(self, code: str) -> pd.Series | None:
        if code in self._returns_cache:
            return self._returns_cache[code]
        df = self._bridge.load_etf_daily(code)
        if df.empty or "close" not in df.columns:
            return None
        ret = df["close"].pct_change(self._config.target_period).shift(-self._config.target_period)
        self._returns_cache[code] = ret
        return ret

    def _evaluate_expression(self, expr: str, code: str) -> pd.Series | None:
        cache_key = f"{code}_{expr}"
        if cache_key in self._factor_cache:
            return self._factor_cache[cache_key]
        df = self._bridge.load_etf_daily(code)
        if df.empty:
            return None
        try:
            result = self._safe_eval(expr, df)
            if result is not None:
                self._factor_cache[cache_key] = result
            return result
        except Exception as exc:
            logger.warning("表达式求值失败: %s, code=%s, error=%s", expr, code, exc)
            return None

    def _safe_eval(self, expr: str, df: pd.DataFrame) -> pd.Series | None:
        close = df.get("close", pd.Series(dtype=float))
        nav = df.get("nav", pd.Series(dtype=float))
        iopv = df.get("iopv", df.get("nav", pd.Series(dtype=float)))
        idx_close = df.get("index_close", pd.Series(dtype=float))

        def _premium_rate():
            if close.empty or nav.empty:
                return pd.Series(0.0, index=df.index)
            return (close - nav) / nav

        def _tracking_error(w=20):
            if close.empty or idx_close.empty:
                return pd.Series(0.0, index=df.index)
            etf_ret = close.pct_change()
            idx_ret = idx_close.pct_change()
            excess = etf_ret - idx_ret
            return excess.rolling(w).std() * (252**0.5)

        def _iopv_deviation():
            if close.empty or iopv.empty:
                return pd.Series(0.0, index=df.index)
            return (close - iopv) / iopv

        namespace: dict[str, Any] = {
            "close": close,
            "open": df.get("open", pd.Series(dtype=float)),
            "high": df.get("high", pd.Series(dtype=float)),
            "low": df.get("low", pd.Series(dtype=float)),
            "volume": df.get("volume", pd.Series(dtype=float)),
            "amount": df.get("amount", pd.Series(dtype=float)),
            "pct_chg": df.get("pct_chg", pd.Series(dtype=float)),
            "preclose": df.get("preclose", pd.Series(dtype=float)),
            "nav": nav,
            "iopv": iopv,
            "index_close": idx_close,
            "np": np,
            "pd": pd,
            "ts_mean": lambda s, w: s.rolling(w).mean(),
            "ts_std": lambda s, w: s.rolling(w).std(),
            "ts_max": lambda s, w: s.rolling(w).max(),
            "ts_min": lambda s, w: s.rolling(w).min(),
            "ts_sum": lambda s, w: s.rolling(w).sum(),
            "ts_rank": lambda s, w: s.rolling(w).rank(pct=True),
            "ts_corr": lambda s1, s2, w: s1.rolling(w).corr(s2),
            "ts_cov": lambda s1, s2, w: s1.rolling(w).cov(s2),
            "ts_delta": lambda s, d: s - s.shift(d),
            "ts_delay": lambda s, d: s.shift(d),
            "ts_return": lambda s, d: s.pct_change(d),
            "premium_rate": _premium_rate,
            "tracking_error": _tracking_error,
            "iopv_deviation": _iopv_deviation,
            "abs": np.abs,
            "log": np.log,
            "sign": np.sign,
            "max": np.maximum,
            "min": np.minimum,
            "power": np.power,
        }
        result = eval(expr, {"__builtins__": {}}, namespace)
        if isinstance(result, (pd.Series, pd.DataFrame)):
            return result
        return None

    def calc_single_IC_ret(self, expr: str) -> float:
        codes = self._bridge.list_etf_codes()[:50]
        ic_values: list[float] = []
        for code in codes:
            factor = self._evaluate_expression(expr, code)
            returns = self._get_forward_returns(code)
            if factor is None or returns is None:
                continue
            merged = pd.concat([factor, returns], axis=1, join="inner").dropna()
            if len(merged) < 30:
                continue
            ic, _ = calculate_ic(merged.iloc[:, 0].values, merged.iloc[:, 1].values)
            ic_values.append(ic)
        return float(np.mean(ic_values)) if ic_values else 0.0

    def calc_single_rIC_ret(self, expr: str) -> float:
        codes = self._bridge.list_etf_codes()[:50]
        ric_values: list[float] = []
        for code in codes:
            factor = self._evaluate_expression(expr, code)
            returns = self._get_forward_returns(code)
            if factor is None or returns is None:
                continue
            merged = pd.concat([factor, returns], axis=1, join="inner").dropna()
            if len(merged) < 30:
                continue
            ric, _ = calculate_rank_ic(merged.iloc[:, 0].values, merged.iloc[:, 1].values)
            ric_values.append(ric)
        return float(np.mean(ric_values)) if ric_values else 0.0

    def calc_single_all_ret(self, expr: str) -> tuple[float, float]:
        return self.calc_single_IC_ret(expr), self.calc_single_rIC_ret(expr)

    def calc_mutual_IC(self, expr1: str, expr2: str) -> float:
        codes = self._bridge.list_etf_codes()[:30]
        ic_values: list[float] = []
        for code in codes:
            f1 = self._evaluate_expression(expr1, code)
            f2 = self._evaluate_expression(expr2, code)
            if f1 is None or f2 is None:
                continue
            merged = pd.concat([f1, f2], axis=1, join="inner").dropna()
            if len(merged) < 30:
                continue
            ic, _ = calculate_ic(merged.iloc[:, 0].values, merged.iloc[:, 1].values)
            ic_values.append(ic)
        return float(np.mean(ic_values)) if ic_values else 0.0


class AlphaPool:
    def __init__(self, config: AlphaConfig) -> None:
        self._config = config
        self._factors: list[AlphaFactor] = []

    @property
    def factors(self) -> list[AlphaFactor]:
        return self._factors

    def add(self, factor: AlphaFactor) -> bool:
        if not factor.is_valid:
            return False
        if len(self._factors) >= self._config.max_factors:
            self._factors.sort(key=lambda f: abs(f.ic), reverse=True)
            self._factors = self._factors[: self._config.max_factors]
        for existing in self._factors:
            if existing.name == factor.name:
                return False
        self._factors.append(factor)
        return True

    def filter_valid(self) -> list[AlphaFactor]:
        return [f for f in self._factors if f.is_valid]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self._factors])

    def save(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        logger.info("因子池已保存: %s (%d 个因子)", path, len(self._factors))

    @classmethod
    def load(cls, path: str, config: AlphaConfig) -> AlphaPool:
        pool = cls(config)
        if not Path(path).exists():
            return pool
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            pool._factors.append(AlphaFactor(**row.to_dict()))
        return pool


class PresetFactors:
    _FACTORS: list[dict[str, str]] = [
        {
            "name": "momentum_20d",
            "expression": "ts_return(close, 20)",
            "description": "20日动量因子",
            "category": "momentum",
        },
        {
            "name": "momentum_5d",
            "expression": "ts_return(close, 5)",
            "description": "5日动量因子",
            "category": "momentum",
        },
        {
            "name": "volatility_20d",
            "expression": "ts_std(ts_return(close, 1), 20)",
            "description": "20日波动率因子",
            "category": "volatility",
        },
        {
            "name": "volatility_ratio",
            "expression": "ts_std(ts_return(close, 1), 5) / ts_std(ts_return(close, 1), 20)",
            "description": "短期/长期波动率比",
            "category": "volatility",
        },
        {
            "name": "volume_ratio_5d",
            "expression": "ts_mean(volume, 5) / ts_mean(volume, 20)",
            "description": "5日/20日量比",
            "category": "volume",
        },
        {
            "name": "turnover_momentum",
            "expression": "ts_corr(ts_return(close, 1), volume, 20)",
            "description": "价量相关性因子",
            "category": "volume",
        },
        {
            "name": "high_low_range_20d",
            "expression": "ts_max(high, 20) - ts_min(low, 20)",
            "description": "20日振幅因子",
            "category": "range",
        },
        {
            "name": "close_to_high_ratio",
            "expression": "(close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20))",
            "description": "收盘价在20日区间中的位置",
            "category": "range",
        },
        {
            "name": "mean_reversion_5d",
            "expression": "-ts_delta(close, 5) / ts_delay(close, 5)",
            "description": "5日均值回归因子",
            "category": "mean_reversion",
        },
        {
            "name": "overnight_gap",
            "expression": "(open - ts_delay(close, 1)) / ts_delay(close, 1)",
            "description": "隔夜跳空因子",
            "category": "microstructure",
        },
        {
            "name": "etf_premium_rate",
            "expression": "premium_rate()",
            "description": "ETF折溢价率因子",
            "category": "etf_specific",
        },
        {
            "name": "etf_tracking_error_20d",
            "expression": "tracking_error(20)",
            "description": "20日跟踪误差因子",
            "category": "etf_specific",
        },
        {
            "name": "etf_iopv_deviation",
            "expression": "iopv_deviation()",
            "description": "IOPV偏离度因子",
            "category": "etf_specific",
        },
        {
            "name": "etf_premium_momentum",
            "expression": "ts_delta(premium_rate(), 5)",
            "description": "折溢价率5日变化",
            "category": "etf_specific",
        },
        {
            "name": "etf_premium_volatility",
            "expression": "ts_std(premium_rate(), 20)",
            "description": "折溢价率20日波动",
            "category": "etf_specific",
        },
    ]

    @classmethod
    def get_all(cls) -> list[AlphaFactor]:
        return [
            AlphaFactor(
                name=f["name"],
                expression=f["expression"],
                description=f["description"],
                category=f["category"],
            )
            for f in cls._FACTORS
        ]

    @classmethod
    def get_by_category(cls, category: str) -> list[AlphaFactor]:
        return [f for f in cls.get_all() if f.category == category]

    @classmethod
    def evaluate_all(cls, calculator: ETFAlphaCalculator, config: AlphaConfig) -> AlphaPool:
        pool = AlphaPool(config)
        factors = cls.get_all()
        total = len(factors)
        for i, factor in enumerate(factors):
            logger.info("评估因子 [%d/%d]: %s", i + 1, total, factor.name)
            ic, ric = calculator.calc_single_all_ret(factor.expression)
            factor.ic = ic
            factor.rank_ic = ric
            factor.icir = ic / (abs(ic) + 1e-8) if abs(ic) > 1e-8 else 0.0
            factor.is_valid = abs(ic) >= config.ic_threshold or abs(ric) >= config.rank_ic_threshold
            status = "有效" if factor.is_valid else "无效"
            logger.info("  IC=%.4f, RankIC=%.4f [%s]", ic, ric, status)
            pool.add(factor)
        return pool
