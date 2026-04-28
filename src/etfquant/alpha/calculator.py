from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from etfquant.core.config import AlphaConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = [
    "AlphaCalculator",
    "ETFAlphaCalculator",
    "LRUCache",
    "AlphaFactor",
    "AlphaPool",
    "PresetFactors",
    "calculate_ic",
    "calculate_rank_ic",
]

logger = get_logger("etfquant.alpha")


def calculate_ic(predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    """计算Pearson IC（信息系数）和p值。"""
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 10:
        return 0.0, 1.0
    p, a = predictions[mask], actuals[mask]
    if np.std(p) < 1e-10 or np.std(a) < 1e-10:
        return 0.0, 1.0
    ic, p_value = stats.pearsonr(p, a)
    return float(ic), float(p_value)


def calculate_rank_ic(predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    """计算Spearman Rank IC（秩信息系数）和p值。"""
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 10:
        return 0.0, 1.0
    p, a = predictions[mask], actuals[mask]
    if np.std(p) < 1e-10 or np.std(a) < 1e-10:
        return 0.0, 1.0
    ric, p_value = stats.spearmanr(p, a)
    return float(ric), float(p_value)


class LRUCache:
    def __init__(self, max_size: int = 128) -> None:
        self._cache: OrderedDict[str, pd.Series] = OrderedDict()
        self._max_size = max(1, max_size)
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> pd.Series | None:
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: pd.Series) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)


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
    def calc_single_IC_ret(self, expr: str) -> tuple[float, float, int]:
        pass

    @abstractmethod
    def calc_single_rIC_ret(self, expr: str) -> tuple[float, float, int]:
        pass

    @abstractmethod
    def calc_single_all_ret(self, expr: str) -> tuple[float, float, float, float, float, float]:
        """计算因子的截面IC、RankIC、ICIR和统计显著性p值。

        Returns:
            (ic_mean, ic_std, ric_mean, ric_std, icir, ic_p)
            - ic_mean: 截面IC均值
            - ic_std: 截面IC标准差
            - ric_mean: 截面RankIC均值
            - ric_std: 截面RankIC标准差
            - icir: IC均值/IC标准差，衡量因子稳定性
            - ic_p: IC的t检验p值（H0: IC=0），p<0.05表示统计显著
        """
        pass

    @abstractmethod
    def calc_mutual_IC(self, expr1: str, expr2: str) -> float:
        pass


class ETFAlphaCalculator(AlphaCalculator):
    """ETF因子计算器，使用截面IC方法评估因子有效性。

    截面IC：在每个时间截面上，计算所有ETF的因子值与未来收益的横截面相关系数，
    然后对所有时间截面的IC取均值。这是量化研究中的标准做法。
    """
    def __init__(self, data_bridge: DataBridge, config: AlphaConfig) -> None:
        self._bridge = data_bridge
        self._config = config
        self._returns_cache = LRUCache(max_size=config.cache_size)
        self._factor_cache = LRUCache(max_size=config.cache_size * 2)

    def _get_forward_returns(self, code: str) -> pd.Series | None:
        cached = self._returns_cache.get(code)
        if cached is not None:
            return cached
        df = self._bridge.load_etf_daily(code)
        if df.empty or "close" not in df.columns:
            return None
        ret = df["close"].pct_change(self._config.target_period).shift(-self._config.target_period)
        self._returns_cache.put(code, ret)
        return ret

    def _evaluate_expression(self, expr: str, code: str) -> pd.Series | None:
        cache_key = f"{code}_{expr}"
        cached = self._factor_cache.get(cache_key)
        if cached is not None:
            return cached
        df = self._bridge.load_etf_daily(code)
        if df.empty:
            return None
        try:
            result = self._safe_eval(expr, df)
            if result is not None:
                self._factor_cache.put(cache_key, result)
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
            "ts_median": lambda s, w: s.rolling(w).median(),
            "ts_argmax": lambda s, w: s.rolling(w).apply(np.argmax, raw=True),
            "ts_argmin": lambda s, w: s.rolling(w).apply(np.argmin, raw=True),
            "ts_skewness": lambda s, w: s.rolling(w).skew(),
            "ts_kurtosis": lambda s, w: s.rolling(w).kurt(),
            "ts_moment": lambda s, w, k: s.rolling(w).apply(lambda x: np.mean((x - np.mean(x))**k), raw=True),
            "ts_decay_linear": lambda s, w: s.rolling(w).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True),
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

    def _collect_cross_section(self, expr: str, max_etf: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        codes = self._bridge.list_etf_codes()[:max_etf]
        factor_frames: dict[str, pd.Series] = {}
        return_frames: dict[str, pd.Series] = {}
        for code in codes:
            factor = self._evaluate_expression(expr, code)
            returns = self._get_forward_returns(code)
            if factor is None or returns is None:
                continue
            factor = factor.rename(code)
            returns = returns.rename(code)
            factor_frames[code] = factor
            return_frames[code] = returns
        if not factor_frames:
            return pd.DataFrame(), pd.DataFrame()
        factor_df = pd.DataFrame(factor_frames)
        return_df = pd.DataFrame(return_frames)
        common_idx = factor_df.index.intersection(return_df.index)
        return factor_df.loc[common_idx], return_df.loc[common_idx]

    def calc_single_IC_ret(self, expr: str) -> tuple[float, float, int]:
        factor_df, return_df = self._collect_cross_section(expr, self._config.max_etf_for_ic)
        if factor_df.empty:
            return 0.0, 1.0, 0
        ic_values: list[float] = []
        for date in factor_df.index:
            f_row = factor_df.loc[date].dropna()
            r_row = return_df.loc[date].dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < 10:
                continue
            f_vals = f_row[common].values
            r_vals = r_row[common].values
            if np.std(f_vals) < 1e-10 or np.std(r_vals) < 1e-10:
                continue
            ic, _ = stats.pearsonr(f_vals, r_vals)
            ic_values.append(ic)
        if ic_values:
            return float(np.mean(ic_values)), float(np.std(ic_values)), len(ic_values)
        return 0.0, 1.0, 0

    def calc_single_rIC_ret(self, expr: str) -> tuple[float, float, int]:
        factor_df, return_df = self._collect_cross_section(expr, self._config.max_etf_for_ic)
        if factor_df.empty:
            return 0.0, 1.0, 0
        ric_values: list[float] = []
        for date in factor_df.index:
            f_row = factor_df.loc[date].dropna()
            r_row = return_df.loc[date].dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < 10:
                continue
            f_vals = f_row[common].values
            r_vals = r_row[common].values
            if np.std(f_vals) < 1e-10 or np.std(r_vals) < 1e-10:
                continue
            ric, _ = stats.spearmanr(f_vals, r_vals)
            ric_values.append(ric)
        if ric_values:
            return float(np.mean(ric_values)), float(np.std(ric_values)), len(ric_values)
        return 0.0, 1.0, 0

    def calc_single_all_ret(self, expr: str) -> tuple[float, float, float, float, float, float]:
        ic_mean, ic_std, n_ic = self.calc_single_IC_ret(expr)
        ric_mean, ric_std, _ = self.calc_single_rIC_ret(expr)
        icir = ic_mean / (ic_std + 1e-8) if ic_std > 1e-8 else 0.0
        ic_p = 1.0
        if n_ic > 2 and ic_std > 1e-8:
            ic_t = abs(ic_mean / ic_std) * (n_ic ** 0.5)
            ic_p = 2.0 * (1.0 - stats.t.cdf(ic_t, df=n_ic - 1))
        return ic_mean, ic_std, ric_mean, ric_std, icir, ic_p

    def calc_mutual_IC(self, expr1: str, expr2: str) -> float:
        codes = self._bridge.list_etf_codes()[:self._config.max_etf_for_mutual_ic]
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

    def clear_cache(self) -> None:
        self._returns_cache.clear()
        self._factor_cache.clear()
        logger.info("缓存已清空")

    def get_cache_stats(self) -> dict[str, dict[str, Any]]:
        return {
            "returns_cache": {
                "size": self._returns_cache.size,
                "max_size": self._returns_cache._max_size,
                "hit_rate": self._returns_cache.hit_rate,
                "hits": self._returns_cache._hits,
                "misses": self._returns_cache._misses,
            },
            "factor_cache": {
                "size": self._factor_cache.size,
                "max_size": self._factor_cache._max_size,
                "hit_rate": self._factor_cache.hit_rate,
                "hits": self._factor_cache._hits,
                "misses": self._factor_cache._misses,
            },
        }


class AlphaPool:
    def __init__(self, config: AlphaConfig) -> None:
        self._config = config
        self._factors: list[AlphaFactor] = []

    @property
    def factors(self) -> list[AlphaFactor]:
        return self._factors

    def add(self, factor: AlphaFactor) -> bool:
        for existing in self._factors:
            if existing.name == factor.name:
                return False
        self._factors.append(factor)
        if len(self._factors) > self._config.max_factors:
            valid = [f for f in self._factors if f.is_valid]
            invalid = [f for f in self._factors if not f.is_valid]
            valid.sort(key=lambda f: abs(f.ic), reverse=True)
            self._factors = valid[: self._config.max_factors] + invalid
        return True

    def filter_valid(self) -> list[AlphaFactor]:
        return [f for f in self._factors if f.is_valid]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self._factors])

    def save(self, path: str) -> None:
        import json
        df = self.to_dataframe()
        df["params"] = df["params"].apply(lambda p: json.dumps(p, ensure_ascii=False) if isinstance(p, dict) else str(p))
        df.to_parquet(path, index=False)
        logger.info("因子池已保存: %s (%d 个因子)", path, len(self._factors))

    @classmethod
    def load(cls, path: str, config: AlphaConfig) -> AlphaPool:
        import json
        pool = cls(config)
        if not Path(path).exists():
            return pool
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            d = row.to_dict()
            if isinstance(d.get("params"), str):
                try:
                    d["params"] = json.loads(d["params"])
                except (json.JSONDecodeError, TypeError):
                    d["params"] = {}
            pool._factors.append(AlphaFactor(**d))
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
    def evaluate_all(cls, calculator: ETFAlphaCalculator, config: AlphaConfig,
                     progress_callback: Callable[[int, int, str], None] | None = None) -> AlphaPool:
        pool = AlphaPool(config)
        factors = cls.get_all()
        total = len(factors)
        for i, factor in enumerate(factors):
            logger.info("评估因子 [%d/%d]: %s", i + 1, total, factor.name)
            ic_mean, ic_std, ric_mean, ric_std, icir, ic_p = calculator.calc_single_all_ret(factor.expression)
            factor.ic = ic_mean
            factor.rank_ic = ric_mean
            factor.icir = icir
            factor.is_valid = (abs(ic_mean) >= config.ic_threshold or abs(ric_mean) >= config.rank_ic_threshold) and abs(icir) >= config.icir_threshold and ic_p <= 0.05
            status = "有效" if factor.is_valid else "无效"
            logger.info("  IC=%.4f, RankIC=%.4f, ICIR=%.4f, p=%.4f [%s]", ic_mean, ric_mean, icir, ic_p, status)
            pool.add(factor)
            if progress_callback:
                progress_callback(i + 1, total, factor.name)
        return pool
