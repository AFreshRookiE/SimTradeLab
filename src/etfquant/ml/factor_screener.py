from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from etfquant.core.config import FactorScreenConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = ["FactorScreener"]

logger = get_logger("etfquant.ml.screener")


class FactorScreener:
    """因子筛选器，三级漏斗：IC筛选 → ICIR筛选 → 去相关筛选。"""

    def __init__(self, config: FactorScreenConfig, data_bridge: DataBridge | None = None) -> None:
        self._config = config
        self._bridge = data_bridge
        self._value_cache: dict[str, pd.DataFrame | None] = {}

    def screen(self, factors: list[dict[str, Any]], progress_callback: Any | None = None) -> list[dict[str, Any]]:
        ic_filtered = self._filter_by_ic(factors)
        icir_filtered = self._filter_by_icir(ic_filtered)
        decorrelated = self._decorrelate(icir_filtered, progress_callback)
        return decorrelated[: self._config.max_factors]

    def _filter_by_ic(self, factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for f in factors:
            ic = abs(f.get("ic") or 0)
            rank_ic = abs(f.get("rank_ic") or 0)
            if ic >= self._config.ic_threshold or rank_ic >= self._config.ic_threshold:
                result.append(f)
        logger.info("IC筛选: %d/%d 因子通过", len(result), len(factors))
        return result

    def _filter_by_icir(self, factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for f in factors:
            icir = abs(f.get("icir") or 0)
            if icir >= self._config.icir_threshold:
                result.append(f)
        if not result:
            result = sorted(factors, key=lambda f: abs(f.get("icir") or 0), reverse=True)[:10]
        logger.info("ICIR筛选: %d/%d 因子通过", len(result), len(factors))
        return result

    def _decorrelate(self, factors: list[dict[str, Any]], progress_callback: Any | None = None) -> list[dict[str, Any]]:
        if len(factors) <= 1:
            return factors

        use_value_corr = self._bridge is not None
        if use_value_corr:
            logger.info("预计算因子值用于去相关（缓存模式）...")
            self._prefetch_factor_values(factors)

        factors_sorted = sorted(factors, key=lambda f: abs(f.get("ic") or 0), reverse=True)
        selected: list[dict[str, Any]] = [factors_sorted[0]]
        total = len(factors_sorted) - 1

        for i, f in enumerate(factors_sorted[1:]):
            if len(selected) >= self._config.max_factors:
                break
            if self._is_low_correlation(f, selected):
                selected.append(f)
            if progress_callback and total > 0:
                progress_callback(i + 1, total)

        logger.info("去相关筛选: %d/%d 因子入选", len(selected), len(factors))
        return selected

    def _prefetch_factor_values(self, factors: list[dict[str, Any]]) -> None:
        from etfquant.alpha.calculator import ETFAlphaCalculator
        from etfquant.core.config import AlphaConfig

        if not self._bridge:
            return

        cfg = AlphaConfig(max_etf_for_ic=50)
        calc = ETFAlphaCalculator(self._bridge, cfg)
        codes = self._bridge.list_etf_codes()[:cfg.max_etf_for_ic]

        for f in factors:
            expr = f.get("expression", "")
            if expr in self._value_cache:
                continue
            frames: dict[str, pd.Series] = {}
            for code in codes:
                try:
                    vals = calc._evaluate_expression(expr, code)
                    if vals is not None:
                        frames[code] = vals.rename(code)
                except Exception:
                    continue
            self._value_cache[expr] = pd.DataFrame(frames) if frames else None

    def _is_low_correlation(self, candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
        if self._bridge is not None:
            return self._is_low_correlation_by_value(candidate, selected)

        c_ic = candidate.get("ic") or 0
        c_ric = candidate.get("rank_ic") or 0

        for s in selected:
            s_ic = s.get("ic") or 0
            s_ric = s.get("rank_ic") or 0

            ic_sign = (c_ic > 0 and s_ic > 0) or (c_ic < 0 and s_ic < 0)
            ic_r = min(abs(c_ic), abs(s_ic)) / (max(abs(c_ic), abs(s_ic)) + 1e-10)
            ric_r = min(abs(c_ric), abs(s_ric)) / (max(abs(c_ric), abs(s_ric)) + 1e-10)

            if ic_sign and ic_r > self._config.mutual_ic_threshold and ric_r > self._config.mutual_ic_threshold:
                return False

        return True

    def _is_low_correlation_by_value(self, candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
        c_expr = candidate.get("expression", "")
        c_df = self._value_cache.get(c_expr)
        if c_df is None or c_df.empty:
            return True

        for s in selected:
            s_expr = s.get("expression", "")
            s_df = self._value_cache.get(s_expr)
            if s_df is None or s_df.empty:
                continue

            common_cols = c_df.columns.intersection(s_df.columns)
            common_idx = c_df.index.intersection(s_df.index)
            if len(common_cols) < 5 or len(common_idx) < 10:
                continue

            corr_values: list[float] = []
            for date in common_idx:
                c_row = c_df.loc[date, common_cols].dropna()
                s_row = s_df.loc[date, common_cols].dropna()
                common = c_row.index.intersection(s_row.index)
                if len(common) < 5:
                    continue
                c_vals = c_row[common].values
                s_vals = s_row[common].values
                if np.std(c_vals) < 1e-10 or np.std(s_vals) < 1e-10:
                    continue
                corr, _ = stats.pearsonr(c_vals, s_vals)
                corr_values.append(corr)

            if corr_values:
                avg_corr = abs(float(np.mean(corr_values)))
                if avg_corr > self._config.mutual_ic_threshold:
                    return False

        return True

    def get_screening_report(self, factors: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total_factors": len(factors),
            "ic_filtered": len(self._filter_by_ic(factors)),
            "icir_filtered": len(self._filter_by_icir(self._filter_by_ic(factors))),
            "selected": len(selected),
            "ic_threshold": self._config.ic_threshold,
            "icir_threshold": self._config.icir_threshold,
            "mutual_ic_threshold": self._config.mutual_ic_threshold,
            "max_factors": self._config.max_factors,
            "selected_names": [f["name"] for f in selected],
            "selected_avg_ic": float(np.mean([abs(f.get("ic") or 0) for f in selected])) if selected else 0.0,
            "selected_avg_icir": float(np.mean([abs(f.get("icir") or 0) for f in selected])) if selected else 0.0,
        }
