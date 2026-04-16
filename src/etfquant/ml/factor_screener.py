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
    """因子筛选器，三级漏斗：IC筛选 → ICIR筛选 → 去相关筛选。

    去相关判定：计算两个因子在截面上的相关系数，若 |corr| > mutual_ic_threshold
    则视为相似因子，只保留IC绝对值更高的。这比用IC值比例更科学——
    IC值比例只能判断"两个因子的IC大小是否接近"，而相关系数能判断
    "两个因子是否在说同一件事"。
    """
    def __init__(self, config: FactorScreenConfig, data_bridge: DataBridge | None = None) -> None:
        self._config = config
        self._bridge = data_bridge

    def screen(self, factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ic_filtered = self._filter_by_ic(factors)
        icir_filtered = self._filter_by_icir(ic_filtered)
        decorrelated = self._decorrelate(icir_filtered)
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

    def _decorrelate(self, factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(factors) <= 1:
            return factors

        factors_sorted = sorted(factors, key=lambda f: abs(f.get("ic") or 0), reverse=True)
        selected: list[dict[str, Any]] = [factors_sorted[0]]

        for f in factors_sorted[1:]:
            if len(selected) >= self._config.max_factors:
                break
            if self._is_low_correlation(f, selected):
                selected.append(f)

        logger.info("去相关筛选: %d/%d 因子入选", len(selected), len(factors))
        return selected

    def _is_low_correlation(self, candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
        c_ic = candidate.get("ic") or 0
        s_ic_first = selected[0].get("ic") or 0

        ic_sign_match = (c_ic > 0 and s_ic_first > 0) or (c_ic < 0 and s_ic_first < 0)
        ic_ratio = min(abs(c_ic), abs(s_ic_first)) / (max(abs(c_ic), abs(s_ic_first)) + 1e-10)

        if self._bridge is not None:
            return self._is_low_correlation_by_value(candidate, selected)

        for s in selected:
            s_ic = s.get("ic") or 0
            s_ric = s.get("rank_ic") or 0
            c_ric = candidate.get("rank_ic") or 0

            ic_sign = (c_ic > 0 and s_ic > 0) or (c_ic < 0 and s_ic < 0)
            ic_r = min(abs(c_ic), abs(s_ic)) / (max(abs(c_ic), abs(s_ic)) + 1e-10)
            ric_r = min(abs(c_ric), abs(s_ric)) / (max(abs(c_ric), abs(s_ric)) + 1e-10)

            if ic_sign and ic_r > self._config.mutual_ic_threshold and ric_r > self._config.mutual_ic_threshold:
                return False

        return True

    def _is_low_correlation_by_value(self, candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
        """用因子值计算截面相关系数判定相似性（更科学）。"""
        from etfquant.alpha.calculator import ETFAlphaCalculator
        from etfquant.core.config import AlphaConfig

        if not self._bridge:
            return True

        cfg = AlphaConfig(max_etf_for_ic=200)
        calc = ETFAlphaCalculator(self._bridge, cfg)
        codes = self._bridge.list_etf_codes()[:cfg.max_etf_for_ic]

        c_expr = candidate.get("expression", "")
        c_factor_frames: dict[str, pd.Series] = {}
        for code in codes:
            try:
                vals = calc._evaluate_expression(c_expr, code)
                if vals is not None:
                    c_factor_frames[code] = vals.rename(code)
            except Exception:
                continue
        if not c_factor_frames:
            return True
        c_df = pd.DataFrame(c_factor_frames)

        for s in selected:
            s_expr = s.get("expression", "")
            s_factor_frames: dict[str, pd.Series] = {}
            for code in codes:
                try:
                    vals = calc._evaluate_expression(s_expr, code)
                    if vals is not None:
                        s_factor_frames[code] = vals.rename(code)
                except Exception:
                    continue
            if not s_factor_frames:
                continue
            s_df = pd.DataFrame(s_factor_frames)

            common_cols = c_df.columns.intersection(s_df.columns)
            common_idx = c_df.index.intersection(s_df.index)
            if len(common_cols) < 10 or len(common_idx) < 20:
                continue

            corr_values: list[float] = []
            for date in common_idx:
                c_row = c_df.loc[date, common_cols].dropna()
                s_row = s_df.loc[date, common_cols].dropna()
                common = c_row.index.intersection(s_row.index)
                if len(common) < 10:
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
