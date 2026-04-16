from __future__ import annotations

from typing import Any

import numpy as np

from etfquant.core.config import FactorScreenConfig
from etfquant.core.logger import get_logger

__all__ = ["FactorScreener"]

logger = get_logger("etfquant.ml.screener")


class FactorScreener:
    """因子筛选器，三级漏斗：IC筛选 → ICIR筛选 → 去相关筛选。

    去相关判定：两个因子IC符号相同且IC值比例 > mutual_ic_threshold 时视为相似，
    只保留IC绝对值更高的因子。
    """
    def __init__(self, config: FactorScreenConfig) -> None:
        self._config = config

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
        c_ric = candidate.get("rank_ic") or 0
        c_icir = candidate.get("icir") or 0

        for s in selected:
            s_ic = s.get("ic") or 0
            s_ric = s.get("rank_ic") or 0

            ic_sign_match = (c_ic > 0 and s_ic > 0) or (c_ic < 0 and s_ic < 0)
            ic_ratio = min(abs(c_ic), abs(s_ic)) / (max(abs(c_ic), abs(s_ic)) + 1e-10)
            ric_ratio = min(abs(c_ric), abs(s_ric)) / (max(abs(c_ric), abs(s_ric)) + 1e-10)

            if ic_sign_match and ic_ratio > self._config.mutual_ic_threshold and ric_ratio > self._config.mutual_ic_threshold:
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
