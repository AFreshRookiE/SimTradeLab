from __future__ import annotations

import math
import random
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from etfquant.alpha.calculator import ETFAlphaCalculator
from etfquant.core.config import AlphaConfig
from etfquant.core.logger import get_logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = ["FactorMiner", "MiningConfig", "MiningResult"]

logger = get_logger("etfquant.alpha.miner")


@dataclass
class MiningConfig:
    """因子挖掘配置参数。"""
    n_steps: int = 2048
    batch_size: int = 64
    timeout_minutes: float = 30.0
    max_factors: int = 50
    strategy: str = "UCB1搜索"
    ic_threshold: float = 0.03
    rank_ic_threshold: float = 0.03
    icir_threshold: float = 0.5
    target_period: int = 20
    max_etf_for_ic: int = 200
    max_etf_for_mutual_ic: int = 100
    seed: int = 42


@dataclass
class MiningResult:
    """因子挖掘结果。"""
    discovered: list[dict[str, Any]] = field(default_factory=list)
    total_evaluated: int = 0
    total_valid: int = 0
    total_skipped: int = 0
    elapsed_seconds: float = 0.0
    best_ic: float = 0.0
    best_expression: str = ""


_TERMINALS = [
    "close", "open", "high", "low", "volume", "amount", "pct_chg",
    "nav", "iopv", "index_close",
]

_UNARY_OPS = [
    ("ts_mean({x}, {w})", "ts_mean"),
    ("ts_std({x}, {w})", "ts_std"),
    ("ts_rank({x}, {w})", "ts_rank"),
    ("ts_sum({x}, {w})", "ts_sum"),
    ("ts_max({x}, {w})", "ts_max"),
    ("ts_min({x}, {w})", "ts_min"),
    ("abs({x})", "abs"),
    ("log(abs({x}) + 1e-8)", "log"),
    ("sign({x})", "sign"),
    ("ts_delta({x}, {d})", "ts_delta"),
    ("ts_return({x}, {d})", "ts_return"),
    ("ts_delay({x}, {d})", "ts_delay"),
    ("1 / ({x} + 1e-8)", "inverse"),
    ("ts_argmax({x}, {w})", "ts_argmax"),
    ("ts_argmin({x}, {w})", "ts_argmin"),
    ("ts_skewness({x}, {w})", "ts_skewness"),
    ("ts_kurtosis({x}, {w})", "ts_kurtosis"),
    ("ts_median({x}, {w})", "ts_median"),
    ("ts_moment({x}, {w}, 2)", "ts_moment2"),
    ("ts_decay_linear({x}, {w})", "ts_decay_linear"),
]

_BINARY_OPS = [
    ("({a} + {b})", "+"),
    ("({a} - {b})", "-"),
    ("({a} * {b})", "*"),
    ("({a} / ({b} + 1e-8))", "/"),
    ("ts_corr({a}, {b}, {w})", "ts_corr"),
    ("ts_cov({a}, {b}, {w})", "ts_cov"),
    ("max({a}, {b})", "max"),
    ("min({a}, {b})", "min"),
]

_ETF_UNARY_OPS = [
    ("premium_rate()", "premium_rate"),
    ("iopv_deviation()", "iopv_deviation"),
    ("tracking_error(20)", "tracking_error"),
    ("tracking_error(10)", "tracking_error_10"),
    ("tracking_error(5)", "tracking_error_5"),
]

_WINDOWS = [5, 10, 20, 60]
_DELTAS = [1, 5, 10, 20]

_HIGH_VALUE_TEMPLATES = [
    "ts_rank({x}, {w1}) * sign({y})",
    "ts_delta({x}, {d}) / (ts_std({x}, {w1}) + 1e-8)",
    "ts_rank(ts_delta({x}, {d}), {w1})",
    "ts_corr({x}, {y}, {w1})",
    "{x} / (ts_mean({x}, {w1}) + 1e-8) - 1",
    "ts_rank({x}, {w1}) - ts_rank({x}, {w2})",
    "sign(ts_delta({x}, {d})) * abs({y})",
    "ts_rank({x}, {w1}) * ts_rank({y}, {w2})",
    "ts_delta({x}, {d}) * sign(premium_rate())",
    "ts_std({x}, {w1}) / (ts_mean({x}, {w1}) + 1e-8)",
    "ts_decay_linear({x}, {w1}) - ts_delay({x}, {d})",
    "ts_skewness({x}, {w1}) * sign(ts_delta({x}, {d}))",
    "ts_corr({x}, volume, {w1})",
    "ts_rank({x}, {w1}) / (ts_std({x}, {w2}) + 1e-8)",
    "ts_delta({x}, {d}) / (abs({x}) + 1e-8)",
]

_OP_NAMES = [t[1] for t in _UNARY_OPS] + [t[1] for t in _ETF_UNARY_OPS] + [t[1] for t in _BINARY_OPS]

_WINDOW_OPS = {"ts_mean", "ts_std", "ts_rank", "ts_sum", "ts_max", "ts_min",
               "ts_argmax", "ts_argmin", "ts_skewness", "ts_kurtosis",
               "ts_median", "ts_moment2", "ts_decay_linear",
               "ts_corr", "ts_cov"}
_DELTA_OPS = {"ts_delta", "ts_return", "ts_delay"}


class ExpressionGenerator:
    """因子表达式生成器：UCB1引导 + n-gram条件概率 + 高价值模板 + 语法树交叉。

    UCB1: 用置信上界自动平衡探索与利用，替代硬编码的30%/70%比例。
    n-gram: 跟踪算子转移概率 P(当前算子有效 | 前一个算子)，实现上下文感知选择。
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._op_success: dict[str, int] = {}
        self._op_attempts: dict[str, int] = {}
        self._total_attempts: int = 0
        self._trans_success: dict[str, dict[str, int]] = {}
        self._trans_attempts: dict[str, dict[str, int]] = {}

    def record_result(self, expr: str, ic: float, is_valid: bool) -> None:
        ops = self._extract_ops(expr)
        self._total_attempts += 1
        for op in ops:
            self._op_attempts[op] = self._op_attempts.get(op, 0) + 1
            if is_valid:
                self._op_success[op] = self._op_success.get(op, 0) + 1
        for i in range(len(ops) - 1):
            prev, curr = ops[i], ops[i + 1]
            if prev not in self._trans_attempts:
                self._trans_attempts[prev] = {}
                self._trans_success[prev] = {}
            self._trans_attempts[prev][curr] = self._trans_attempts[prev].get(curr, 0) + 1
            if is_valid:
                self._trans_success[prev][curr] = self._trans_success[prev].get(curr, 0) + 1

    def _extract_ops(self, expr: str) -> list[str]:
        ops: list[str] = []
        for _, name in _UNARY_OPS:
            if name in expr:
                ops.append(name)
        for _, name in _ETF_UNARY_OPS:
            base = name.split("_")[0]
            if base in expr:
                ops.append(name)
        for t in _TERMINALS:
            if t in expr:
                ops.append(f"term_{t}")
        return ops

    def _ucb1_score(self, op: str) -> float:
        attempts = self._op_attempts.get(op, 0)
        if attempts == 0:
            return float("inf")
        success = self._op_success.get(op, 0)
        return success / attempts + math.sqrt(2 * math.log(self._total_attempts + 1) / attempts)

    def _select_op(self, prev_op: str | None = None, candidates: list[str] | None = None) -> str:
        if candidates is None:
            candidates = list(_OP_NAMES)

        if prev_op and prev_op in self._trans_success and self._total_attempts > 20:
            trans_s = self._trans_success[prev_op]
            trans_a = self._trans_attempts.get(prev_op, {})
            scores: dict[str, float] = {}
            for op in candidates:
                a = trans_a.get(op, 0)
                s = trans_s.get(op, 0)
                if a == 0:
                    scores[op] = self._ucb1_score(op)
                else:
                    scores[op] = s / a + math.sqrt(2 * math.log(self._total_attempts + 1) / a)
            return self._boltzmann_select(scores)

        scores = {op: self._ucb1_score(op) for op in candidates}
        return self._boltzmann_select(scores)

    def _boltzmann_select(self, scores: dict[str, float]) -> str:
        sorted_ops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = sorted_ops[:5]
        total_score = sum(max(s, 0.01) for _, s in top)
        r = self._rng.random() * total_score
        cum = 0.0
        for op, score in top:
            cum += max(score, 0.01)
            if r <= cum:
                return op
        return top[0][0]

    def random_terminal(self) -> str:
        return self._rng.choice(_TERMINALS)

    def random_window(self) -> int:
        return self._rng.choice(_WINDOWS)

    def random_delta(self) -> int:
        return self._rng.choice(_DELTAS)

    def random_etf_op(self) -> str:
        template, _ = self._rng.choice(_ETF_UNARY_OPS)
        return template

    def _apply_op_by_name(self, op_name: str, expr: str) -> str:
        for template, name in _UNARY_OPS:
            if name == op_name:
                if name in _WINDOW_OPS:
                    return template.format(x=expr, w=self.random_window())
                elif name in _DELTA_OPS:
                    return template.format(x=expr, d=self.random_delta())
                else:
                    return template.format(x=expr)
        for template, name in _ETF_UNARY_OPS:
            if name == op_name:
                return template
        return expr

    def generate_simple(self, prev_op: str | None = None) -> str:
        use_guided = self._op_success and self._rng.random() < 0.6
        if use_guided:
            term_op = self._select_op(prev_op, [f"term_{t}" for t in _TERMINALS])
            terminal = term_op.replace("term_", "")
            expr = terminal
            last_op = term_op
        else:
            base = self._rng.choice([self.random_terminal, self.random_etf_op])
            expr = base()
            last_op = None

        depth = self._rng.randint(0, 3)
        for _ in range(depth):
            op = self._select_op(last_op, [t[1] for t in _UNARY_OPS])
            expr = self._apply_op_by_name(op, expr)
            last_op = op
        return expr

    def generate_composite(self, prev_op: str | None = None) -> str:
        left = self.generate_simple(prev_op)
        right_fn = self._rng.choice([self.generate_simple, self.random_etf_op])
        right = right_fn()
        template, bin_name = self._rng.choice(_BINARY_OPS)
        expr = self._format_binary(template, left, right)
        if self._rng.random() < 0.3:
            op = self._select_op(bin_name, [t[1] for t in _UNARY_OPS])
            expr = self._apply_op_by_name(op, expr)
        return expr

    def generate_from_template(self) -> str:
        template = self._rng.choice(_HIGH_VALUE_TEMPLATES)
        x = self._rng.choice(_TERMINALS)
        y = self._rng.choice(_TERMINALS)
        w1 = self.random_window()
        w2 = self.random_window()
        while w2 == w1:
            w2 = self.random_window()
        d = self.random_delta()
        return template.format(x=x, y=y, w1=w1, w2=w2, d=d)

    def generate(self) -> str:
        r = self._rng.random()
        if r < 0.2:
            return self.generate_simple()
        elif r < 0.5:
            return self.generate_composite()
        elif r < 0.7:
            return self.generate_from_template()
        else:
            if self._op_success:
                return self.generate_simple(prev_op=None)
            return self.generate_composite()

    def mutate(self, expr: str) -> str:
        choice = self._rng.random()
        if choice < 0.25:
            op = self._select_op(None, [t[1] for t in _UNARY_OPS])
            return self._apply_op_by_name(op, expr)
        elif choice < 0.45:
            other = self._rng.choice([self.generate_simple, self.random_etf_op])
            template, _ = self._rng.choice(_BINARY_OPS)
            return self._format_binary(template, expr, other())
        elif choice < 0.65:
            for w in _WINDOWS:
                if f", {w})" in expr:
                    new_w = self.random_window()
                    while new_w == w:
                        new_w = self.random_window()
                    expr = expr.replace(f", {w})", f", {new_w})")
                    break
            for d in _DELTAS:
                if f", {d})" in expr:
                    new_d = self.random_delta()
                    while new_d == d:
                        new_d = self.random_delta()
                    expr = expr.replace(f", {d})", f", {new_d})")
                    break
            return expr
        elif choice < 0.8:
            return self.generate_from_template()
        else:
            return self.generate()

    def _format_binary(self, template: str, a: str, b: str) -> str:
        fmt_kwargs = {"a": a, "b": b}
        if "{w}" in template:
            fmt_kwargs["w"] = self.random_window()
        return template.format(**fmt_kwargs)

    def crossover(self, a: str, b: str) -> str:
        parts_a = self._split_at_shallowest(a)
        parts_b = self._split_at_shallowest(b)
        if parts_a and parts_b:
            pa = self._rng.choice(parts_a)
            pb = self._rng.choice(parts_b)
            template, _ = self._rng.choice(_BINARY_OPS)
            child = self._format_binary(template, pa, pb)
        elif parts_a:
            pa = self._rng.choice(parts_a)
            template, _ = self._rng.choice(_BINARY_OPS)
            child = self._format_binary(template, pa, b)
        elif parts_b:
            pb = self._rng.choice(parts_b)
            template, _ = self._rng.choice(_BINARY_OPS)
            child = self._format_binary(template, a, pb)
        else:
            template, _ = self._rng.choice(_BINARY_OPS)
            child = self._format_binary(template, a, b)
        if self._rng.random() < 0.3:
            op = self._select_op(None, [t[1] for t in _UNARY_OPS])
            child = self._apply_op_by_name(op, child)
        return child

    def _split_at_shallowest(self, expr: str) -> list[str]:
        depth = 0
        min_op_depth = float("inf")
        for ch in expr:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch in "*/" and depth > 0:
                min_op_depth = min(min_op_depth, depth)
        if min_op_depth == float("inf"):
            depth = 0
            for ch in expr:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch in "+-" and depth > 0:
                    min_op_depth = min(min_op_depth, depth)
        if min_op_depth == float("inf"):
            return []
        parts: list[str] = []
        depth = 0
        current = ""
        for ch in expr:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == min_op_depth and ch in "*/+-":
                part = current.strip()
                if part:
                    parts.append(part)
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())
        return parts if len(parts) >= 2 else []


class SmartParameterTuner:
    """智能参数微调器：基于语法分析提取参数位置，精确替换生成候选。

    替代原有的四层嵌套暴力搜索（4×4×4×4=256次），通过：
    1. 正则提取表达式中所有数值参数及其上下文
    2. 区分窗口参数（_WINDOW_OPS）和delta参数（_DELTA_OPS）
    3. 对每个参数位置独立生成少量高质量候选值
    4. 精确位置替换，避免全局replace的误匹配

    复杂度从 O(n⁴) 降至 O(k·m)，k=参数位置数，m=每位置候选数（≤4）。
    """

    _WINDOW_PARAM_RE = re.compile(
        r'(ts_mean|ts_std|ts_rank|ts_sum|ts_max|ts_min|ts_argmax|ts_argmin'
        r'|ts_skewness|ts_kurtosis|ts_median|ts_moment|ts_decay_linear'
        r'|ts_corr|ts_cov)\([^,]+,\s*(\d+)\)'
    )
    _DELTA_PARAM_RE = re.compile(
        r'(ts_delta|ts_return|ts_delay)\([^,]+,\s*(\d+)\)'
    )

    def __init__(self, windows: list[int], deltas: list[int],
                 rng: random.Random, seen: set[str],
                 max_candidates_per_expr: int = 24) -> None:
        self._windows = windows
        self._deltas = deltas
        self._rng = rng
        self._seen = seen
        self._max_candidates = max_candidates_per_expr

    def generate_candidates(self, expr: str) -> list[str]:
        positions = self._analyze_parameters(expr)
        if not positions:
            return []

        candidates: list[str] = []
        for pos in positions:
            for new_val in self._smart_values(pos):
                new_expr = self._replace_at(expr, pos, new_val)
                if new_expr != expr and new_expr not in self._seen:
                    self._seen.add(new_expr)
                    candidates.append(new_expr)
                    if len(candidates) >= self._max_candidates:
                        return candidates
        return candidates

    def _analyze_parameters(self, expr: str) -> list[dict[str, Any]]:
        positions: list[dict[str, Any]] = []
        for m in self._WINDOW_PARAM_RE.finditer(expr):
            positions.append({
                "type": "window",
                "func": m.group(1),
                "current": int(m.group(2)),
                "start": m.start(2),
                "end": m.end(2),
            })
        for m in self._DELTA_PARAM_RE.finditer(expr):
            positions.append({
                "type": "delta",
                "func": m.group(1),
                "current": int(m.group(2)),
                "start": m.start(2),
                "end": m.end(2),
            })
        return positions

    def _smart_values(self, pos: dict[str, Any]) -> list[int]:
        current = pos["current"]
        if pos["type"] == "window":
            pool = [w for w in self._windows if w != current]
            nearby = sorted(pool, key=lambda w: abs(w - current))[:3]
            if len(pool) > 3:
                nearby.append(self._rng.choice(pool))
            return nearby
        else:
            pool = [d for d in self._deltas if d != current]
            nearby = sorted(pool, key=lambda d: abs(d - current))[:3]
            if len(pool) > 3:
                nearby.append(self._rng.choice(pool))
            return nearby

    def _replace_at(self, expr: str, pos: dict[str, Any], new_val: int) -> str:
        return expr[:pos["start"]] + str(new_val) + expr[pos["end"]:]


class FactorMiner:
    """因子挖掘引擎：UCB1引导搜索 + 遗传编程 + 表达式预筛选。

    改进点：
    1. UCB1多臂老虎机替代硬编码探索/利用比例，自动平衡
    2. n-gram条件概率实现上下文感知的算子选择
    3. 表达式预筛选：快速评估1-2只ETF，跳过全NaN/常数表达式
    4. 语法树交叉：在最浅层二元运算符处拆分，交换子表达式
    5. 高价值模板 + 更深嵌套(0-3层)
    """

    def __init__(self, calculator: ETFAlphaCalculator, config: MiningConfig,
                 existing_factors: list[dict[str, Any]] | None = None) -> None:
        self._calc = calculator
        self._config = config
        self._rng = random.Random(config.seed)
        self._gen = ExpressionGenerator(self._rng)
        self._seen: set[str] = set()
        self._population: list[tuple[str, float]] = []
        self._best_ic: float = 0.0
        self._best_expr: str = ""
        self._prescreen_codes: list[str] = []
        self._cancel_event = threading.Event()
        self._existing_factors = existing_factors or []
        try:
            self._prescreen_codes = calculator._bridge.list_etf_codes()[:3]
        except Exception:
            pass

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _quick_prescreen(self, expr: str) -> bool:
        if self._cancel_event.is_set():
            return False
        if not self._prescreen_codes:
            return True
        try:
            for code in self._prescreen_codes:
                vals = self._calc._evaluate_expression(expr, code)
                if vals is None:
                    continue
                valid = vals.replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) < 10:
                    continue
                if valid.std() < 1e-10:
                    return False
                return True
            return False
        except Exception:
            return False

    def _evaluate_expression(self, expr: str) -> tuple[float, float, float, float]:
        try:
            ic_mean, ic_std, ric_mean, ric_std, icir, ic_p = self._calc.calc_single_all_ret(expr)
            ic_mean = ic_mean if ic_mean is not None and not (isinstance(ic_mean, float) and (ic_mean != ic_mean)) else 0.0
            ric_mean = ric_mean if ric_mean is not None and not (isinstance(ric_mean, float) and (ric_mean != ric_mean)) else 0.0
            icir = icir if icir is not None else 0.0
            ic_p = ic_p if ic_p is not None else 1.0
            return ic_mean, ric_mean, icir, ic_p
        except Exception:
            return 0.0, 0.0, 0.0, 1.0

    def _is_valid(self, ic: float, ric: float, icir: float, ic_p: float) -> bool:
        return (abs(ic) >= self._config.ic_threshold or abs(ric) >= self._config.rank_ic_threshold) and abs(icir) >= self._config.icir_threshold and ic_p <= 0.05

    def _make_factor_name(self, expr: str, idx: int) -> str:
        return f"mined_{idx:04d}"

    def mine(self, progress_callback: Callable[[int, int, str, dict], None] | None = None,
             on_discover: Callable[[dict[str, Any]], None] | None = None) -> MiningResult:
        start_time = time.time()
        timeout = self._config.timeout_minutes * 60
        result = MiningResult()
        factor_idx = 0
        valid_expressions: dict[str, dict[str, Any]] = {}
        valid_factor_values: dict[str, pd.Series] = {}
        total_budget = self._config.n_steps * self._config.batch_size
        evaluated_count = 0
        last_report_time = start_time

        def _report(force: bool = False) -> None:
            nonlocal last_report_time
            now = time.time()
            if not force and (now - last_report_time) < 2.0:
                return
            last_report_time = now
            if progress_callback:
                pct = min(int(evaluated_count / total_budget * 100), 99) if total_budget > 0 else 0
                elapsed_min = (now - start_time) / 60
                progress_callback(evaluated_count, total_budget,
                                  f"评估{evaluated_count}个 | 有效{result.total_valid}个 | 跳过{result.total_skipped}个 | {elapsed_min:.1f}min",
                                  {"valid": result.total_valid, "evaluated": evaluated_count, "pct": pct})

        def _eval_and_record(expr: str, step: int | None = None, desc_prefix: str = "UCB1搜索") -> float:
            nonlocal evaluated_count, factor_idx
            if self._cancel_event.is_set():
                return 0.0
            if not self._quick_prescreen(expr):
                result.total_skipped += 1
                _report()
                return 0.0
            ic, ric, icir, ic_p = self._evaluate_expression(expr)
            evaluated_count += 1
            result.total_evaluated += 1
            is_valid = self._is_valid(ic, ric, icir, ic_p)
            self._gen.record_result(expr, ic, is_valid)
            if is_valid:
                is_redundant = False
                try:
                    new_vals = self._calc._evaluate_expression(expr, self._prescreen_codes[0]) if self._prescreen_codes else None
                    if new_vals is not None and not new_vals.empty and new_vals.std() > 1e-10:
                        for existing_name, existing_vals in valid_factor_values.items():
                            merged = pd.concat([new_vals, existing_vals], axis=1).dropna()
                            if len(merged) > 30:
                                corr = abs(merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman"))
                                if corr > 0.95:
                                    is_redundant = True
                                    result.total_skipped += 1
                                    break
                        if not is_redundant:
                            valid_factor_values[name] = new_vals
                except Exception:
                    pass
                if is_redundant:
                    _report()
                    return 0.0
                name = self._make_factor_name(expr, factor_idx)
                factor_idx += 1
                desc = f"{desc_prefix}发现 (第{evaluated_count}个表达式"
                if step is not None:
                    desc += f", step={step}"
                desc += ")"
                factor_dict = {
                    "name": name,
                    "expression": expr,
                    "ic": ic,
                    "rank_ic": ric,
                    "icir": icir,
                    "is_valid": True,
                    "category": "mined",
                    "description": desc,
                }
                valid_expressions[expr] = factor_dict
                self._population.append((expr, abs(ic)))
                result.total_valid += 1
                if abs(ic) > abs(self._best_ic):
                    self._best_ic = ic
                    self._best_expr = expr
                if on_discover:
                    try:
                        on_discover(factor_dict)
                    except Exception as exc:
                        logger.warning("on_discover回调异常: %s", exc)
            _report()
            return ic

        if self._config.strategy in ("UCB1搜索", "UCB1+遗传组合"):
            logger.info("开始UCB1引导因子搜索: n_steps=%d, batch_size=%d", self._config.n_steps, self._config.batch_size)
            for step in range(self._config.n_steps):
                if self._cancel_event.is_set():
                    logger.info("挖掘被用户取消，已运行 %d 步", step)
                    break
                if time.time() - start_time > timeout:
                    logger.info("搜索超时，已运行 %.1f 分钟", (time.time() - start_time) / 60)
                    break
                if len(valid_expressions) >= self._config.max_factors:
                    logger.info("已达到最大因子数 %d，停止搜索", self._config.max_factors)
                    break

                batch_expressions = []
                for _ in range(self._config.batch_size):
                    expr = self._gen.generate()
                    if expr not in self._seen:
                        self._seen.add(expr)
                        batch_expressions.append(expr)

                for expr in batch_expressions:
                    if time.time() - start_time > timeout:
                        break
                    _eval_and_record(expr, step=step, desc_prefix="UCB1搜索")

        if self._config.strategy in ("遗传搜索", "UCB1+遗传组合"):
            logger.info("开始遗传编程因子搜索")
            if not self._population:
                for _ in range(min(100, self._config.batch_size)):
                    if time.time() - start_time > timeout:
                        break
                    expr = self._gen.generate()
                    if expr not in self._seen:
                        self._seen.add(expr)
                        _eval_and_record(expr, desc_prefix="遗传初始化")

            n_generations = min(self._config.n_steps // 10, 200)
            population_size = min(self._config.batch_size, 64)

            for gen in range(n_generations):
                if self._cancel_event.is_set():
                    logger.info("遗传搜索被用户取消，已运行 %d 代", gen)
                    break
                if time.time() - start_time > timeout:
                    logger.info("遗传搜索超时，已运行 %.1f 分钟", (time.time() - start_time) / 60)
                    break
                if len(valid_expressions) >= self._config.max_factors:
                    break

                self._population.sort(key=lambda x: x[1], reverse=True)
                elite = self._population[:max(population_size // 4, 4)]
                offspring = list(elite)

                while len(offspring) < population_size:
                    t1 = self._rng.choice(elite)[0]
                    t2 = self._rng.choice(elite)[0]
                    child = self._gen.crossover(t1, t2)
                    if self._rng.random() < 0.3:
                        child = self._gen.mutate(child)
                    if child not in self._seen:
                        self._seen.add(child)
                        offspring.append((child, 0.0))

                for i, (expr, _) in enumerate(offspring[len(elite):]):
                    if time.time() - start_time > timeout:
                        break
                    ic = _eval_and_record(expr, desc_prefix="遗传搜索")
                    offspring[len(elite) + i] = (expr, abs(ic) if ic else 0.0)

                self._population = offspring

        if self._config.strategy == "因子组合搜索":
            logger.info("开始因子组合搜索: 已有%d个因子", len(self._existing_factors))
            valid_factors = [f for f in self._existing_factors if f.get("is_valid") or f.get("ic", 0) != 0]
            if len(valid_factors) < 2:
                logger.warning("因子组合搜索需要至少2个有效因子，当前仅%d个，退化为UCB1搜索", len(valid_factors))
                for step in range(min(self._config.n_steps, 512)):
                    if self._cancel_event.is_set() or time.time() - start_time > timeout:
                        break
                    if len(valid_expressions) >= self._config.max_factors:
                        break
                    batch_expressions = []
                    for _ in range(self._config.batch_size):
                        expr = self._gen.generate()
                        if expr not in self._seen:
                            self._seen.add(expr)
                            batch_expressions.append(expr)
                    for expr in batch_expressions:
                        if time.time() - start_time > timeout:
                            break
                        _eval_and_record(expr, step=step, desc_prefix="组合退化为UCB1")
            else:
                valid_factors.sort(key=lambda f: abs(f.get("ic", 0)), reverse=True)
                top_factors = valid_factors[:min(30, len(valid_factors))]
                combo_templates = [
                    "({a}) * ({b})",
                    "({a}) + ({b})",
                    "({a}) - ({b})",
                    "({a}) / (({b}) + 1e-8)",
                    "ts_rank({a}, 20) * sign({b})",
                    "sign({a}) * abs({b})",
                    "ts_rank({a}, 20) - ts_rank({b}, 20)",
                    "ts_corr({a}, {b}, 20)",
                ]
                combo_count = 0
                for i, f1 in enumerate(top_factors):
                    if self._cancel_event.is_set() or time.time() - start_time > timeout:
                        break
                    if len(valid_expressions) >= self._config.max_factors:
                        break
                    for j in range(i + 1, len(top_factors)):
                        if self._cancel_event.is_set() or time.time() - start_time > timeout:
                            break
                        if len(valid_expressions) >= self._config.max_factors:
                            break
                        f2 = top_factors[j]
                        e1 = f1.get("expression", "")
                        e2 = f2.get("expression", "")
                        if not e1 or not e2:
                            continue
                        for tmpl in combo_templates:
                            if self._cancel_event.is_set() or time.time() - start_time > timeout:
                                break
                            if len(valid_expressions) >= self._config.max_factors:
                                break
                            expr = tmpl.format(a=e1, b=e2)
                            if expr not in self._seen:
                                self._seen.add(expr)
                                combo_count += 1
                                _eval_and_record(expr, desc_prefix="因子组合")
                logger.info("因子组合搜索完成: 尝试%d个组合", combo_count)

        if self._config.strategy == "模因搜索":
            logger.info("开始模因搜索: 遗传搜索+智能参数微调")
            if not self._population:
                for _ in range(min(100, self._config.batch_size)):
                    if time.time() - start_time > timeout:
                        break
                    expr = self._gen.generate()
                    if expr not in self._seen:
                        self._seen.add(expr)
                        _eval_and_record(expr, desc_prefix="模因初始化")

            n_generations = min(self._config.n_steps // 10, 200)
            population_size = min(self._config.batch_size, 64)
            param_tuner = SmartParameterTuner(_WINDOWS, _DELTAS, self._rng, self._seen)

            for gen in range(n_generations):
                if self._cancel_event.is_set():
                    logger.info("模因搜索被用户取消，已运行 %d 代", gen)
                    break
                if time.time() - start_time > timeout:
                    break
                if len(valid_expressions) >= self._config.max_factors:
                    break

                self._population.sort(key=lambda x: x[1], reverse=True)
                elite = self._population[:max(population_size // 4, 4)]
                offspring = list(elite)

                while len(offspring) < population_size:
                    t1 = self._rng.choice(elite)[0]
                    t2 = self._rng.choice(elite)[0]
                    child = self._gen.crossover(t1, t2)
                    if self._rng.random() < 0.3:
                        child = self._gen.mutate(child)
                    if child not in self._seen:
                        self._seen.add(child)
                        offspring.append((child, 0.0))

                memetic_offspring = []
                for expr, score in offspring:
                    if self._cancel_event.is_set() or time.time() - start_time > timeout:
                        break
                    best_expr_local = expr
                    best_ic_local = score
                    for candidate in param_tuner.generate_candidates(expr):
                        if self._cancel_event.is_set() or time.time() - start_time > timeout:
                            break
                        ic_val = self._quick_prescreen(candidate)
                        if ic_val is False:
                            continue
                        ic, ric, icir, ic_p = self._evaluate_expression(candidate)
                        evaluated_count += 1
                        result.total_evaluated += 1
                        is_valid = self._is_valid(ic, ric, icir, ic_p)
                        self._gen.record_result(candidate, ic, is_valid)
                        if abs(ic) > abs(best_ic_local):
                            best_expr_local = candidate
                            best_ic_local = abs(ic)
                        if is_valid:
                            name = self._make_factor_name(candidate, factor_idx)
                            factor_idx += 1
                            factor_dict = {
                                "name": name, "expression": candidate,
                                "ic": ic, "rank_ic": ric, "icir": icir,
                                "is_valid": True, "category": "mined",
                                "description": "模因搜索发现(智能微调)",
                            }
                            valid_expressions[candidate] = factor_dict
                            self._population.append((candidate, abs(ic)))
                            result.total_valid += 1
                            if abs(ic) > abs(self._best_ic):
                                self._best_ic = ic
                                self._best_expr = candidate
                            if on_discover:
                                try:
                                    on_discover(factor_dict)
                                except Exception as exc:
                                    logger.warning("on_discover回调异常: %s", exc)
                        _report()
                    memetic_offspring.append((best_expr_local, best_ic_local))

                self._population = memetic_offspring

        _report(force=True)
        result.discovered = list(valid_expressions.values())
        result.elapsed_seconds = time.time() - start_time
        result.best_ic = self._best_ic if self._best_ic else 0.0
        result.best_expression = self._best_expr
        logger.info("因子挖掘完成: 评估%d个, 跳过%d个, 发现%d个有效因子, 耗时%.1fs",
                     result.total_evaluated, result.total_skipped, result.total_valid, result.elapsed_seconds)
        return result
