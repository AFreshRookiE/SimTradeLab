from __future__ import annotations

import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

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
    strategy: str = "RL搜索"
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
    elapsed_seconds: float = 0.0
    best_ic: float = 0.0
    best_expression: str = ""


_TERMINALS = [
    "close", "open", "high", "low", "volume", "amount", "pct_chg",
    "nav", "iopv", "index_close",
]

_UNARY_OPS = [
    ("ts_mean({x}, {w})", "mean"),
    ("ts_std({x}, {w})", "std"),
    ("ts_rank({x}, {w})", "rank"),
    ("ts_sum({x}, {w})", "sum"),
    ("ts_max({x}, {w})", "max"),
    ("ts_min({x}, {w})", "min"),
    ("abs({x})", None),
    ("log(abs({x}) + 1e-8)", None),
    ("sign({x})", None),
    ("ts_delta({x}, {d})", "delta"),
    ("ts_return({x}, {d})", "return"),
    ("ts_delay({x}, {d})", "delay"),
]

_BINARY_OPS = [
    "({a} + {b})",
    "({a} - {b})",
    "({a} * {b})",
    "({a} / ({b} + 1e-8))",
]

_ETF_UNARY_OPS = [
    "premium_rate()",
    "iopv_deviation()",
    "tracking_error(20)",
    "tracking_error(10)",
    "tracking_error(5)",
]

_WINDOWS = [5, 10, 20, 60]
_DELTAS = [1, 5, 10, 20]


class ExpressionGenerator:
    """因子表达式生成器，支持随机生成、变异和交叉操作。"""
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def random_terminal(self) -> str:
        return self._rng.choice(_TERMINALS)

    def random_window(self) -> int:
        return self._rng.choice(_WINDOWS)

    def random_delta(self) -> int:
        return self._rng.choice(_DELTAS)

    def random_unary(self, expr: str) -> str:
        template, kind = self._rng.choice(_UNARY_OPS)
        if kind == "mean" or kind == "std" or kind == "rank" or kind == "sum" or kind == "max" or kind == "min":
            return template.format(x=expr, w=self.random_window())
        elif kind == "delta" or kind == "return" or kind == "delay":
            return template.format(x=expr, d=self.random_delta())
        return template.format(x=expr)

    def random_binary(self, a: str, b: str) -> str:
        template = self._rng.choice(_BINARY_OPS)
        return template.format(a=a, b=b)

    def random_etf_op(self) -> str:
        return self._rng.choice(_ETF_UNARY_OPS)

    def generate_simple(self) -> str:
        base = self._rng.choice([self.random_terminal, self.random_etf_op])
        expr = base()
        depth = self._rng.randint(0, 2)
        for _ in range(depth):
            expr = self.random_unary(expr)
        return expr

    def generate_composite(self) -> str:
        left = self.generate_simple()
        right = self._rng.choice([self.generate_simple, self.random_etf_op])
        expr = self.random_binary(left, right())
        if self._rng.random() < 0.3:
            expr = self.random_unary(expr)
        return expr

    def generate(self) -> str:
        if self._rng.random() < 0.4:
            return self.generate_simple()
        return self.generate_composite()

    def mutate(self, expr: str) -> str:
        choice = self._rng.random()
        if choice < 0.3:
            return self.random_unary(expr)
        elif choice < 0.5:
            other = self._rng.choice([self.generate_simple, self.random_etf_op])
            return self.random_binary(expr, other())
        elif choice < 0.7:
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
        else:
            return self.generate()

    def crossover(self, a: str, b: str) -> str:
        choice = self._rng.random()
        if choice < 0.5:
            return self.random_binary(a, b)
        else:
            if self._rng.random() < 0.5:
                return self.random_unary(a) if len(a) < len(b) else self.random_unary(b)
            return a if self._rng.random() < 0.5 else b


class FactorMiner:
    """因子挖掘引擎，支持RL搜索、遗传搜索和组合策略。

    RL搜索：基于经验反馈的启发式搜索，维护算子成功率统计指导生成方向。
    遗传搜索：模拟自然选择，对有效因子进行交叉变异产生新因子。
    """
    def __init__(self, calculator: ETFAlphaCalculator, config: MiningConfig) -> None:
        self._calc = calculator
        self._config = config
        self._rng = random.Random(config.seed)
        self._gen = ExpressionGenerator(self._rng)
        self._seen: set[str] = set()
        self._population: list[tuple[str, float]] = []
        self._best_ic: float = 0.0
        self._best_expr: str = ""
        self._step_weights: dict[str, float] = {}
        self._op_success: dict[str, float] = {}
        self._op_attempts: dict[str, float] = {}

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

    def _record_success(self, expr: str, ic: float) -> None:
        if "ts_mean" in expr:
            self._op_success["ts_mean"] = self._op_success.get("ts_mean", 0) + 1
        if "ts_std" in expr:
            self._op_success["ts_std"] = self._op_success.get("ts_std", 0) + 1
        if "ts_rank" in expr:
            self._op_success["ts_rank"] = self._op_success.get("ts_rank", 0) + 1
        if "ts_delta" in expr:
            self._op_success["ts_delta"] = self._op_success.get("ts_delta", 0) + 1
        if "ts_return" in expr:
            self._op_success["ts_return"] = self._op_success.get("ts_return", 0) + 1
        if "premium_rate" in expr:
            self._op_success["premium_rate"] = self._op_success.get("premium_rate", 0) + 1
        if "tracking_error" in expr:
            self._op_success["tracking_error"] = self._op_success.get("tracking_error", 0) + 1
        if "log" in expr:
            self._op_success["log"] = self._op_success.get("log", 0) + 1
        if "sign" in expr:
            self._op_success["sign"] = self._op_success.get("sign", 0) + 1
        if "volume" in expr:
            self._op_success["volume"] = self._op_success.get("volume", 0) + 1
        if "close" in expr:
            self._op_success["close"] = self._op_success.get("close", 0) + 1
        if abs(ic) > abs(self._best_ic):
            self._best_ic = ic
            self._best_expr = expr

    def _weighted_generate(self) -> str:
        if not self._op_success or self._rng.random() < 0.3:
            return self._gen.generate()
        sorted_ops = sorted(self._op_success.items(), key=lambda x: x[1], reverse=True)
        top_ops = [op for op, _ in sorted_ops[:5]]
        base = self._rng.choice(_TERMINALS)
        expr = base
        for _ in range(self._rng.randint(1, 3)):
            chosen_op = self._rng.choice(top_ops + ["random"])
            if chosen_op == "random" or chosen_op not in [t[1] for t in _UNARY_OPS]:
                expr = self._gen.random_unary(expr)
            elif chosen_op == "ts_mean":
                expr = f"ts_mean({expr}, {self._gen.random_window()})"
            elif chosen_op == "ts_std":
                expr = f"ts_std({expr}, {self._gen.random_window()})"
            elif chosen_op == "ts_rank":
                expr = f"ts_rank({expr}, {self._gen.random_window()})"
            elif chosen_op == "ts_delta":
                expr = f"ts_delta({expr}, {self._gen.random_delta()})"
            elif chosen_op == "ts_return":
                expr = f"ts_return({expr}, {self._gen.random_delta()})"
            elif chosen_op == "log":
                expr = f"log(abs({expr}) + 1e-8)"
            elif chosen_op == "sign":
                expr = f"sign({expr})"
        if self._rng.random() < 0.4:
            right = self._rng.choice(_TERMINALS)
            if "premium_rate" in top_ops:
                right = "premium_rate()"
            elif "tracking_error" in top_ops:
                right = f"tracking_error({self._gen.random_window()})"
            expr = self._gen.random_binary(expr, right)
        return expr

    def mine(self, progress_callback: Callable[[int, int, str, dict], None] | None = None) -> MiningResult:
        start_time = time.time()
        timeout = self._config.timeout_minutes * 60
        result = MiningResult()
        factor_idx = 0
        valid_expressions: dict[str, dict[str, Any]] = {}
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
                                  f"已评估{evaluated_count}个 | 发现{result.total_valid}个有效 | {elapsed_min:.1f}min",
                                  {"valid": result.total_valid, "evaluated": evaluated_count, "pct": pct})

        if self._config.strategy in ("RL搜索", "RL+遗传组合"):
            logger.info("开始RL因子搜索: n_steps=%d, batch_size=%d", self._config.n_steps, self._config.batch_size)
            for step in range(self._config.n_steps):
                if time.time() - start_time > timeout:
                    logger.info("RL搜索超时，已运行 %.1f 分钟", (time.time() - start_time) / 60)
                    break
                if len(valid_expressions) >= self._config.max_factors:
                    logger.info("已达到最大因子数 %d，停止搜索", self._config.max_factors)
                    break

                batch_expressions = []
                for _ in range(self._config.batch_size):
                    expr = self._weighted_generate()
                    if expr not in self._seen:
                        self._seen.add(expr)
                        batch_expressions.append(expr)

                for expr in batch_expressions:
                    if time.time() - start_time > timeout:
                        break
                    ic, ric, icir, ic_p = self._evaluate_expression(expr)
                    evaluated_count += 1
                    result.total_evaluated += 1
                    if self._is_valid(ic, ric, icir, ic_p):
                        self._record_success(expr, ic)
                        name = self._make_factor_name(expr, factor_idx)
                        factor_idx += 1
                        valid_expressions[expr] = {
                            "name": name,
                            "expression": expr,
                            "ic": ic,
                            "rank_ic": ric,
                            "icir": icir,
                            "is_valid": True,
                            "category": "mined",
                            "description": f"RL搜索发现 (第{evaluated_count}个表达式, step={step})",
                        }
                        self._population.append((expr, abs(ic)))
                        result.total_valid += 1
                    _report()

        if self._config.strategy in ("遗传搜索", "RL+遗传组合"):
            logger.info("开始遗传算法因子搜索")
            if not self._population:
                for _ in range(min(100, self._config.batch_size)):
                    if time.time() - start_time > timeout:
                        break
                    expr = self._gen.generate()
                    if expr not in self._seen:
                        self._seen.add(expr)
                        ic, ric, icir, ic_p = self._evaluate_expression(expr)
                        evaluated_count += 1
                        result.total_evaluated += 1
                        self._population.append((expr, abs(ic) if ic else 0.0))
                        if self._is_valid(ic, ric, icir, ic_p):
                            self._record_success(expr, ic)
                            name = self._make_factor_name(expr, factor_idx)
                            factor_idx += 1
                            valid_expressions[expr] = {
                                "name": name,
                                "expression": expr,
                                "ic": ic,
                                "rank_ic": ric,
                                "icir": icir,
                                "is_valid": True,
                                "category": "mined",
                                "description": "遗传搜索发现",
                            }
                            result.total_valid += 1
                        _report()

            n_generations = min(self._config.n_steps // 10, 200)
            population_size = min(self._config.batch_size, 64)

            for gen in range(n_generations):
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
                    ic, ric, icir, ic_p = self._evaluate_expression(expr)
                    evaluated_count += 1
                    result.total_evaluated += 1
                    offspring[len(elite) + i] = (expr, abs(ic) if ic else 0.0)
                    if self._is_valid(ic, ric, icir, ic_p):
                        self._record_success(expr, ic)
                        name = self._make_factor_name(expr, factor_idx)
                        factor_idx += 1
                        valid_expressions[expr] = {
                            "name": name,
                            "expression": expr,
                            "ic": ic,
                            "rank_ic": ric,
                            "icir": icir,
                            "is_valid": True,
                            "category": "mined",
                            "description": f"遗传搜索发现 (gen={gen})",
                        }
                        result.total_valid += 1
                    _report()

                self._population = offspring

        _report(force=True)
        result.discovered = list(valid_expressions.values())
        result.elapsed_seconds = time.time() - start_time
        result.best_ic = self._best_ic if self._best_ic else 0.0
        result.best_expression = self._best_expr
        logger.info("因子挖掘完成: 评估%d个表达式, 发现%d个有效因子, 耗时%.1fs",
                     result.total_evaluated, result.total_valid, result.elapsed_seconds)
        return result
