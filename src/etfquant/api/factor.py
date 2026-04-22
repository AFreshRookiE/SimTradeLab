from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from etfquant.alpha.calculator import AlphaFactor, AlphaPool, ETFAlphaCalculator, PresetFactors
from etfquant.alpha.factor_store import FactorStore
from etfquant.alpha.miner import FactorMiner, MiningConfig
from etfquant.alpha.scheduler import AlphaScheduler
from etfquant.core.config import AlphaConfig, MLConfig, ScheduleConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = ["FactorService"]

logger = get_logger("etfquant.api.factor")

_SCHEDULE_STATE_FILE = "output/schedule_state.json"
_MINING_LOG_FILE = "output/mining_log.json"


class FactorService:
    _mining_state: dict[str, Any] = {"running": False, "pct": 0, "msg": "", "result": None, "error": None}
    _mining_lock = threading.Lock()
    _current_miner: FactorMiner | None = None
    _miner_lock = threading.Lock()

    def __init__(self, config: AlphaConfig, data_config: Any, ml_config: MLConfig | None = None) -> None:
        self._config = config
        self._data_config = data_config
        self._ml_config = ml_config
        self._store = FactorStore(config.db_path)
        self._scheduler: AlphaScheduler | None = None
        self._load_schedule_state()

    def _load_schedule_state(self) -> None:
        state_path = Path(_SCHEDULE_STATE_FILE)
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self._config.schedule = ScheduleConfig(
                    enabled=state.get("enabled", False),
                    start_time=state.get("start_time", "18:00"),
                    end_time=state.get("end_time", "22:00"),
                    days=state.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
                )
                if self._config.schedule.enabled:
                    self._scheduler = AlphaScheduler(self._config.schedule, self._scheduled_task)
                    self._scheduler.start()
                    logger.info("已从持久化状态恢复定时任务调度器")
            except Exception as exc:
                logger.warning("加载定时任务状态失败: %s", exc)

    def _save_schedule_state(self) -> None:
        state_path = Path(_SCHEDULE_STATE_FILE)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "enabled": self._config.schedule.enabled,
            "start_time": self._config.schedule.start_time,
            "end_time": self._config.schedule.end_time,
            "days": self._config.schedule.days,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _scheduled_task(self) -> None:
        for attempt in range(2):
            try:
                gen_result = self.generate_preset_factors()
                mine_result = self.mine_factors(
                    n_steps=2048,
                    batch_size=64,
                    timeout_minutes=60,
                    max_factors=50,
                    strategy="UCB1搜索",
                )
                combined = {
                    "total": gen_result.get("total", 0) + mine_result.get("total_evaluated", 0),
                    "valid": gen_result.get("valid", 0) + mine_result.get("total_valid", 0),
                    "total_evaluated": mine_result.get("total_evaluated", 0),
                    "total_valid": mine_result.get("total_valid", 0),
                    "total_skipped": mine_result.get("total_skipped", 0),
                    "elapsed_seconds": mine_result.get("elapsed_seconds", 0),
                    "best_ic": mine_result.get("best_ic", 0),
                    "best_expression": mine_result.get("best_expression", ""),
                    "preset_total": gen_result.get("total", 0),
                    "preset_valid": gen_result.get("valid", 0),
                    "source": "定时任务",
                }
                self._record_mining(combined)
                return
            except Exception as exc:
                if attempt == 0:
                    logger.warning("定时任务执行失败，10秒后重试: %s", exc)
                    import time
                    time.sleep(10)
                else:
                    logger.error("定时任务重试仍失败: %s", exc)
                    self._record_mining({
                        "total": 0, "valid": 0,
                        "total_evaluated": 0, "total_valid": 0, "total_skipped": 0,
                        "elapsed_seconds": 0, "best_ic": 0, "best_expression": "",
                        "preset_total": 0, "preset_valid": 0,
                    })

    def _record_mining(self, result: dict[str, Any]) -> None:
        log_path = Path(_MINING_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logs: list[dict[str, Any]] = []
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        entry = {
            "timestamp": datetime.now().isoformat(),
            "total_factors": result.get("total", 0),
            "valid_factors": result.get("valid", 0),
            "total_evaluated": result.get("total_evaluated", 0),
            "total_valid": result.get("total_valid", 0),
            "total_skipped": result.get("total_skipped", 0),
            "elapsed_seconds": round(result.get("elapsed_seconds", 0), 1),
            "best_ic": round(result.get("best_ic", 0), 4),
            "best_expression": result.get("best_expression", ""),
            "preset_total": result.get("preset_total", 0),
            "preset_valid": result.get("preset_valid", 0),
            "source": result.get("source", "未知"),
            "strategy": result.get("strategy", ""),
            "n_steps": result.get("n_steps", 0),
            "batch_size": result.get("batch_size", 0),
        }
        logs.append(entry)
        if len(logs) > 100:
            logs = logs[-100:]
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    def list_factors(self, category: str | None = None, valid_only: bool = False) -> list[dict[str, Any]]:
        if valid_only:
            rows = self._store.list_valid()
        elif category:
            rows = self._store.list_by_category(category)
        else:
            rows = self._store.list_all()
        return rows

    def get_factor(self, name: str) -> dict[str, Any] | None:
        return self._store.get(name)

    def delete_factor(self, name: str) -> bool:
        return self._store.delete(name)

    def clear_invalid(self) -> int:
        return self._store.delete_invalid()

    def save_screen_result(self, ic_threshold: float, icir_threshold: float,
                           mutual_ic_threshold: float, max_factors: int,
                           selected_names: list[str]) -> int:
        return self._store.save_screen_result(ic_threshold, icir_threshold, mutual_ic_threshold, max_factors, selected_names)

    def get_latest_screen_result(self) -> dict[str, Any] | None:
        return self._store.get_latest_screen_result()

    def get_selected_factor_expressions(self) -> list[dict[str, Any]]:
        return self._store.get_selected_factor_expressions()

    def revalidate_all(self, ic_threshold: float | None = None,
                       rank_ic_threshold: float | None = None,
                       icir_threshold: float | None = None) -> dict[str, Any]:
        """按当前规则重新判定因子池中所有因子的有效性（不重新计算IC）。"""
        all_factors = self._store.list_all()
        ic_t = ic_threshold if ic_threshold is not None else self._config.ic_threshold
        ric_t = rank_ic_threshold if rank_ic_threshold is not None else self._config.rank_ic_threshold
        icir_t = icir_threshold if icir_threshold is not None else self._config.icir_threshold
        updated = 0
        for f_dict in all_factors:
            ic = f_dict.get("ic") or 0
            ric = f_dict.get("rank_ic") or 0
            icir = f_dict.get("icir") or 0
            new_valid = (abs(ic) >= ic_t or abs(ric) >= ric_t) and abs(icir) >= icir_t
            old_valid = bool(f_dict.get("is_valid"))
            if new_valid != old_valid:
                f_dict["is_valid"] = new_valid
                factor = AlphaFactor(
                    name=f_dict["name"],
                    expression=f_dict["expression"],
                    description=f_dict.get("description", ""),
                    ic=ic,
                    rank_ic=ric,
                    icir=icir,
                    is_valid=new_valid,
                    category=f_dict.get("category", "custom"),
                )
                self._store.upsert(factor)
                updated += 1
        logger.info("重新校验: %d个因子有效性变更 (规则: IC>=%.3f, ICIR>=%.2f)", updated, ic_t, icir_t)
        return {"total": len(all_factors), "updated": updated}

    def generate_preset_factors(self, ic_threshold: float | None = None, rank_ic_threshold: float | None = None,
                                icir_threshold: float | None = None, target_period: int | None = None,
                                max_etf_for_ic: int | None = None, max_etf_for_mutual_ic: int | None = None,
                                progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
        cfg = AlphaConfig(
            ic_threshold=ic_threshold if ic_threshold is not None else self._config.ic_threshold,
            rank_ic_threshold=rank_ic_threshold if rank_ic_threshold is not None else self._config.rank_ic_threshold,
            icir_threshold=icir_threshold if icir_threshold is not None else self._config.icir_threshold,
            target_period=target_period if target_period is not None else self._config.target_period,
            max_etf_for_ic=max_etf_for_ic if max_etf_for_ic is not None else self._config.max_etf_for_ic,
            max_etf_for_mutual_ic=max_etf_for_mutual_ic if max_etf_for_mutual_ic is not None else self._config.max_etf_for_mutual_ic,
            db_path=self._config.db_path,
            save_path=self._config.save_path,
            schedule=self._config.schedule,
            resources=self._config.resources,
        )
        logger.info("生成因子参数: ic=%.4f, ric=%.4f, icir=%.2f, period=%d, max_etf_ic=%d, max_etf_mutual=%d",
                     cfg.ic_threshold, cfg.rank_ic_threshold, cfg.icir_threshold,
                     cfg.target_period, cfg.max_etf_for_ic, cfg.max_etf_for_mutual_ic)
        bridge = DataBridge(self._data_config)
        calculator = ETFAlphaCalculator(bridge, cfg)
        pool = PresetFactors.evaluate_all(calculator, cfg, progress_callback=progress_callback)
        for f in pool.factors:
            self._store.upsert(f)
        valid_count = len(pool.filter_valid())
        return {
            "total": len(pool.factors),
            "valid": valid_count,
            "factors": [{"name": f.name, "ic": f.ic, "rank_ic": f.rank_ic, "icir": f.icir, "is_valid": f.is_valid} for f in pool.factors],
        }

    def evaluate_expression(self, expression: str) -> dict[str, Any]:
        bridge = DataBridge(self._data_config)
        calculator = ETFAlphaCalculator(bridge, self._config)
        ic_mean, _, ric_mean, _, icir, ic_p = calculator.calc_single_all_ret(expression)
        return {"expression": expression, "ic": ic_mean, "rank_ic": ric_mean, "icir": icir, "ic_p": ic_p}

    def mine_factors(self, n_steps: int = 2048, batch_size: int = 64,
                     timeout_minutes: float = 30.0, max_factors: int = 50,
                     strategy: str = "UCB1搜索",
                     ic_threshold: float | None = None,
                     rank_ic_threshold: float | None = None,
                     icir_threshold: float | None = None,
                     target_period: int | None = None,
                     max_etf_for_ic: int | None = None,
                     max_etf_for_mutual_ic: int | None = None,
                     progress_callback: Callable | None = None) -> dict[str, Any]:
        with self._mining_lock:
            if self._mining_state.get("running"):
                return {"error": "挖掘任务正在执行中，请等待完成"}
            self._mining_state = {
                "running": True,
                "pct": 0,
                "msg": "准备中...",
                "start_time": datetime.now().isoformat(),
                "params": {
                    "n_steps": n_steps, "batch_size": batch_size,
                    "timeout_minutes": timeout_minutes, "max_factors": max_factors,
                    "strategy": strategy,
                },
                "result": None,
                "error": None,
            }

        def _wrapped_progress(current, total, msg, info):
            pct = info.get("pct", 0)
            with self._mining_lock:
                self._mining_state["pct"] = pct
                self._mining_state["msg"] = msg
            if progress_callback:
                progress_callback(current, total, msg, info)

        mining_cfg = MiningConfig(
            n_steps=n_steps,
            batch_size=batch_size,
            timeout_minutes=timeout_minutes,
            max_factors=max_factors,
            strategy=strategy,
            ic_threshold=ic_threshold if ic_threshold is not None else self._config.ic_threshold,
            rank_ic_threshold=rank_ic_threshold if rank_ic_threshold is not None else self._config.rank_ic_threshold,
            icir_threshold=icir_threshold if icir_threshold is not None else self._config.icir_threshold,
            target_period=target_period if target_period is not None else self._config.target_period,
            max_etf_for_ic=max_etf_for_ic if max_etf_for_ic is not None else self._config.max_etf_for_ic,
            max_etf_for_mutual_ic=max_etf_for_mutual_ic if max_etf_for_mutual_ic is not None else self._config.max_etf_for_mutual_ic,
        )
        alpha_cfg = AlphaConfig(
            ic_threshold=mining_cfg.ic_threshold,
            rank_ic_threshold=mining_cfg.rank_ic_threshold,
            icir_threshold=mining_cfg.icir_threshold,
            target_period=mining_cfg.target_period,
            max_etf_for_ic=mining_cfg.max_etf_for_ic,
            max_etf_for_mutual_ic=mining_cfg.max_etf_for_mutual_ic,
            db_path=self._config.db_path,
            save_path=self._config.save_path,
            schedule=self._config.schedule,
            resources=self._config.resources,
        )
        logger.info("因子挖掘参数: strategy=%s, n_steps=%d, batch=%d, timeout=%.0fm, max=%d",
                     strategy, n_steps, batch_size, timeout_minutes, max_factors)
        bridge = DataBridge(self._data_config)
        calculator = ETFAlphaCalculator(bridge, alpha_cfg)
        existing_factors = []
        if strategy == "因子组合搜索":
            existing_factors = self.list_factors(valid_only=True)
        miner = FactorMiner(calculator, mining_cfg, existing_factors=existing_factors)
        with self._miner_lock:
            self._current_miner = miner

        def _on_discover(factor_dict: dict[str, Any]) -> None:
            factor = AlphaFactor(
                name=factor_dict["name"],
                expression=factor_dict["expression"],
                description=factor_dict.get("description", ""),
                ic=factor_dict["ic"],
                rank_ic=factor_dict["rank_ic"],
                icir=factor_dict["icir"],
                is_valid=factor_dict["is_valid"],
                category=factor_dict.get("category", "custom"),
            )
            self._store.upsert(factor)
            logger.debug("增量保存因子: %s (IC=%.4f)", factor.name, factor.ic)

        try:
            mining_result = miner.mine(progress_callback=_wrapped_progress, on_discover=_on_discover)
            for f_dict in mining_result.discovered:
                factor = AlphaFactor(
                    name=f_dict["name"],
                    expression=f_dict["expression"],
                    description=f_dict.get("description", ""),
                    ic=f_dict["ic"],
                    rank_ic=f_dict["rank_ic"],
                    icir=f_dict["icir"],
                    is_valid=f_dict["is_valid"],
                    category=f_dict.get("category", "custom"),
                )
                self._store.upsert(factor)
            result = {
                "total_evaluated": mining_result.total_evaluated,
                "total_valid": mining_result.total_valid,
                "total_skipped": mining_result.total_skipped,
                "elapsed_seconds": mining_result.elapsed_seconds,
                "best_ic": mining_result.best_ic,
                "best_expression": mining_result.best_expression,
                "factors": mining_result.discovered,
            }
            with self._mining_lock:
                self._mining_state["running"] = False
                self._mining_state["pct"] = 100
                self._mining_state["result"] = result
            with self._miner_lock:
                self._current_miner = None
            self._record_mining({
                "total": mining_result.total_evaluated,
                "valid": mining_result.total_valid,
                "total_evaluated": mining_result.total_evaluated,
                "total_valid": mining_result.total_valid,
                "total_skipped": mining_result.total_skipped,
                "elapsed_seconds": mining_result.elapsed_seconds,
                "best_ic": mining_result.best_ic,
                "best_expression": mining_result.best_expression,
                "preset_total": 0,
                "preset_valid": 0,
                "source": "手动挖掘",
                "strategy": strategy,
                "n_steps": n_steps,
                "batch_size": batch_size,
            })
            return result
        except Exception as e:
            with self._mining_lock:
                self._mining_state["running"] = False
                self._mining_state["error"] = str(e)
            with self._miner_lock:
                self._current_miner = None
            raise

    def stop_mining(self) -> dict[str, str]:
        with self._miner_lock:
            if self._current_miner is not None:
                self._current_miner.cancel()
                return {"status": "cancelling"}
        return {"status": "no_task"}

    def get_mining_status(self) -> dict[str, Any]:
        with self._mining_lock:
            return dict(self._mining_state)

    def get_schedule_status(self) -> dict[str, Any]:
        if self._scheduler:
            return self._scheduler.get_status()
        return {"enabled": self._config.schedule.enabled, "running": False, "paused": False, "status": "idle"}

    def update_schedule_config(self, enabled: bool, start_time: str, end_time: str, days: list[str]) -> dict[str, Any]:
        self._config.schedule = ScheduleConfig(
            enabled=enabled,
            start_time=start_time,
            end_time=end_time,
            days=days,
        )
        self._save_schedule_state()
        if self._scheduler and self._scheduler.is_running:
            self._scheduler.stop()
            self._scheduler = None
        if enabled:
            self._scheduler = AlphaScheduler(self._config.schedule, self._scheduled_task)
            self._scheduler.start()
        return {"status": "updated", "enabled": enabled}

    def start_schedule(self) -> dict[str, str]:
        if self._scheduler and self._scheduler.is_running:
            return {"status": "already_running"}
        self._config.schedule.enabled = True
        self._save_schedule_state()
        self._scheduler = AlphaScheduler(self._config.schedule, self._scheduled_task)
        self._scheduler.start()
        return {"status": "started"}

    def stop_schedule(self) -> dict[str, str]:
        if self._scheduler:
            self._scheduler.stop()
            self._scheduler = None
        self._config.schedule.enabled = False
        self._save_schedule_state()
        return {"status": "stopped"}

    def pause_schedule(self) -> dict[str, str]:
        if self._scheduler:
            self._scheduler.pause()
        return {"status": "paused"}

    def resume_schedule(self) -> dict[str, str]:
        if self._scheduler:
            self._scheduler.resume()
        return {"status": "resumed"}

    def get_mining_log(self) -> list[dict[str, Any]]:
        log_path = Path(_MINING_LOG_FILE)
        if not log_path.exists():
            return []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_operators(self) -> list[dict[str, str]]:
        return [
            {"op": "premium_rate()", "name": "折溢价率", "desc": "(close - nav) / nav", "example": "premium_rate()", "tip": "ETF专属，跨境ETF效果最好"},
            {"op": "tracking_error(w)", "name": "跟踪误差", "desc": "滚动w日超额收益标准差×年化因子", "example": "tracking_error(20)", "tip": "w=20为常用窗口"},
            {"op": "iopv_deviation()", "name": "IOPV偏离度", "desc": "(close - iopv) / iopv", "example": "iopv_deviation()", "tip": "盘中交易时更有效"},
            {"op": "ts_return(s, d)", "name": "滚动收益率", "desc": "d日收益率", "example": "ts_return(close, 20)", "tip": "d=5周频, d=20月频"},
            {"op": "ts_mean(s, w)", "name": "滚动均值", "desc": "w日移动平均", "example": "ts_mean(close, 20)", "tip": "常用5/10/20/60日"},
            {"op": "ts_std(s, w)", "name": "滚动标准差", "desc": "w日波动率", "example": "ts_std(close, 20)", "tip": "衡量波动大小"},
            {"op": "ts_rank(s, w)", "name": "滚动排名", "desc": "w日内百分位排名", "example": "ts_rank(close, 20)", "tip": "0~1之间，越接近1越强势"},
            {"op": "ts_corr(s1, s2, w)", "name": "滚动相关性", "desc": "两序列w日相关系数", "example": "ts_corr(close, volume, 20)", "tip": "量价相关性常用"},
            {"op": "ts_delta(s, d)", "name": "差值", "desc": "s - s.shift(d)", "example": "ts_delta(close, 5)", "tip": "当前值与d天前的差"},
            {"op": "ts_delay(s, d)", "name": "延迟", "desc": "s.shift(d)", "example": "ts_delay(close, 1)", "tip": "取d天前的值"},
        ]

    def export_factors(self, path: str | None = None) -> str:
        import pandas as pd
        rows = self._store.list_all()
        if not rows:
            return ""
        df = pd.DataFrame(rows)
        save_path = path or str(Path(self._config.save_path) / "alpha_pool.parquet")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        return save_path

    def train_model(self, etf_codes: list[str] | None = None, etf_count: int = 100, predict_days: int | None = None, factor_names: list[str] | None = None) -> dict[str, Any]:
        from etfquant.ml.trainer import ETFDataSource, FeatureEngineer, ModelTrainer

        if not self._ml_config:
            return {"success": False, "error": "ML配置未提供"}

        if predict_days is not None:
            self._ml_config.predict_days = predict_days

        bridge = DataBridge(self._data_config)
        ds = ETFDataSource(bridge, self._ml_config)

        factor_exprs = []
        if factor_names:
            with self._store._lock:
                placeholders = ",".join("?" for _ in factor_names)
                rows = self._store._conn.execute(
                    f"SELECT name, expression, ic, rank_ic, icir FROM factors WHERE name IN ({placeholders})",
                    factor_names,
                ).fetchall()
                factor_exprs = [{"name": r["name"], "expression": r["expression"], "ic": r["ic"], "rank_ic": r["rank_ic"], "icir": r["icir"]} for r in rows]
            logger.info("使用手动选择因子训练: %d 个", len(factor_exprs))
        if not factor_exprs:
            factor_exprs = self._store.get_selected_factor_expressions()
        if factor_exprs:
            logger.info("使用筛选因子训练: %d 个因子表达式", len(factor_exprs))
        else:
            logger.info("未找到筛选结果，使用硬编码特征训练")

        fe = FeatureEngineer(ds, self._ml_config, factor_expressions=factor_exprs if factor_exprs else None)

        codes = etf_codes or ds.get_stock_list()[:etf_count]
        logger.info("开始训练模型: %d 只ETF, 预测%d天, 特征来源=%s", len(codes), self._ml_config.predict_days, "筛选因子" if factor_exprs else "硬编码")

        X, y, dates = fe.build_dataset(codes)
        if X.empty:
            return {"success": False, "error": "训练数据为空，请检查数据或减少ETF数量"}

        trainer = ModelTrainer(self._ml_config)
        model_pkg = trainer.train(X, y, dates, factor_expressions=factor_exprs if factor_exprs else None)

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        model_name = f"Model_{len(codes)}ETFs_{self._ml_config.predict_days}d_{timestamp}"
        save_path = Path(self._ml_config.save_path) / f"{model_name}.ptp"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_pkg.save(str(save_path))
        if model_pkg.metadata:
            model_pkg.metadata["save_path"] = str(save_path)
            model_pkg.metadata["feature_source"] = "screened_factors" if factor_exprs else "hardcoded"

        return {
            "success": True,
            "model_path": str(save_path),
            "train_samples": model_pkg.metadata.get("train_samples", 0) if model_pkg.metadata else 0,
            "val_samples": model_pkg.metadata.get("val_samples", 0) if model_pkg.metadata else 0,
            "feature_count": model_pkg.metadata.get("feature_count", 0) if model_pkg.metadata else 0,
            "feature_source": "筛选因子" if factor_exprs else "硬编码特征",
            "train_period": model_pkg.metadata.get("train_period", "") if model_pkg.metadata else "",
            "val_period": model_pkg.metadata.get("val_period", "") if model_pkg.metadata else "",
        }

    def list_saved_models(self) -> list[dict[str, Any]]:
        if not self._ml_config:
            return []
        model_dir = Path(self._ml_config.save_path)
        if not model_dir.exists():
            return []
        models = []
        for f in model_dir.glob("*.ptp"):
            models.append({
                "name": f.stem,
                "path": str(f),
                "size_mb": f.stat().st_size / (1024 * 1024),
                "modified": f.stat().st_mtime,
            })
        return models

    def export_model(self, model_name: str, fmt: str = "json") -> str:
        from etfquant.ml.trainer import ModelPackage
        if not self._ml_config:
            return ""
        model_dir = Path(self._ml_config.save_path)
        ptp_path = model_dir / f"{model_name}.ptp"
        if not ptp_path.exists():
            return ""
        pkg = ModelPackage.load(str(ptp_path))
        meta = pkg.metadata or {}
        feat_names = pkg.feature_names or []
        export_dir = model_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            import json as json_mod
            export_data = {
                "model_name": model_name,
                "feature_names": feat_names,
                "feature_count": meta.get("feature_count", 0),
                "train_samples": meta.get("train_samples", 0),
                "val_samples": meta.get("val_samples", 0),
                "train_period": meta.get("train_period", ""),
                "val_period": meta.get("val_period", ""),
                "feature_source": meta.get("feature_source", ""),
                "model_type": type(pkg.model).__name__ if pkg.model else "",
            }
            out_path = export_dir / f"{model_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json_mod.dump(export_data, f, ensure_ascii=False, indent=2)

        elif fmt == "py":
            feat_str = repr(feat_names)
            model_type = type(pkg.model).__name__ if pkg.model else "Unknown"
            py_code = f'''import pickle
import pandas as pd
import numpy as np

def load_model(path="{model_name}.ptp"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def predict(model_data, features_df):
    model = model_data["model"]
    scaler = model_data.get("scaler")
    feature_names = model_data.get("feature_names", [])
    X = features_df[feature_names] if feature_names else features_df
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict(X)

# Model: {model_name}
# Type: {model_type}
# Features: {feat_str}
# Train samples: {meta.get("train_samples", 0)}
# Val samples: {meta.get("val_samples", 0)}
# Train period: {meta.get("train_period", "")}
# Val period: {meta.get("val_period", "")}
'''
            out_path = export_dir / f"{model_name}.py"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(py_code)

        elif fmt == "md":
            md_content = f'''# {model_name}

## Model Info
- **Type**: {type(pkg.model).__name__ if pkg.model else "Unknown"}
- **Feature Source**: {meta.get("feature_source", "")}
- **Train Samples**: {meta.get("train_samples", 0)}
- **Val Samples**: {meta.get("val_samples", 0)}
- **Train Period**: {meta.get("train_period", "")}
- **Val Period**: {meta.get("val_period", "")}

## Features ({len(feat_names)})
| # | Feature Name |
|---|-------------|
'''
            for i, fn in enumerate(feat_names, 1):
                md_content += f"| {i} | `{fn}` |\n"
            out_path = export_dir / f"{model_name}.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_content)

        elif fmt == "txt":
            lines = [
                f"Model: {model_name}",
                f"Type: {type(pkg.model).__name__ if pkg.model else 'Unknown'}",
                f"Feature Source: {meta.get('feature_source', '')}",
                f"Train Samples: {meta.get('train_samples', 0)}",
                f"Val Samples: {meta.get('val_samples', 0)}",
                f"Train Period: {meta.get('train_period', '')}",
                f"Val Period: {meta.get('val_period', '')}",
                f"Features ({len(feat_names)}):",
            ]
            for i, fn in enumerate(feat_names, 1):
                lines.append(f"  {i}. {fn}")
            out_path = export_dir / f"{model_name}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        else:
            return ""

        return str(out_path)

    def get_model_detail(self, model_name: str) -> dict[str, Any]:
        from etfquant.ml.trainer import ModelPackage
        if not self._ml_config:
            return {}
        model_dir = Path(self._ml_config.save_path)
        ptp_path = model_dir / f"{model_name}.ptp"
        if not ptp_path.exists():
            return {}
        try:
            pkg = ModelPackage.load(str(ptp_path))
        except Exception as e:
            return {"error": str(e)}
        meta = pkg.metadata or {}
        feat_names = pkg.feature_names or []
        model_type = type(pkg.model).__name__ if pkg.model else "Unknown"
        model_module = type(pkg.model).__module__ if pkg.model else ""
        scaler_type = type(pkg.scaler).__name__ if pkg.scaler else "None"
        return {
            "model_name": model_name,
            "model_type": model_type,
            "model_module": model_module,
            "scaler_type": scaler_type,
            "feature_count": meta.get("feature_count", len(feat_names)),
            "feature_names": feat_names,
            "train_samples": meta.get("train_samples", 0),
            "val_samples": meta.get("val_samples", 0),
            "train_period": meta.get("train_period", ""),
            "val_period": meta.get("val_period", ""),
            "feature_source": meta.get("feature_source", ""),
            "file_size_kb": round(ptp_path.stat().st_size / 1024, 1),
            "file_path": str(ptp_path),
            "model_params": meta.get("model_params", {}),
        }

    def close(self) -> None:
        if self._scheduler:
            self._scheduler.stop()
        self._store.close()
