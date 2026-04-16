from __future__ import annotations

import json
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
        result = self.generate_preset_factors()
        self._record_mining(result)

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
            "factors": result.get("factors", []),
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
                     strategy: str = "RL搜索",
                     ic_threshold: float | None = None,
                     rank_ic_threshold: float | None = None,
                     icir_threshold: float | None = None,
                     target_period: int | None = None,
                     max_etf_for_ic: int | None = None,
                     max_etf_for_mutual_ic: int | None = None,
                     progress_callback: Callable | None = None) -> dict[str, Any]:
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
        miner = FactorMiner(calculator, mining_cfg)
        mining_result = miner.mine(progress_callback=progress_callback)
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
        return {
            "total_evaluated": mining_result.total_evaluated,
            "total_valid": mining_result.total_valid,
            "elapsed_seconds": mining_result.elapsed_seconds,
            "best_ic": mining_result.best_ic,
            "best_expression": mining_result.best_expression,
            "factors": mining_result.discovered,
        }

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

    def train_model(self, etf_codes: list[str] | None = None) -> dict[str, Any]:
        from etfquant.ml.trainer import ETFDataSource, FeatureEngineer, ModelTrainer
        from etfquant.ml.factor_screener import FactorScreener

        if not self._ml_config:
            return {"success": False, "error": "ML配置未提供"}

        bridge = DataBridge(self._data_config)
        ds = ETFDataSource(bridge, self._ml_config)
        fe = FeatureEngineer(ds, self._ml_config)

        codes = etf_codes or ds.get_stock_list()[:100]
        logger.info("开始训练模型: %d 只ETF", len(codes))

        X, y, dates = fe.build_dataset(codes)
        if X.empty:
            return {"success": False, "error": "训练数据为空，请检查数据或减少ETF数量"}

        trainer = ModelTrainer(self._ml_config)
        model_pkg = trainer.train(X, y, dates)

        save_path = Path(self._ml_config.save_path) / "etf_model.ptp"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_pkg.save(str(save_path))
        if model_pkg.metadata:
            model_pkg.metadata["save_path"] = str(save_path)

        return {
            "success": True,
            "model_path": str(save_path),
            "train_samples": model_pkg.metadata.get("train_samples", 0) if model_pkg.metadata else 0,
            "val_samples": model_pkg.metadata.get("val_samples", 0) if model_pkg.metadata else 0,
            "feature_count": model_pkg.metadata.get("feature_count", 0) if model_pkg.metadata else 0,
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

    def close(self) -> None:
        if self._scheduler:
            self._scheduler.stop()
        self._store.close()
