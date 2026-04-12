from __future__ import annotations

from typing import Any

from etfquant.alpha.calculator import AlphaFactor, AlphaPool, ETFAlphaCalculator, PresetFactors
from etfquant.alpha.factor_store import FactorStore
from etfquant.alpha.scheduler import AlphaScheduler
from etfquant.core.config import AlphaConfig, MLConfig, ScheduleConfig
from etfquant.core.logger import get_logger
from etfquant.data.bridge import DataBridge

__all__ = ["FactorService"]

logger = get_logger("etfquant.api.factor")


class FactorService:
    def __init__(self, config: AlphaConfig, data_config: Any, ml_config: MLConfig | None = None) -> None:
        self._config = config
        self._data_config = data_config
        self._ml_config = ml_config
        self._store = FactorStore(config.db_path)
        self._scheduler: AlphaScheduler | None = None

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

    def generate_preset_factors(self) -> dict[str, Any]:
        bridge = DataBridge(self._data_config)
        calculator = ETFAlphaCalculator(bridge, self._config)
        pool = PresetFactors.evaluate_all(calculator, self._config)
        for f in pool.factors:
            self._store.upsert(f)
        valid = len(pool.filter_valid())
        return {
            "total": len(pool.factors),
            "valid": valid,
            "factors": [{"name": f.name, "ic": f.ic, "rank_ic": f.rank_ic, "is_valid": f.is_valid} for f in pool.factors],
        }

    def evaluate_expression(self, expression: str) -> dict[str, Any]:
        bridge = DataBridge(self._data_config)
        calculator = ETFAlphaCalculator(bridge, self._config)
        ic, ric = calculator.calc_single_all_ret(expression)
        return {"expression": expression, "ic": ic, "rank_ic": ric}

    def get_schedule_status(self) -> dict[str, Any]:
        if self._scheduler:
            return self._scheduler.get_status()
        return {"enabled": self._config.schedule.enabled, "running": False, "paused": False, "status": "idle"}

    def start_schedule(self) -> dict[str, str]:
        if self._scheduler and self._scheduler.is_running:
            return {"status": "already_running"}
        self._scheduler = AlphaScheduler(self._config.schedule, self.generate_preset_factors)
        self._scheduler.start()
        return {"status": "started"}

    def stop_schedule(self) -> dict[str, str]:
        if self._scheduler:
            self._scheduler.stop()
            self._scheduler = None
        return {"status": "stopped"}

    def pause_schedule(self) -> dict[str, str]:
        if self._scheduler:
            self._scheduler.pause()
        return {"status": "paused"}

    def resume_schedule(self) -> dict[str, str]:
        if self._scheduler:
            self._scheduler.resume()
        return {"status": "resumed"}

    def get_operators(self) -> list[dict[str, str]]:
        return [
            {"op": "premium_rate()", "name": "折溢价率", "desc": "(close - nav) / nav"},
            {"op": "tracking_error(w)", "name": "跟踪误差", "desc": "滚动w日超额收益标准差×年化因子"},
            {"op": "iopv_deviation()", "name": "IOPV偏离度", "desc": "(close - iopv) / iopv"},
            {"op": "ts_return(s, d)", "name": "滚动收益率", "desc": "d日收益率"},
            {"op": "ts_mean(s, w)", "name": "滚动均值", "desc": "w日移动平均"},
            {"op": "ts_std(s, w)", "name": "滚动标准差", "desc": "w日波动率"},
            {"op": "ts_rank(s, w)", "name": "滚动排名", "desc": "w日内百分位排名"},
            {"op": "ts_corr(s1, s2, w)", "name": "滚动相关性", "desc": "两序列w日相关系数"},
            {"op": "ts_delta(s, d)", "name": "差值", "desc": "s - s.shift(d)"},
            {"op": "ts_delay(s, d)", "name": "延迟", "desc": "s.shift(d)"},
        ]

    def export_factors(self, path: str | None = None) -> str:
        import pandas as pd
        from pathlib import Path
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

        from pathlib import Path
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
        from pathlib import Path
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
