from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

__all__ = ["ETFQuantConfig", "DataConfig", "AlphaConfig", "MLConfig", "BacktestConfig", "UIConfig", "load_config"]


class DataConfig(BaseModel):
    data_root: str = r"D:\AKShare_Module\data\export"
    etf_daily_dir: str = "etf_daily"
    etf_nav_dir: str = "etf_nav"
    etf_premium_dir: str = "etf_premium"
    index_daily_dir: str = "index_daily"
    adjust_factor_dir: str = "adjust_factor"
    metadata_dir: str = "metadata"
    classification_file: str = r"D:\AKShare_Module\etf_classification.yaml"

    @property
    def etf_daily_path(self) -> Path:
        return Path(self.data_root) / self.etf_daily_dir

    @property
    def etf_nav_path(self) -> Path:
        return Path(self.data_root) / self.etf_nav_dir

    @property
    def etf_premium_path(self) -> Path:
        return Path(self.data_root) / self.etf_premium_dir

    @property
    def index_daily_path(self) -> Path:
        return Path(self.data_root) / self.index_daily_dir

    @property
    def adjust_factor_path(self) -> Path:
        return Path(self.data_root) / self.adjust_factor_dir


class ScheduleConfig(BaseModel):
    enabled: bool = False
    start_time: str = "18:00"
    end_time: str = "22:00"
    days: list[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


class ResourceConfig(BaseModel):
    gpu_utilization_limit: float = 0.8
    memory_limit_gb: float = 8.0
    max_concurrent_tasks: int = 1


class AlphaConfig(BaseModel):
    enabled: bool = True
    ic_threshold: float = 0.03
    rank_ic_threshold: float = 0.03
    mutual_ic_threshold: float = 0.7
    icir_threshold: float = 0.5
    max_factors: int = 50
    target_period: int = 20
    use_gpu: bool = True
    device: str = "cuda"
    rl_batch_size: int = 256
    save_path: str = "output/factors"
    db_path: str = "output/factors/factor_store.db"
    schedule: ScheduleConfig = ScheduleConfig()
    resources: ResourceConfig = ResourceConfig()
    cache_size: int = 128
    max_etf_for_ic: int = 1035
    max_etf_for_mutual_ic: int = 500


class FactorScreenConfig(BaseModel):
    ic_threshold: float = 0.03
    icir_threshold: float = 0.5
    mutual_ic_threshold: float = 0.7
    max_factors: int = 30


class MLConfig(BaseModel):
    enabled: bool = True
    model_type: str = "xgboost"
    lookback_days: int = 60
    predict_days: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    parallel_jobs: int = -1
    model_params: dict[str, Any] = {
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_estimators": 200,
    }
    save_path: str = "output/models"
    factor_screen: FactorScreenConfig = FactorScreenConfig()


class BacktestConfig(BaseModel):
    enabled: bool = True
    initial_capital: float = 100000.0
    frequency: str = "1d"
    t_plus_1: bool = False
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    min_trade_unit: int = 100
    price_limit_pct: float = 0.10
    benchmark: str = "510300.SH"
    save_path: str = "output/backtest"


class UIConfig(BaseModel):
    theme: str = "dark"
    language: str = "zh"
    host: str = "localhost"
    port: int = 8080
    title: str = "ETFQuantDesk"


class ETFQuantConfig(BaseModel):
    data: DataConfig = DataConfig()
    alpha: AlphaConfig = AlphaConfig()
    ml: MLConfig = MLConfig()
    backtest: BacktestConfig = BacktestConfig()
    ui: UIConfig = UIConfig()


def load_config(config_path: str = "config/etfquant.yaml") -> ETFQuantConfig:
    if not os.path.exists(config_path):
        return ETFQuantConfig()
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return ETFQuantConfig.model_validate(raw)
