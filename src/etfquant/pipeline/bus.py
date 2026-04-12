from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from etfquant.core.config import ETFQuantConfig
from etfquant.core.logger import get_logger

__all__ = [
    "PipelineBus",
    "TaskStatus",
    "TaskInfo",
    "PipelineStage",
    "PipelineResult",
]

logger = get_logger("etfquant.pipeline")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(str, Enum):
    DATA_CHECK = "data_check"
    ALPHA_GENERATE = "alpha_generate"
    ML_TRAIN = "ml_train"
    BACKTEST = "backtest"


@dataclass
class TaskInfo:
    stage: PipelineStage
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    start_time: str = ""
    end_time: str = ""
    error: str = ""
    result_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
            "result_path": self.result_path,
        }


@dataclass
class PipelineResult:
    success: bool = False
    stages: dict[str, TaskInfo] = field(default_factory=dict)
    total_elapsed: float = 0.0
    alpha_count: int = 0
    model_path: str = ""
    backtest_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "total_elapsed": self.total_elapsed,
            "alpha_count": self.alpha_count,
            "model_path": self.model_path,
            "backtest_summary": self.backtest_summary,
        }


class PipelineBus:
    def __init__(self, config: ETFQuantConfig) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_future: Future | None = None
        self._tasks: dict[PipelineStage, TaskInfo] = {}
        self._callbacks: list[Callable[[TaskInfo], None]] = []
        self._cancelled = False
        for stage in PipelineStage:
            self._tasks[stage] = TaskInfo(stage=stage)

    @property
    def tasks(self) -> dict[PipelineStage, TaskInfo]:
        return self._tasks

    def add_callback(self, callback: Callable[[TaskInfo], None]) -> None:
        self._callbacks.append(callback)

    def _notify(self, task: TaskInfo) -> None:
        for cb in self._callbacks:
            try:
                cb(task)
            except Exception:
                pass

    def _update_task(self, stage: PipelineStage, status: TaskStatus, progress: float = 0.0, message: str = "", **kwargs: Any) -> None:
        task = self._tasks[stage]
        task.status = status
        task.progress = progress
        task.message = message
        if status == TaskStatus.RUNNING and not task.start_time:
            task.start_time = datetime.now().isoformat()
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.end_time = datetime.now().isoformat()
        for k, v in kwargs.items():
            setattr(task, k, v)
        self._notify(task)

    def cancel(self) -> None:
        self._cancelled = True
        if self._current_future and not self._current_future.done():
            self._current_future.cancel()

    def run_full_pipeline(self, etf_codes: list[str] | None = None) -> Future:
        self._cancelled = False
        for stage in PipelineStage:
            self._tasks[stage] = TaskInfo(stage=stage)
        future = self._executor.submit(self._execute_full_pipeline, etf_codes)
        self._current_future = future
        return future

    def run_stage(self, stage: PipelineStage, **kwargs: Any) -> Future:
        self._cancelled = False
        self._tasks[stage] = TaskInfo(stage=stage)
        future = self._executor.submit(self._execute_stage, stage, **kwargs)
        self._current_future = future
        return future

    def _execute_full_pipeline(self, etf_codes: list[str] | None = None) -> PipelineResult:
        result = PipelineResult()
        start_time = time.time()

        try:
            self._execute_data_check(etf_codes)
            if self._cancelled:
                return result

            alpha_pool = self._execute_alpha_generate(etf_codes)
            if self._cancelled:
                return result

            model_pkg = self._execute_ml_train(alpha_pool, etf_codes)
            if self._cancelled:
                return result

            backtest_result = self._execute_backtest(model_pkg, etf_codes)

            result.success = True
            result.alpha_count = len(alpha_pool.filter_valid()) if alpha_pool else 0
            result.model_path = model_pkg.metadata.get("save_path", "") if model_pkg and model_pkg.metadata else ""
            if backtest_result:
                result.backtest_summary = backtest_result.to_summary_dict()

        except Exception as exc:
            logger.error("Pipeline 执行失败: %s\n%s", exc, traceback.format_exc())
            result.success = False

        result.stages = dict(self._tasks)
        result.total_elapsed = time.time() - start_time
        return result

    def _execute_stage(self, stage: PipelineStage, **kwargs: Any) -> Any:
        if stage == PipelineStage.DATA_CHECK:
            return self._execute_data_check(kwargs.get("etf_codes"))
        elif stage == PipelineStage.ALPHA_GENERATE:
            return self._execute_alpha_generate(kwargs.get("etf_codes"))
        elif stage == PipelineStage.ML_TRAIN:
            return self._execute_ml_train(kwargs.get("alpha_pool"), kwargs.get("etf_codes"))
        elif stage == PipelineStage.BACKTEST:
            return self._execute_backtest(kwargs.get("model_package"), kwargs.get("etf_codes"))
        return None

    def _execute_data_check(self, etf_codes: list[str] | None = None) -> None:
        from etfquant.data.bridge import DataBridge

        self._update_task(PipelineStage.DATA_CHECK, TaskStatus.RUNNING, 0.0, "检查数据...")
        bridge = DataBridge(self._config.data)
        codes = etf_codes or bridge.list_etf_codes()
        if not codes:
            self._update_task(PipelineStage.DATA_CHECK, TaskStatus.FAILED, 0.0, "未找到ETF数据", error="数据目录为空")
            return
        summary = bridge.get_data_summary()
        self._update_task(
            PipelineStage.DATA_CHECK, TaskStatus.COMPLETED, 1.0,
            f"数据检查完成: {summary['total_etf_count']} 只ETF",
            result_path=str(self._config.data.data_root),
        )

    def _execute_alpha_generate(self, etf_codes: list[str] | None = None) -> Any:
        from etfquant.alpha.calculator import ETFAlphaCalculator, PresetFactors
        from etfquant.data.bridge import DataBridge

        self._update_task(PipelineStage.ALPHA_GENERATE, TaskStatus.RUNNING, 0.0, "生成因子...")
        bridge = DataBridge(self._config.data)
        calculator = ETFAlphaCalculator(bridge, self._config.alpha)
        self._update_task(PipelineStage.ALPHA_GENERATE, TaskStatus.RUNNING, 0.3, "评估预置因子...")
        pool = PresetFactors.evaluate_all(calculator, self._config.alpha)
        valid_count = len(pool.filter_valid())
        save_path = Path(self._config.alpha.save_path) / "alpha_pool.parquet"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pool.save(str(save_path))
        self._update_task(
            PipelineStage.ALPHA_GENERATE, TaskStatus.COMPLETED, 1.0,
            f"因子生成完成: {valid_count} 个有效因子",
            result_path=str(save_path),
        )
        return pool

    def _execute_ml_train(self, alpha_pool: Any = None, etf_codes: list[str] | None = None) -> Any:
        from etfquant.data.bridge import DataBridge
        from etfquant.ml.trainer import ETFDataSource, FeatureEngineer, ModelTrainer

        self._update_task(PipelineStage.ML_TRAIN, TaskStatus.RUNNING, 0.0, "准备训练数据...")
        bridge = DataBridge(self._config.data)
        ds = ETFDataSource(bridge, self._config.ml)
        fe = FeatureEngineer(ds, self._config.ml)

        codes = etf_codes or ds.get_stock_list()
        self._update_task(PipelineStage.ML_TRAIN, TaskStatus.RUNNING, 0.3, f"构建特征 ({len(codes)} 只ETF)...")
        X, y, dates = fe.build_dataset(codes)
        if X.empty:
            self._update_task(PipelineStage.ML_TRAIN, TaskStatus.FAILED, 0.0, "训练数据为空", error="无法构建特征数据集")
            return None

        self._update_task(PipelineStage.ML_TRAIN, TaskStatus.RUNNING, 0.6, f"训练模型 ({len(X)} 样本)...")
        trainer = ModelTrainer(self._config.ml)
        model_pkg = trainer.train(X, y, dates)

        save_path = Path(self._config.ml.save_path) / "etf_model.ptp"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_pkg.save(str(save_path))
        if model_pkg.metadata:
            model_pkg.metadata["save_path"] = str(save_path)

        self._update_task(
            PipelineStage.ML_TRAIN, TaskStatus.COMPLETED, 1.0,
            f"模型训练完成: {model_pkg.metadata.get('train_samples', 0)} 样本",
            result_path=str(save_path),
        )
        return model_pkg

    def _execute_backtest(self, model_package: Any = None, etf_codes: list[str] | None = None) -> Any:
        from etfquant.backtest.engine import ETFBacktester
        from etfquant.data.bridge import DataBridge

        self._update_task(PipelineStage.BACKTEST, TaskStatus.RUNNING, 0.0, "运行回测...")
        bridge = DataBridge(self._config.data)
        backtester = ETFBacktester(bridge, self._config.backtest)

        codes = etf_codes or [self._config.backtest.benchmark]
        if not codes:
            self._update_task(PipelineStage.BACKTEST, TaskStatus.FAILED, 0.0, "无回测标的", error="未指定ETF代码")
            return None

        code = codes[0]
        self._update_task(PipelineStage.BACKTEST, TaskStatus.RUNNING, 0.5, f"回测 {code}...")

        if model_package is not None:
            result = backtester.run_model_backtest(model_package, code)
        else:
            price_df = bridge.load_etf_daily(code)
            if price_df.empty:
                self._update_task(PipelineStage.BACKTEST, TaskStatus.FAILED, 0.0, f"无数据: {code}", error="数据为空")
                return None
            signals = pd.DataFrame(index=price_df.index)
            signals["signal"] = 0
            ma_short = price_df["close"].rolling(5).mean()
            ma_long = price_df["close"].rolling(20).mean()
            signals.loc[ma_short > ma_long, "signal"] = 1
            signals.loc[ma_short < ma_long, "signal"] = -1
            result = backtester.run_signal_backtest(signals, code, "MA5/MA20均线策略")

        self._update_task(
            PipelineStage.BACKTEST, TaskStatus.COMPLETED, 1.0,
            f"回测完成: 年化收益 {result.annual_return:.2%}",
        )
        return result

    def get_status(self) -> dict[str, Any]:
        return {stage.value: task.to_dict() for stage, task in self._tasks.items()}

    def save_status(self, path: str = "output/pipeline_status.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_status(), f, ensure_ascii=False, indent=2)
