from __future__ import annotations

import threading
from datetime import date, datetime, time as dt_time
from typing import Any, Callable

from etfquant.core.config import ScheduleConfig
from etfquant.core.logger import get_logger

__all__ = ["AlphaScheduler"]

logger = get_logger("etfquant.alpha.scheduler")

_DAY_CN = {
    "Monday": "周一", "Tuesday": "周二", "Wednesday": "周三",
    "Thursday": "周四", "Friday": "周五", "Saturday": "周六", "Sunday": "周日",
}


class AlphaScheduler:
    def __init__(self, config: ScheduleConfig, task_func: Callable[[], Any]) -> None:
        self._config = config
        self._task_func = task_func
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status = "idle"
        self._last_run: str = ""
        self._next_run: str = ""
        self._last_run_date: str = ""

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def last_run(self) -> str:
        return self._last_run

    @property
    def next_run(self) -> str:
        return self._next_run

    def start(self) -> None:
        if self._running:
            logger.warning("调度器已在运行")
            return
        self._running = True
        self._paused = False
        self._stop_event.clear()
        self._status = "running"
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        days_cn = ", ".join(_DAY_CN.get(d, d) for d in self._config.days)
        logger.info("调度器已启动: %s-%s, 运行日=%s", self._config.start_time, self._config.end_time, days_cn)

    def stop(self) -> None:
        self._running = False
        self._paused = False
        self._stop_event.set()
        self._status = "stopped"
        logger.info("调度器已停止")

    def pause(self) -> None:
        if not self._running:
            return
        self._paused = True
        self._status = "paused"
        logger.info("调度器已暂停")

    def resume(self) -> None:
        if not self._running:
            return
        self._paused = False
        self._status = "running"
        logger.info("调度器已恢复")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._config.enabled:
                self._stop_event.wait(60)
                continue

            if self._paused:
                self._stop_event.wait(5)
                continue

            now = datetime.now()
            day_name = now.strftime("%A")

            if day_name not in self._config.days:
                self._next_run = f"下一个运行日（今日{_DAY_CN.get(day_name, day_name)}不在运行日）"
                self._stop_event.wait(60)
                continue

            try:
                start_parts = self._config.start_time.split(":")
                end_parts = self._config.end_time.split(":")
                start_t = dt_time(int(start_parts[0]), int(start_parts[1]))
                end_t = dt_time(int(end_parts[0]), int(end_parts[1]))
            except (ValueError, IndexError):
                logger.error("时间格式错误: start=%s, end=%s", self._config.start_time, self._config.end_time)
                self._stop_event.wait(300)
                continue

            current_t = now.time()
            today_str = date.today().isoformat()

            if start_t <= current_t <= end_t:
                if self._last_run_date == today_str:
                    self._next_run = "今日已执行，明日继续"
                    self._stop_event.wait(60)
                    continue

                self._next_run = "正在执行..."
                self._status = "executing"
                try:
                    logger.info("定时任务开始执行: %s", now.isoformat())
                    self._task_func()
                    self._last_run = now.isoformat()
                    self._last_run_date = today_str
                    logger.info("定时任务执行完成: %s", now.isoformat())
                except Exception as exc:
                    logger.error("定时任务执行失败: %s", exc)
                self._status = "running"
                self._stop_event.wait(300)
            else:
                if current_t < start_t:
                    self._next_run = f"今日 {self._config.start_time}"
                else:
                    self._next_run = f"明日 {self._config.start_time}"
                self._stop_event.wait(30)

    def get_status(self) -> dict[str, Any]:
        days_cn = [_DAY_CN.get(d, d) for d in self._config.days]
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "paused": self._paused,
            "status": self._status,
            "schedule": f"{self._config.start_time}-{self._config.end_time}",
            "days": self._config.days,
            "days_cn": days_cn,
            "last_run": self._last_run,
            "next_run": self._next_run,
        }
