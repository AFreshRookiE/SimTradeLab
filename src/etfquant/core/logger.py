from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["get_logger"]

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_handler_cache: dict[str, logging.Handler] = {}


def get_logger(name: str = "etfquant", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if name not in _handler_cache:
        fh = logging.FileHandler(_LOG_DIR / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        _handler_cache[name] = fh

    return logger
