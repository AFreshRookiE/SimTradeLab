from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from etfquant.core.config import BacktestConfig
from etfquant.core.logger import get_logger

__all__ = ["StrategyService"]

logger = get_logger("etfquant.api.strategy")

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\u4e00-\u9fff]+$")


class StrategyService:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        if config is not None:
            self._strategy_dir = Path(config.save_path) / "user_strategies"
        else:
            self._strategy_dir = Path("strategies")
        self._strategy_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, strategy_id: str) -> Path:
        if not _SAFE_ID_PATTERN.match(strategy_id):
            raise ValueError(f"非法策略名称: {strategy_id!r}（仅允许字母、数字、下划线、中文）")
        resolved = (self._strategy_dir / f"{strategy_id}.py").resolve()
        if not str(resolved).startswith(str(self._strategy_dir.resolve())):
            raise ValueError(f"策略名称越权访问: {strategy_id!r}")
        return resolved

    def list_strategies(self) -> list[dict[str, Any]]:
        result = []
        for f in self._strategy_dir.glob("*.py"):
            name = f.stem
            stat = f.stat()
            result.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
        return result

    def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        path = self._resolve_path(strategy_id)
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return {"id": strategy_id, "name": strategy_id, "content": content}

    def save_strategy(self, strategy_id: str, content: str) -> dict[str, str]:
        path = self._resolve_path(strategy_id)
        path.write_text(content, encoding="utf-8")
        logger.info("策略已保存: %s", strategy_id)
        return {"status": "ok", "path": str(path)}

    def delete_strategy(self, strategy_id: str) -> bool:
        path = self._resolve_path(strategy_id)
        if path.exists():
            path.unlink()
            logger.info("策略已删除: %s", strategy_id)
            return True
        return False

    def get_template(self, template_id: str) -> str:
        from etfquant.api.backtest import _STRATEGY_TEMPLATES
        return _STRATEGY_TEMPLATES.get(template_id, "")

    def list_templates(self) -> list[dict[str, str]]:
        from etfquant.api.backtest import _STRATEGY_TEMPLATES, _STRATEGY_DESC
        return [
            {"id": k, "name": v_name, "description": _STRATEGY_DESC.get(k, "")}
            for k, v_name in [
                ("ma_cross", "MA均线交叉"),
                ("momentum", "动量策略"),
                ("mean_reversion", "均值回归"),
                ("etf_premium", "ETF折溢价套利"),
            ]
        ]
