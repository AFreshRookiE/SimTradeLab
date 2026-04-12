from __future__ import annotations

from pathlib import Path
from typing import Any

from etfquant.core.config import MLConfig
from etfquant.core.logger import get_logger

__all__ = ["StrategyService"]

logger = get_logger("etfquant.api.strategy")

_STRATEGY_DIR = Path("strategies")


class StrategyService:
    def __init__(self) -> None:
        _STRATEGY_DIR.mkdir(exist_ok=True)

    def list_strategies(self) -> list[dict[str, Any]]:
        result = []
        for f in _STRATEGY_DIR.glob("*.py"):
            content = f.read_text(encoding="utf-8")
            name = f.stem
            result.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "path": str(f),
                "size": len(content),
                "modified": f.stat().st_mtime,
            })
        return result

    def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        path = _STRATEGY_DIR / f"{strategy_id}.py"
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return {"id": strategy_id, "name": strategy_id, "content": content, "path": str(path)}

    def save_strategy(self, strategy_id: str, content: str) -> dict[str, str]:
        path = _STRATEGY_DIR / f"{strategy_id}.py"
        path.write_text(content, encoding="utf-8")
        return {"status": "ok", "path": str(path)}

    def delete_strategy(self, strategy_id: str) -> bool:
        path = _STRATEGY_DIR / f"{strategy_id}.py"
        if path.exists():
            path.unlink()
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
