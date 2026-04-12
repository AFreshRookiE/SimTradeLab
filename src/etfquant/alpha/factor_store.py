from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from etfquant.core.logger import get_logger

__all__ = ["FactorStore"]

logger = get_logger("etfquant.alpha.store")


class FactorStore:
    def __init__(self, db_path: str = "output/factors/factor_store.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS factors (
                name TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                description TEXT DEFAULT '',
                ic REAL DEFAULT 0.0,
                rank_ic REAL DEFAULT 0.0,
                icir REAL DEFAULT 0.0,
                is_valid INTEGER DEFAULT 0,
                category TEXT DEFAULT 'custom',
                params TEXT DEFAULT '{}',
                created_at TEXT DEFAULT '',
                updated_at TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_factors_category ON factors(category);
            CREATE INDEX IF NOT EXISTS idx_factors_valid ON factors(is_valid);
            CREATE INDEX IF NOT EXISTS idx_factors_ic ON factors(ic);
        """)
        self._conn.commit()

    def upsert(self, factor: Any) -> None:
        now = datetime.now().isoformat()
        params_json = json.dumps(factor.params if hasattr(factor, "params") else {}, ensure_ascii=False)
        self._conn.execute(
            """INSERT INTO factors (name, expression, description, ic, rank_ic, icir, is_valid, category, params, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   expression=excluded.expression,
                   description=excluded.description,
                   ic=excluded.ic,
                   rank_ic=excluded.rank_ic,
                   icir=excluded.icir,
                   is_valid=excluded.is_valid,
                   category=excluded.category,
                   params=excluded.params,
                   updated_at=excluded.updated_at
            """,
            (
                factor.name,
                factor.expression,
                factor.description if hasattr(factor, "description") else "",
                factor.ic if hasattr(factor, "ic") else 0.0,
                factor.rank_ic if hasattr(factor, "rank_ic") else 0.0,
                factor.icir if hasattr(factor, "icir") else 0.0,
                1 if (factor.is_valid if hasattr(factor, "is_valid") else False) else 0,
                factor.category if hasattr(factor, "category") else "custom",
                params_json,
                now,
                now,
            ),
        )
        self._conn.commit()

    def list_all(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name, expression, description, ic, rank_ic, icir, is_valid, category, params, created_at, updated_at FROM factors ORDER BY ABS(ic) DESC"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_valid(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name, expression, description, ic, rank_ic, icir, is_valid, category, params, created_at, updated_at FROM factors WHERE is_valid=1 ORDER BY ABS(ic) DESC"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_by_category(self, category: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name, expression, description, ic, rank_ic, icir, is_valid, category, params, created_at, updated_at FROM factors WHERE category=? ORDER BY ABS(ic) DESC",
            (category,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT name, expression, description, ic, rank_ic, icir, is_valid, category, params, created_at, updated_at FROM factors WHERE name=?",
            (name,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def delete(self, name: str) -> bool:
        cursor = self._conn.execute("DELETE FROM factors WHERE name=?", (name,))
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_invalid(self) -> int:
        cursor = self._conn.execute("DELETE FROM factors WHERE is_valid=0")
        self._conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM factors").fetchone()
        return row[0]

    def count_valid(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM factors WHERE is_valid=1").fetchone()
        return row[0]

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "name": row["name"],
            "expression": row["expression"],
            "description": row["description"],
            "ic": row["ic"],
            "rank_ic": row["rank_ic"],
            "icir": row["icir"],
            "is_valid": bool(row["is_valid"]),
            "category": row["category"],
            "params": json.loads(row["params"]) if row["params"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def close(self) -> None:
        self._conn.close()
