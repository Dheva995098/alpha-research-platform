"""Import an exported seed JSON into the configured SQLAlchemy database."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Integer, inspect, text
from sqlalchemy.dialects.postgresql import JSON as PG_JSON
from sqlalchemy.sql.sqltypes import JSON


TABLE_ORDER = [
    "accounts",
    "simulations",
    "results",
    "alpha_registry",
    "leaderboard_alphas",
    "data_fields",
    "alert_configs",
    "ml_models",
]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_datetime(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return value


def _coerce_row(table, row: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(row)
    for column in table.columns:
        if column.name not in coerced:
            continue
        value = coerced[column.name]
        if value is None:
            continue
        if isinstance(column.type, DateTime):
            coerced[column.name] = _parse_datetime(value)
        elif isinstance(column.type, (JSON, PG_JSON)) and isinstance(value, str):
            try:
                coerced[column.name] = json.loads(value)
            except json.JSONDecodeError:
                pass
        elif isinstance(column.type, Integer) and value == "":
            coerced[column.name] = None
    return coerced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=".tmp/local_seed.json", help="Seed JSON exported by export_local_seed.py")
    parser.add_argument("--database-url", help="Target DATABASE_URL. Defaults to current environment.")
    parser.add_argument("--keep-existing", action="store_true", help="Append rows instead of clearing known tables first")
    args = parser.parse_args()

    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url

    seed_path = Path(args.seed)
    if not seed_path.exists():
        raise SystemExit(f"Seed file not found: {seed_path}")

    from backend.models import Base, engine, init_db

    init_db()
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    tables = {table.name: table for table in Base.metadata.sorted_tables}
    imported = 0

    with engine.begin() as connection:
        if not args.keep_existing:
            for table_name in reversed(TABLE_ORDER):
                table = tables.get(table_name)
                if table is not None:
                    connection.execute(table.delete())

        for table_name in TABLE_ORDER:
            table = tables.get(table_name)
            rows = payload.get("tables", {}).get(table_name, [])
            if table is None or not rows:
                continue
            connection.execute(table.insert(), [_coerce_row(table, row) for row in rows])
            imported += len(rows)

        if engine.dialect.name == "postgresql":
            inspector = inspect(connection)
            for table_name in TABLE_ORDER:
                if table_name not in inspector.get_table_names():
                    continue
                connection.execute(
                    text(
                        """
                        SELECT setval(
                            pg_get_serial_sequence(:table_name, 'id'),
                            COALESCE((SELECT MAX(id) FROM {table}), 1),
                            true
                        )
                        """.format(table=table_name)
                    ),
                    {"table_name": table_name},
                )

    print(f"Imported {imported} rows into {engine.url.render_as_string(hide_password=True)}")


if __name__ == "__main__":
    main()
