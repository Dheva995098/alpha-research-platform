"""Export local SQLite training data to a portable JSON seed file."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


TABLES = [
    "accounts",
    "simulations",
    "results",
    "alpha_registry",
    "leaderboard_alphas",
    "data_fields",
    "alert_configs",
    "ml_models",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="alpha_research.db", help="Local SQLite database path")
    parser.add_argument("--out", default=".tmp/local_seed.json", help="Output JSON path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"SQLite database not found: {db_path}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"tables": {}}
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        existing_tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in TABLES:
            if table not in existing_tables:
                payload["tables"][table] = []
                continue
            rows = connection.execute(f'SELECT * FROM "{table}"').fetchall()
            payload["tables"][table] = [dict(row) for row in rows]

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Exported {sum(len(rows) for rows in payload['tables'].values())} rows to {output_path}")


if __name__ == "__main__":
    main()
