"""Seed user-curated perfect alphas as positive ML training examples."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.curated_alpha_seeds import upsert_curated_perfect_alpha_seeds
from backend.models import SessionLocal, init_db


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", action="store_true", help="Only seed rows; do not retrain after import")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        summary = upsert_curated_perfect_alpha_seeds(db, train=not args.no_train)
    finally:
        db.close()

    print(
        "curated_perfect_alphas total={total} seeded={seeded} updated={updated} "
        "skipped={skipped} trained={trained} examples={example_count} "
        "positives={positive_count} negatives={negative_count} "
        "training_seed_count={training_seed_count}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
