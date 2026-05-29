"""Import submitted alpha examples from a text-based PDF into training data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.generation.pdf_importer import extract_alpha_examples_from_pdf
from backend.ml.service import MLRankingService
from backend.models import LeaderboardAlpha, SessionLocal, init_db


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to exported/submitted alpha PDF")
    parser.add_argument("--train", action="store_true", help="Train the ranker after import")
    args = parser.parse_args()

    init_db()
    result = extract_alpha_examples_from_pdf(args.path)
    print(f"pages={result.page_count} text_chars={result.extracted_text_chars} extracted={len(result.examples)}")
    for warning in result.warnings:
        print(f"warning: {warning}")

    if not result.examples:
        return 0

    db = SessionLocal()
    try:
        imported = 0
        for example in result.examples:
            existing = db.query(LeaderboardAlpha).filter(LeaderboardAlpha.expression == example.expression).first()
            if existing:
                continue
            db.add(
                LeaderboardAlpha(
                    expression=example.expression,
                    sharpe=example.sharpe,
                    fitness=example.fitness,
                    turnover=example.turnover,
                    self_correlation=example.self_correlation,
                    passes_checks=example.passes_checks,
                )
            )
            imported += 1
        db.commit()
        print(f"imported={imported}")

        if args.train:
            training = MLRankingService(db).train_from_db()
            print(
                "trained={trained} examples={examples} positives={positives} negatives={negatives} accuracy={accuracy} message={message}".format(
                    trained=training.trained,
                    examples=training.example_count,
                    positives=training.positive_count,
                    negatives=training.negative_count,
                    accuracy=training.accuracy,
                    message=training.message,
                )
            )
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
