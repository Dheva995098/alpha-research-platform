"""Seed curated good-alpha examples as positive ML training signals."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.expression_normalizer import normalize_brain_expression
from backend.core.simulation_settings import merge_simulation_settings
from backend.ml.auto_learner import AutoLearningService
from backend.models import Result, SessionLocal, init_db


GOOD_ALPHA_EXAMPLES = [
    {
        "id": "good-seed-001",
        "expression": "-rank(ts_sum(open < close, 50) / ts_sum(open < close, 250))",
        "settings": {"decay": 4, "truncation": 0.08, "neutralization": "SUBINDUSTRY"},
    },
    {
        "id": "good-seed-002",
        "expression": "rank(ts_mean(-ts_delta(close / open, 150), 79))",
        "settings": {"decay": 2, "truncation": 0.08, "neutralization": "SUBINDUSTRY"},
    },
    {
        "id": "good-seed-003",
        "expression": "ts_corr(open, close, 352)",
        "settings": {"decay": 4, "truncation": 0.08, "neutralization": "SUBINDUSTRY"},
    },
    {
        "id": "good-seed-004",
        "expression": "rank(rank(volume / adv20) * rank(ts_mean(open - close, 8)))",
        "settings": {"decay": 62, "truncation": 0.08, "neutralization": "SUBINDUSTRY"},
    },
    {
        "id": "good-seed-005",
        "expression": (
            "group_neutralize(trade_when(pcr_oi_270 < 1, "
            "ts_backfill(implied_volatility_call_270 - implied_volatility_put_270, 10)))"
        ),
        "settings": {"decay": 4, "truncation": 0.08, "neutralization": "SUBINDUSTRY"},
    },
    {
        "id": "good-seed-006",
        "expression": "rank(ts_mean(open - close, 8) + ts_rank(operating_income / cap, 252))",
        "settings": {"decay": 4, "truncation": 0.08, "neutralization": "SUBINDUSTRY"},
    },
    {
        "id": "good-seed-007",
        "expression": "rank(-ts_zscore(close, 6))",
        "settings": {"decay": 4, "truncation": 0.08, "neutralization": "SUBINDUSTRY"},
    },
    {
        "id": "good-seed-008",
        "expression": (
            "trade_when(pcr_oi_720 < 0.4, "
            "(implied_volatility_call_720 - implied_volatility_put_720), -1)"
        ),
        "settings": {"decay": 4, "truncation": 0.09, "neutralization": "SECTOR"},
    },
]


def seed_examples() -> int:
    """Insert or update curated positive examples and retrain once."""
    init_db()
    db = SessionLocal()
    try:
        seeded = 0
        for item in GOOD_ALPHA_EXAMPLES:
            settings = merge_simulation_settings(
                {
                    "instrumentType": "EQUITY",
                    "region": "USA",
                    "universe": "TOP3000",
                    "language": "FASTEXPR",
                    "delay": 1,
                    "pasteurization": "ON",
                    "nanHandling": "OFF",
                    "unitHandling": "VERIFY",
                    "maxTrade": "OFF",
                    "maxPosition": "OFF",
                    **item["settings"],
                }
            )
            expression = normalize_brain_expression(item["expression"])
            alpha_id = f"training-seed-{item['id']}"
            result = db.query(Result).filter(Result.brain_alpha_id == alpha_id).first()
            if result is None:
                result = Result(brain_alpha_id=alpha_id)
                db.add(result)
            result.expression = expression
            result.sharpe = None
            result.fitness = None
            result.turnover = None
            result.self_correlation = None
            result.all_checks_passed = True
            result.raw_metrics = {
                "source": "training_seed",
                "seed_id": item["id"],
                "settings": settings,
                "grade": "GOOD",
                "all_checks_passed": True,
                "copy_policy": "training_signal_only_do_not_copy",
                "checks": [{"name": "CURATED_GOOD_ALPHA", "result": "PASS"}],
            }
            result.human_approved = True
            result.submitted_to_brain = False
            seeded += 1

        db.commit()
        summary = AutoLearningService(db).run_once(limit=500, min_examples=5)
        print(
            f"seeded={seeded} trained={summary.get('trained')} "
            f"examples={summary.get('training', {}).get('example_count')} "
            f"training_seed_count={summary.get('training_seed_count')}"
        )
        return seeded
    finally:
        db.close()


if __name__ == "__main__":
    seed_examples()
