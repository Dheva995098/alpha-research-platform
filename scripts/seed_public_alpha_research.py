"""Seed public, metrics-backed alpha research as ML training signals.

This importer uses public sources only as learning examples. Seeded rows are
tagged so the generator learns patterns and settings without treating the
source expressions as copy templates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Iterable, List

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.expression_normalizer import normalize_brain_expression
from backend.core.simulation_settings import merge_simulation_settings
from backend.ml.auto_learner import AutoLearningService
from backend.models import Result, SessionLocal, init_db


JGLAZAR_SUBMITTED_ALPHAS_URL = (
    "https://raw.githubusercontent.com/jglazar/notes/refs/heads/main/"
    "quant_interview/submitted_alphas.md"
)


@dataclass(frozen=True)
class PublicAlphaSeed:
    """One public alpha example with reported simulation settings and metrics."""

    seed_id: str
    expression: str
    source_name: str
    source_url: str
    region: str
    universe: str
    decay: float
    delay: float
    truncation: float
    neutralization: str
    sharpe: float
    turnover: float
    fitness: float
    returns_pct: float
    drawdown_pct: float
    margin_bps: float


SUBMITTED_ALPHA_RE = re.compile(
    r"(USA|CHN),\s*(TOP\d+),\s*Decay\s*([\d.]+),\s*Delay\s*([\d.]+),\s*"
    r"Truncation\s*([\d.]+),\s*Neutralization\s*([^\\\n]+)\\?\s*\n?\s*"
    r"Sharpe\s*([-\d.]+),\s*Turnover\s*([\d.]+)%?,\s*Fitness\s*([-\d.]+),\s*"
    r"Returns\s*([-\d.]+)%?,\s*Drawdown\s*([\d.]+)%?,\s*Margin\s*([-\d.]+)\S*\s*"
    r"```\s*(.*?)\s*```",
    re.S,
)


def parse_jglazar_submitted_alphas(text: str) -> List[PublicAlphaSeed]:
    """Extract submitted alpha rows from the public markdown document."""
    seeds: List[PublicAlphaSeed] = []
    for index, match in enumerate(SUBMITTED_ALPHA_RE.findall(text or ""), start=1):
        (
            region,
            universe,
            decay,
            delay,
            truncation,
            neutralization,
            sharpe,
            turnover_pct,
            fitness,
            returns_pct,
            drawdown_pct,
            margin_bps,
            expression,
        ) = match
        seeds.append(
            PublicAlphaSeed(
                seed_id=f"jglazar-submitted-{index:03d}",
                expression=_clean_expression(expression),
                source_name="jglazar/notes submitted_alphas.md",
                source_url=JGLAZAR_SUBMITTED_ALPHAS_URL,
                region=region.upper(),
                universe=universe.upper(),
                decay=float(decay),
                delay=float(delay),
                truncation=float(truncation),
                neutralization=_normalize_setting_token(neutralization),
                sharpe=float(sharpe),
                turnover=float(turnover_pct) / 100.0,
                fitness=float(fitness),
                returns_pct=float(returns_pct),
                drawdown_pct=float(drawdown_pct),
                margin_bps=float(margin_bps),
            )
        )
    return seeds


def fetch_public_alpha_seeds(timeout: int = 20) -> List[PublicAlphaSeed]:
    """Download and parse public, metrics-backed alpha seeds."""
    response = requests.get(JGLAZAR_SUBMITTED_ALPHAS_URL, timeout=timeout)
    response.raise_for_status()
    return parse_jglazar_submitted_alphas(response.text)


def seed_public_research(seeds: Iterable[PublicAlphaSeed], train: bool = True) -> dict:
    """Upsert public research rows and optionally retrain the learner."""
    init_db()
    db = SessionLocal()
    try:
        seeded = 0
        updated = 0
        now = datetime.now(timezone.utc).isoformat()
        for seed in seeds:
            settings = merge_simulation_settings(
                {
                    "instrumentType": "EQUITY",
                    "region": seed.region,
                    "universe": seed.universe,
                    "language": "FASTEXPR",
                    "delay": _number_or_int(seed.delay),
                    "decay": _number_or_int(seed.decay),
                    "truncation": seed.truncation,
                    "neutralization": seed.neutralization,
                    "pasteurization": "ON",
                    "nanHandling": "OFF",
                    "unitHandling": "VERIFY",
                }
            )
            alpha_id = f"public-seed-{seed.seed_id}"
            expression = normalize_brain_expression(seed.expression)
            result = db.query(Result).filter(Result.brain_alpha_id == alpha_id).first()
            if result is None:
                result = Result(brain_alpha_id=alpha_id)
                db.add(result)
                seeded += 1
            else:
                updated += 1

            passes_quality = seed.sharpe >= 1.25 and seed.fitness >= 0.95 and seed.turnover <= 0.70
            result.expression = expression
            result.sharpe = seed.sharpe
            result.fitness = seed.fitness
            result.turnover = seed.turnover
            result.self_correlation = None
            result.all_checks_passed = passes_quality
            result.raw_metrics = {
                "source": "training_seed",
                "seed_id": seed.seed_id,
                "seed_kind": "public_metrics_backed_alpha",
                "source_name": seed.source_name,
                "source_url": seed.source_url,
                "research_observed_at": now,
                "settings": settings,
                "grade": _grade(seed.sharpe, seed.fitness),
                "all_checks_passed": passes_quality,
                "public_metrics": {
                    "returns_pct": seed.returns_pct,
                    "drawdown_pct": seed.drawdown_pct,
                    "margin_bps": seed.margin_bps,
                },
                "copy_policy": "training_signal_only_do_not_copy",
                "checks": [
                    {"name": "PUBLIC_SUBMITTED_ALPHA", "result": "PASS"},
                    {"name": "SHARPE_PUBLIC_REPORTED", "result": "PASS" if seed.sharpe >= 1.25 else "FAIL"},
                    {"name": "FITNESS_PUBLIC_REPORTED", "result": "PASS" if seed.fitness >= 0.95 else "FAIL"},
                    {"name": "TURNOVER_PUBLIC_REPORTED", "result": "PASS" if seed.turnover <= 0.70 else "FAIL"},
                ],
            }
            result.human_approved = True
            result.submitted_to_brain = False

        db.commit()
        summary = {
            "seeded": seeded,
            "updated": updated,
            "total": seeded + updated,
            "trained": False,
            "example_count": None,
            "positive_count": None,
            "negative_count": None,
            "training_seed_count": None,
        }
        if train:
            learning = AutoLearningService(db).run_once(limit=1000, min_examples=5)
            summary.update(
                {
                    "trained": bool(learning.get("trained")),
                    "example_count": learning.get("training", {}).get("example_count"),
                    "positive_count": learning.get("training", {}).get("positive_count"),
                    "negative_count": learning.get("training", {}).get("negative_count"),
                    "training_seed_count": learning.get("training_seed_count"),
                }
            )
        return summary
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", action="store_true", help="Only seed rows; do not retrain after import")
    args = parser.parse_args()

    seeds = fetch_public_alpha_seeds()
    summary = seed_public_research(seeds, train=not args.no_train)
    print(
        "public_alpha_research total={total} seeded={seeded} updated={updated} "
        "trained={trained} examples={example_count} positives={positive_count} "
        "negatives={negative_count} training_seed_count={training_seed_count}".format(**summary)
    )
    return 0


def _clean_expression(expression: str) -> str:
    lines = []
    for line in (expression or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _normalize_setting_token(value: str) -> str:
    normalized = str(value or "").strip().upper()
    return "NONE" if normalized in {"", "NON", "NONE"} else normalized


def _number_or_int(value: float) -> float | int:
    return int(value) if float(value).is_integer() else value


def _grade(sharpe: float, fitness: float) -> str:
    if sharpe >= 2.0 and fitness >= 1.5:
        return "SUPERIOR"
    if sharpe >= 1.5 and fitness >= 1.0:
        return "EXCELLENT"
    if sharpe >= 1.25 and fitness >= 0.95:
        return "GOOD"
    return "ABOVE_AVERAGE"


if __name__ == "__main__":
    raise SystemExit(main())
