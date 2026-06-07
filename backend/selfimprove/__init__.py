"""Self-improving research loop for the Alpha Research Platform.

This package grafts the "self-improving / memory" engine onto the existing
generator + simulator + ranker stack. It closes the loop so that each attempt
is conditioned on the outcomes of previous attempts:

- evaluator.py  -> objective gates + failure diagnosis + continuous score (Pattern A)
- memory.py     -> persistent tried[]/failures[]/library[] (Patterns B, H)
- refiner.py    -> deterministic failure->fix repair before paying for regen (Pattern E)
- feedback.py   -> turn recent failures + tried set into generation context (Patterns C, D)

Nothing here changes model weights; it is in-context / agentic learning with
persistent memory layered on top of whatever ML already exists.
"""
from backend.selfimprove.evaluator import (
    CONCENTRATED_WEIGHT,
    COVERAGE_FAIL,
    FAILED_CHECKS,
    HIGH_PROD_CORRELATION,
    HIGH_SELF_CORRELATION,
    HIGH_TURNOVER,
    LOW_2Y_SHARPE,
    LOW_FITNESS,
    LOW_RETURNS,
    LOW_SHARPE,
    LOW_SUB_UNIVERSE_SHARPE,
    LOW_TURNOVER,
    UNITS,
    GateConfig,
    Verdict,
    composite_score,
    diagnose,
    evaluate,
    evaluate_result,
    metrics_from_result,
)

__all__ = [
    "GateConfig",
    "Verdict",
    "evaluate",
    "evaluate_result",
    "diagnose",
    "composite_score",
    "metrics_from_result",
    "LOW_SHARPE",
    "LOW_FITNESS",
    "HIGH_TURNOVER",
    "LOW_TURNOVER",
    "HIGH_SELF_CORRELATION",
    "HIGH_PROD_CORRELATION",
    "LOW_SUB_UNIVERSE_SHARPE",
    "CONCENTRATED_WEIGHT",
    "COVERAGE_FAIL",
    "LOW_RETURNS",
    "LOW_2Y_SHARPE",
    "UNITS",
    "FAILED_CHECKS",
]
