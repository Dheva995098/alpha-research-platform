"""Objective evaluator with failure diagnosis (self-improving loop Pattern A).

Without a machine-checkable definition of "good" the loop cannot select, rank, or
know whether it improved. This module is that fitness function. It returns a
``Verdict`` with:

- ``is_ok``    : did the attempt clear the hard gates (a confirmed win)?
- ``failures``: *why* it fell short, as stable tags the refiner can act on.
- ``score``   : a continuous composite score, defined even on a pass, so the
                "best attempted" is always rankable and improvement is measurable.
- ``outcome`` : ``win`` / ``near`` / ``fail`` / ``error``. A ``near`` is a
                repairable near-miss -> the cheap deterministic refiner targets it
                before paying for a fresh generation.

Gate thresholds live in :class:`GateConfig` (config, not code) so they can be
tightened as the system gets stronger (curriculum learning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Failure tags (aligned with the REAL WorldQuant BRAIN is.checks `name` strings) --------
LOW_SHARPE = "LOW_SHARPE"
LOW_FITNESS = "LOW_FITNESS"
HIGH_TURNOVER = "HIGH_TURNOVER"
LOW_TURNOVER = "LOW_TURNOVER"  # distinct defect: trades too little (< ~1%), NOT a turnover penalty
HIGH_SELF_CORRELATION = "HIGH_SELF_CORRELATION"
HIGH_PROD_CORRELATION = "HIGH_PROD_CORRELATION"
LOW_SUB_UNIVERSE_SHARPE = "LOW_SUB_UNIVERSE_SHARPE"
CONCENTRATED_WEIGHT = "CONCENTRATED_WEIGHT"  # a single instrument holds too much book
COVERAGE_FAIL = "COVERAGE_FAIL"  # NaN / coverage / unit problems
LOW_RETURNS = "LOW_RETURNS"
LOW_2Y_SHARPE = "LOW_2Y_SHARPE"
UNITS = "UNITS"
FAILED_CHECKS = "FAILED_CHECKS"

# BRAIN is.checks `name` (lowercased) -> our stable failure tag.
# Names confirmed from BRAIN automation repos (AlphaPower CheckType, WQOS, Brainiac schemas).
_CHECK_TAGS = {
    "low_sharpe": LOW_SHARPE,
    "is_sharpe": LOW_SHARPE,
    "rank_sharpe": LOW_SHARPE,
    "is_ladder_sharpe": LOW_2Y_SHARPE,
    "low_2y_sharpe": LOW_2Y_SHARPE,
    "low_fitness": LOW_FITNESS,
    "high_turnover": HIGH_TURNOVER,
    "low_turnover": LOW_TURNOVER,
    "low_sub_universe_sharpe": LOW_SUB_UNIVERSE_SHARPE,
    "sub_universe_sharpe": LOW_SUB_UNIVERSE_SHARPE,
    "super_universe_sharpe": LOW_SUB_UNIVERSE_SHARPE,
    "low_robust_universe_sharpe": LOW_SUB_UNIVERSE_SHARPE,
    "low_robust_universe_returns": LOW_RETURNS,
    "low_after_cost_illiquid_universe_sharpe": LOW_SUB_UNIVERSE_SHARPE,
    "low_returns": LOW_RETURNS,
    "self_correlation": HIGH_SELF_CORRELATION,
    "high_self_correlation": HIGH_SELF_CORRELATION,
    "prod_correlation": HIGH_PROD_CORRELATION,
    "power_pool_correlation": HIGH_PROD_CORRELATION,
    "concentrated_weight": CONCENTRATED_WEIGHT,
    "high_weight": CONCENTRATED_WEIGHT,
    "low_coverage": COVERAGE_FAIL,
    "coverage": COVERAGE_FAIL,
    "units": UNITS,
}

# Eligibility / scoring-multiplier flags — a non-PASS here is NOT a quality rejection.
_ELIGIBILITY_CHECKS = {
    "matches_competition",
    "matches_pyramid",
    "matches_themes",
}


@dataclass(frozen=True)
class GateConfig:
    """Hard gates + the band that makes a failure count as a repairable near-miss."""

    # Hard gates (a "win" must clear all of these). Defaults = standard USA TOP3000
    # Delay-1 submission bar (Sharpe>1.25, Fitness>=1.0, 1%<=turnover<=70%, self-corr<0.7).
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    max_turnover: float = 0.70
    min_turnover: float = 0.01
    max_self_correlation: float = 0.70
    max_prod_correlation: float = 0.70
    require_checks_passed: bool = True

    # Near-miss band: signal is present and the gap looks deterministically fixable.
    near_sharpe: float = 0.90          # real signal threshold
    near_fitness: float = 0.60         # fitness floor for "worth repairing"
    near_turnover_slack: float = 0.30  # turnover <= max + slack -> decay can rescue it
    near_self_corr_slack: float = 0.15

    @classmethod
    def relaxed(cls) -> "GateConfig":
        """A looser bar (useful early, before enough live data exists)."""
        return cls(min_sharpe=1.0, min_fitness=0.8)


@dataclass(frozen=True)
class Verdict:
    """The evaluator's judgment of one attempt."""

    is_ok: bool
    failures: List[str]
    score: float
    outcome: str  # win | near | fail | error
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_win(self) -> bool:
        return self.outcome == "win"

    @property
    def is_near(self) -> bool:
        return self.outcome == "near"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_ok": self.is_ok,
            "failures": list(self.failures),
            "score": self.score,
            "outcome": self.outcome,
        }


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "pass", "passed"}:
            return True
        if token in {"false", "0", "no", "fail", "failed"}:
            return False
        return None
    return bool(value)


def _checks(metrics: Dict[str, Any]) -> Optional[list]:
    """Pull the BRAIN checks list from common locations in a metrics blob."""
    checks = metrics.get("checks")
    if isinstance(checks, list):
        return checks
    raw = metrics.get("raw_metrics")
    if isinstance(raw, dict):
        if isinstance(raw.get("checks"), list):
            return raw["checks"]
        is_block = raw.get("is")
        if isinstance(is_block, dict) and isinstance(is_block.get("checks"), list):
            return is_block["checks"]
    return None


def _failed_check_tags(metrics: Dict[str, Any]) -> List[str]:
    """Map any FAIL-ing BRAIN checks to our stable failure tags.

    Only a 'FAIL' result counts (PENDING/WAITING/WARNING are not failures), and
    MATCHES_* eligibility flags are never treated as quality failures.
    """
    checks = _checks(metrics)
    if not checks:
        return []
    tags: List[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if str(check.get("result") or "").strip().lower() != "fail":
            continue
        name = str(check.get("name") or "").strip().lower()
        if name in _ELIGIBILITY_CHECKS:
            continue
        tags.append(_CHECK_TAGS.get(name, FAILED_CHECKS))
    return tags


def composite_score(metrics: Dict[str, Any]) -> float:
    """Continuous fitness score, defined even on a pass so wins are still rankable.

    Mirrors the source system's shape: reward sharpe + fitness, penalise turnover
    and self-correlation overshoot, with a small bonus for clean BRAIN checks.
    """
    sharpe = _coerce_float(metrics.get("sharpe")) or 0.0
    fitness = _coerce_float(metrics.get("fitness")) or 0.0
    turnover = _coerce_float(metrics.get("turnover"))
    self_corr = _coerce_float(metrics.get("self_correlation"))
    checks_passed = _coerce_bool(metrics.get("all_checks_passed"))

    score = sharpe * 1.0 + fitness * 0.5
    if turnover is not None:
        score -= max(turnover - 0.5, 0.0) * 1.0
    if self_corr is not None:
        score -= max(self_corr - 0.5, 0.0) * 0.5
    if checks_passed:
        score += 0.10
    return round(score, 4)


def diagnose(metrics: Dict[str, Any], config: Optional[GateConfig] = None) -> List[str]:
    """Return the ordered, de-duplicated set of failure tags for these metrics."""
    config = config or GateConfig()
    failures: List[str] = []

    sharpe = _coerce_float(metrics.get("sharpe"))
    fitness = _coerce_float(metrics.get("fitness"))
    turnover = _coerce_float(metrics.get("turnover"))
    self_corr = _coerce_float(metrics.get("self_correlation"))

    if sharpe is not None and sharpe < config.min_sharpe:
        failures.append(LOW_SHARPE)
    if fitness is not None and fitness < config.min_fitness:
        failures.append(LOW_FITNESS)
    if turnover is not None and turnover > config.max_turnover:
        failures.append(HIGH_TURNOVER)
    elif turnover is not None and 0.0 < turnover < config.min_turnover:
        failures.append(LOW_TURNOVER)
    if self_corr is not None and self_corr > config.max_self_correlation:
        failures.append(HIGH_SELF_CORRELATION)

    # Add anything the BRAIN checks list flagged (e.g. LOW_SUB_UNIVERSE_SHARPE, coverage).
    for tag in _failed_check_tags(metrics):
        failures.append(tag)

    checks_passed = _coerce_bool(metrics.get("all_checks_passed"))
    if config.require_checks_passed and checks_passed is False and FAILED_CHECKS not in failures:
        # When a checks list is present we already captured the specific real failures
        # above (eligibility flags excluded); only fall back to the generic tag when no
        # per-check detail is available at all.
        if _checks(metrics) is None and not _failed_check_tags(metrics):
            failures.append(FAILED_CHECKS)

    # Stable de-dup preserving first occurrence.
    seen = set()
    ordered: List[str] = []
    for tag in failures:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def _is_win(metrics: Dict[str, Any], failures: List[str], config: GateConfig) -> bool:
    sharpe = _coerce_float(metrics.get("sharpe"))
    fitness = _coerce_float(metrics.get("fitness"))
    if sharpe is None or fitness is None:
        return False
    if failures:
        return False
    checks_passed = _coerce_bool(metrics.get("all_checks_passed"))
    if config.require_checks_passed and checks_passed is False:
        return False
    return True


def _is_near(metrics: Dict[str, Any], config: GateConfig) -> bool:
    """A repairable near-miss: signal is present and every gap is modest/fixable."""
    sharpe = _coerce_float(metrics.get("sharpe"))
    if sharpe is None or sharpe < config.near_sharpe:
        return False
    fitness = _coerce_float(metrics.get("fitness"))
    turnover = _coerce_float(metrics.get("turnover"))
    self_corr = _coerce_float(metrics.get("self_correlation"))

    if fitness is not None and fitness < config.near_fitness:
        return False
    if turnover is not None and turnover > config.max_turnover + config.near_turnover_slack:
        return False
    if self_corr is not None and self_corr > config.max_self_correlation + config.near_self_corr_slack:
        return False
    return True


def evaluate(metrics: Dict[str, Any], config: Optional[GateConfig] = None) -> Verdict:
    """Evaluate a metrics blob into a Verdict {is_ok, failures, score, outcome}."""
    config = config or GateConfig()
    metrics = metrics or {}

    # A genuine execution error (no usable metrics) is its own outcome.
    if metrics.get("error") or metrics.get("status") in {"failed", "error"}:
        if _coerce_float(metrics.get("sharpe")) is None:
            return Verdict(False, [FAILED_CHECKS], composite_score(metrics), "error", metrics)

    failures = diagnose(metrics, config)
    score = composite_score(metrics)
    is_ok = _is_win(metrics, failures, config)

    if is_ok:
        outcome = "win"
    elif _is_near(metrics, config):
        outcome = "near"
    else:
        outcome = "fail"

    return Verdict(is_ok=is_ok, failures=failures, score=score, outcome=outcome, metrics=metrics)


def metrics_from_result(result: Any) -> Dict[str, Any]:
    """Adapt a ``Result`` ORM row (or any object with the same attrs) into a metrics blob.

    Standalone so the evaluator stays free of heavy ML/service imports.
    """
    raw = getattr(result, "raw_metrics", None)
    raw = raw if isinstance(raw, dict) else {}
    settings = None
    simulation = getattr(result, "simulation", None)
    if simulation is not None:
        settings = getattr(simulation, "settings", None)
    if not isinstance(settings, dict):
        settings = raw.get("settings") if isinstance(raw.get("settings"), dict) else {}

    return {
        "sharpe": getattr(result, "sharpe", None),
        "fitness": getattr(result, "fitness", None),
        "turnover": getattr(result, "turnover", None),
        "self_correlation": getattr(result, "self_correlation", None),
        "all_checks_passed": getattr(result, "all_checks_passed", None),
        "raw_metrics": raw,
        "checks": _checks({"raw_metrics": raw}),
        "settings": settings,
    }


def evaluate_result(result: Any, config: Optional[GateConfig] = None) -> Verdict:
    """Convenience: evaluate a ``Result`` ORM row directly."""
    return evaluate(metrics_from_result(result), config)
