"""Persistent attempt memory + auto-growing win library (Patterns B and H).

State that survives across runs is the difference between a system that retries
the same dead-ends forever and one that accumulates experience. ``AttemptMemory``
is the tried[]/failures[] log; ``AlphaLibrary`` is the curated good[] pool. Both
are read back into generation so run #10 is smarter than run #1.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Set

from sqlalchemy.orm import Session

from backend.core.expression_normalizer import normalize_brain_expression
from backend.generation.dedup import expression_signature
from backend.models import AlphaLibrary, AttemptMemory
from backend.selfimprove.evaluator import (
    GateConfig,
    Verdict,
    evaluate,
    evaluate_result,
)


def settings_signature(settings: Optional[Dict[str, Any]]) -> Optional[str]:
    """Stable signature for a settings dict so (expression, settings) pairs are distinct."""
    if not settings:
        return None
    try:
        payload = json.dumps(settings, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = str(sorted(settings.items())) if hasattr(settings, "items") else str(settings)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _f(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class AttemptMemoryService:
    """Read/write the self-improving memory tables.

    Outcomes feed three loops:
      * recent_failures()   -> injected as negative examples into generation (Pattern C)
      * tried_signatures()  -> steer the proposer away from explored ground (Pattern D)
      * recent_near_misses()-> handed to the deterministic refiner (Pattern E)
    Confirmed wins are promoted into the library and injected as seeds (Pattern H).
    """

    # Library quality bar (matches the platform's "good alpha" definition).
    LIBRARY_MIN_SHARPE = 1.25
    LIBRARY_MIN_FITNESS = 1.0

    def __init__(self, db: Session):
        self.db = db

    # ----- writing ---------------------------------------------------------
    def record_attempt(
        self,
        expression: str,
        verdict: Verdict,
        *,
        settings: Optional[Dict[str, Any]] = None,
        focus: Optional[str] = None,
        dataset_id: Optional[str] = None,
        source: str = "autopilot",
        repaired_from: Optional[str] = None,
        result_id: Optional[int] = None,
        commit: bool = True,
    ) -> AttemptMemory:
        """Upsert one attempt keyed by (expression_signature, settings_signature)."""
        expression = normalize_brain_expression(expression)
        sig = expression_signature(expression)
        ssig = settings_signature(settings)
        metrics = verdict.metrics or {}

        row = (
            self.db.query(AttemptMemory)
            .filter(AttemptMemory.expression_signature == sig)
            .filter(AttemptMemory.settings_signature == ssig)
            .first()
        )
        if row is None:
            row = AttemptMemory(
                expression_signature=sig,
                settings_signature=ssig,
                expression=expression,
                attempts=1,
            )
            self.db.add(row)
        else:
            row.attempts = int(row.attempts or 1) + 1

        row.settings = settings
        row.focus = focus or row.focus
        row.dataset_id = dataset_id or row.dataset_id
        row.outcome = verdict.outcome
        row.failures = list(verdict.failures)
        row.score = float(verdict.score)
        row.sharpe = _f(metrics.get("sharpe"))
        row.fitness = _f(metrics.get("fitness"))
        row.turnover = _f(metrics.get("turnover"))
        row.self_correlation = _f(metrics.get("self_correlation"))
        row.source = source
        if repaired_from:
            row.repaired_from = repaired_from
        if result_id is not None:
            row.result_id = result_id

        if commit:
            self.db.commit()
            self.db.refresh(row)
        return row

    def record_result(
        self,
        result: Any,
        *,
        config: Optional[GateConfig] = None,
        focus: Optional[str] = None,
        dataset_id: Optional[str] = None,
        source: str = "live",
        commit: bool = True,
    ) -> Verdict:
        """Evaluate a ``Result`` row, persist the attempt, and auto-promote wins."""
        verdict = evaluate_result(result, config)
        settings = verdict.metrics.get("settings") if isinstance(verdict.metrics, dict) else None
        self.record_attempt(
            getattr(result, "expression", "") or "",
            verdict,
            settings=settings if isinstance(settings, dict) else None,
            focus=focus,
            dataset_id=dataset_id,
            source=source,
            result_id=getattr(result, "id", None),
            commit=False,
        )
        if verdict.is_win:
            self.promote_to_library(
                getattr(result, "expression", "") or "",
                settings if isinstance(settings, dict) else None,
                verdict,
                focus=focus,
                dataset_id=dataset_id,
                source="win",
                commit=False,
            )
        if commit:
            self.db.commit()
        return verdict

    def promote_to_library(
        self,
        expression: str,
        settings: Optional[Dict[str, Any]],
        verdict_or_metrics: Any,
        *,
        focus: Optional[str] = None,
        dataset_id: Optional[str] = None,
        source: str = "win",
        note: Optional[str] = None,
        commit: bool = True,
    ) -> Optional[AlphaLibrary]:
        """Add a confirmed-good alpha to the auto-growing library (upsert by signature)."""
        expression = normalize_brain_expression(expression)
        if not expression:
            return None
        metrics = (
            verdict_or_metrics.metrics
            if isinstance(verdict_or_metrics, Verdict)
            else (verdict_or_metrics or {})
        )
        score = (
            verdict_or_metrics.score
            if isinstance(verdict_or_metrics, Verdict)
            else None
        )
        sig = expression_signature(expression)
        row = (
            self.db.query(AlphaLibrary)
            .filter(AlphaLibrary.expression_signature == sig)
            .first()
        )
        if row is None:
            row = AlphaLibrary(expression_signature=sig, expression=expression)
            self.db.add(row)
        row.settings = settings
        row.focus = focus or row.focus
        row.dataset_id = dataset_id or row.dataset_id
        row.sharpe = _f(metrics.get("sharpe"))
        row.fitness = _f(metrics.get("fitness"))
        row.turnover = _f(metrics.get("turnover"))
        row.self_correlation = _f(metrics.get("self_correlation"))
        if score is not None:
            row.score = float(score)
        row.source = source
        if note:
            row.note = note
        if commit:
            self.db.commit()
            self.db.refresh(row)
        return row

    # ----- reading ---------------------------------------------------------
    def recent_failures(self, limit: int = 3) -> List[AttemptMemory]:
        """Most recent fail/near attempts, newest first (negative examples for Pattern C)."""
        return (
            self.db.query(AttemptMemory)
            .filter(AttemptMemory.outcome.in_(["fail", "near", "error"]))
            .order_by(AttemptMemory.updated_at.desc(), AttemptMemory.id.desc())
            .limit(max(0, limit))
            .all()
        )

    def recent_near_misses(self, limit: int = 5, *, max_attempts: int = 3) -> List[AttemptMemory]:
        """Repairable near-misses for the deterministic refiner, best score first.

        ``max_attempts`` avoids re-grinding the same near-miss forever.
        """
        return (
            self.db.query(AttemptMemory)
            .filter(AttemptMemory.outcome == "near")
            .filter(AttemptMemory.attempts <= max_attempts)
            .order_by(AttemptMemory.score.desc(), AttemptMemory.updated_at.desc())
            .limit(max(0, limit))
            .all()
        )

    def tried_signatures(self, limit: int = 5000) -> Set[str]:
        """All attempted expression signatures (diversify against everything tried)."""
        rows = (
            self.db.query(AttemptMemory.expression_signature)
            .order_by(AttemptMemory.id.desc())
            .limit(max(0, limit))
            .all()
        )
        return {row[0] for row in rows if row[0]}

    def tried_expressions(self, limit: int = 2000) -> List[str]:
        rows = (
            self.db.query(AttemptMemory.expression)
            .order_by(AttemptMemory.id.desc())
            .limit(max(0, limit))
            .all()
        )
        return [row[0] for row in rows if row[0]]

    def library_examples(
        self,
        limit: int = 12,
        *,
        focus: Optional[str] = None,
    ) -> List[AlphaLibrary]:
        """Top confirmed-good alphas, best score first (seed material for Pattern H)."""
        query = self.db.query(AlphaLibrary)
        if focus:
            query = query.filter(AlphaLibrary.focus == focus)
        rows = query.order_by(AlphaLibrary.score.desc(), AlphaLibrary.id.desc()).limit(max(0, limit)).all()
        if not rows and focus:
            # Fall back to the global best if nothing matches the focus.
            rows = (
                self.db.query(AlphaLibrary)
                .order_by(AlphaLibrary.score.desc(), AlphaLibrary.id.desc())
                .limit(max(0, limit))
                .all()
            )
        return rows

    def library_expressions(self, limit: int = 12, *, focus: Optional[str] = None) -> List[str]:
        return [row.expression for row in self.library_examples(limit, focus=focus) if row.expression]

    def arm_stats(self, dimension: str = "focus") -> Dict[str, tuple]:
        """Aggregate (wins, trials) per arm for the bandit explorer.

        dimension="focus" groups by AttemptMemory.focus; "dataset" by dataset_id.
        A trial is any recorded attempt; a win is outcome=="win". This is the
        observed reward signal Thompson sampling draws its Beta posteriors from.
        """
        column = AttemptMemory.dataset_id if dimension == "dataset" else AttemptMemory.focus
        rows = self.db.query(column, AttemptMemory.outcome).all()
        stats: Dict[str, tuple] = {}
        for key, outcome in rows:
            if not key:
                continue
            wins, trials = stats.get(key, (0, 0))
            trials += 1
            if outcome == "win":
                wins += 1
            stats[key] = (wins, trials)
        return stats

    def stats(self) -> Dict[str, Any]:
        """Summary of what the memory currently holds (for dashboards / debugging)."""
        total = self.db.query(AttemptMemory).count()
        wins = self.db.query(AttemptMemory).filter(AttemptMemory.outcome == "win").count()
        near = self.db.query(AttemptMemory).filter(AttemptMemory.outcome == "near").count()
        fail = self.db.query(AttemptMemory).filter(AttemptMemory.outcome == "fail").count()
        library = self.db.query(AlphaLibrary).count()
        return {
            "attempts": total,
            "wins": wins,
            "near_misses": near,
            "failures": fail,
            "win_rate": round(wins / total, 4) if total else 0.0,
            "library_size": library,
        }
