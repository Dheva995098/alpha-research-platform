"""Read-only API for observing the self-improving research loop.

Lets the dashboard show that the system is actually *learning*: what it has
tried, the near-misses queued for cheap repair, and the growing library of wins.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.models import AlphaLibrary, AttemptMemory, get_db
from backend.selfimprove.memory import AttemptMemoryService

router = APIRouter()


def _attempt_row(row: AttemptMemory) -> Dict[str, Any]:
    return {
        "id": row.id,
        "expression": row.expression,
        "outcome": row.outcome,
        "failures": list(row.failures or []),
        "score": row.score,
        "sharpe": row.sharpe,
        "fitness": row.fitness,
        "turnover": row.turnover,
        "self_correlation": row.self_correlation,
        "focus": row.focus,
        "dataset_id": row.dataset_id,
        "attempts": row.attempts,
        "repaired_from": row.repaired_from,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _library_row(row: AlphaLibrary) -> Dict[str, Any]:
    return {
        "id": row.id,
        "expression": row.expression,
        "settings": row.settings,
        "focus": row.focus,
        "dataset_id": row.dataset_id,
        "score": row.score,
        "sharpe": row.sharpe,
        "fitness": row.fitness,
        "turnover": row.turnover,
        "self_correlation": row.self_correlation,
        "source": row.source,
        "note": row.note,
    }


@router.get("/stats")
def selfimprove_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """High-level counters: attempts, wins, near-misses, win-rate, library size."""
    return AttemptMemoryService(db).stats()


@router.get("/memory")
def selfimprove_memory(
    outcome: Optional[str] = Query(default=None, description="win | near | fail | error"),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Recent attempts in memory, newest first, optionally filtered by outcome."""
    query = db.query(AttemptMemory)
    if outcome:
        query = query.filter(AttemptMemory.outcome == outcome.strip().lower())
    rows = query.order_by(AttemptMemory.updated_at.desc(), AttemptMemory.id.desc()).limit(limit).all()
    return {"count": len(rows), "attempts": [_attempt_row(row) for row in rows]}


@router.get("/near-misses")
def selfimprove_near_misses(
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Repairable near-misses currently eligible for the deterministic refiner."""
    rows = AttemptMemoryService(db).recent_near_misses(limit=limit)
    return {"count": len(rows), "near_misses": [_attempt_row(row) for row in rows]}


@router.get("/library")
def selfimprove_library(
    focus: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The auto-growing library of confirmed-good alphas, best score first."""
    rows = AttemptMemoryService(db).library_examples(limit=limit, focus=focus)
    return {"count": len(rows), "library": [_library_row(row) for row in rows]}
