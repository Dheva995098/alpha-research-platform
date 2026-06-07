"""Turn memory into generation context (self-improving loop Patterns C and D).

Pattern C (feedback-into-prompt): recent failures become explicit negative
examples the generator/advisor is told to avoid -- in-context learning, no weight
update. Pattern D (diversification): steer away from everything already tried, and
penalise candidate shapes that have been failing, so the loop keeps exploring new
ground instead of polishing one dead-end.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

_OPERATOR_TOKEN = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")


def _records_to_rows(records: Iterable[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "expression": getattr(record, "expression", "") or "",
                "sharpe": getattr(record, "sharpe", None),
                "fitness": getattr(record, "fitness", None),
                "turnover": getattr(record, "turnover", None),
                "self_correlation": getattr(record, "self_correlation", None),
                "failures": list(getattr(record, "failures", None) or []),
                "outcome": getattr(record, "outcome", None),
            }
        )
    return rows


def build_failure_context(records: Sequence[Any], *, limit: int = 3) -> str:
    """Format recent failed/near attempts as an 'avoid these issues' prompt block.

    Be specific about *why* each one fell short (the diagnosis), not just that it
    failed -- the model needs the reason to adapt.
    """
    rows = _records_to_rows(records)[: max(0, limit)]
    if not rows:
        return ""
    lines = ["Prior attempts that fell short (avoid repeating these issues):"]
    for row in rows:
        parts = []
        if row["sharpe"] is not None:
            parts.append(f"sharpe={row['sharpe']:.2f}")
        if row["fitness"] is not None:
            parts.append(f"fitness={row['fitness']:.2f}")
        if row["turnover"] is not None:
            parts.append(f"turnover={row['turnover']:.2f}")
        if row["self_correlation"] is not None:
            parts.append(f"self_corr={row['self_correlation']:.2f}")
        issues = ", ".join(row["failures"]) if row["failures"] else "underperformed"
        metric_text = (" " + ", ".join(parts)) if parts else ""
        lines.append(f"- `{row['expression']}` ->{metric_text} issues=[{issues}]")
    return "\n".join(lines)


def failure_rows(records: Sequence[Any], *, limit: int = 3) -> List[Dict[str, Any]]:
    """Structured negative examples for a JSON prompt payload."""
    return _records_to_rows(records)[: max(0, limit)]


def avoid_signatures(records: Iterable[Any]) -> Set[str]:
    """Signatures of attempts to steer away from (diversify against everything tried)."""
    out: Set[str] = set()
    for record in records:
        sig = getattr(record, "expression_signature", None)
        if sig:
            out.add(sig)
    return out


def failure_term_weights(records: Iterable[Any]) -> Dict[str, float]:
    """Operator tokens that recur in failed attempts, weighted by how often they failed.

    Used to nudge ranking away from shapes that keep losing. Weights are small and
    bounded so they bias, never dominate, the existing score.
    """
    counts: Dict[str, int] = {}
    seen_any = 0
    for record in records:
        outcome = getattr(record, "outcome", None)
        if outcome not in {"fail", "near", "error"}:
            continue
        seen_any += 1
        expression = (getattr(record, "expression", "") or "").lower()
        for token in set(_OPERATOR_TOKEN.findall(expression)):
            counts[token] = counts.get(token, 0) + 1
    if not seen_any:
        return {}
    return {token: round(min(count / seen_any, 1.0), 4) for token, count in counts.items()}


def shape_penalty(expression: str, term_weights: Dict[str, float], *, scale: float = 0.05) -> float:
    """Small rank-time penalty for candidates built from frequently-failing operators."""
    if not term_weights:
        return 0.0
    tokens = set(_OPERATOR_TOKEN.findall((expression or "").lower()))
    if not tokens:
        return 0.0
    penalty = sum(term_weights.get(token, 0.0) for token in tokens)
    return round(min(penalty * scale, scale * 4), 4)
