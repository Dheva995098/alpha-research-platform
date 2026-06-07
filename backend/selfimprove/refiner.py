"""Deterministic failure->fix refiner (self-improving loop Pattern E).

Cheap repair before paying for an expensive regeneration. Each known failure mode
maps to a known structural fix, applied as a pure function (no LLM call):

    HIGH_TURNOVER            -> wrap in ts_decay_linear / smooth, raise decay
    COVERAGE_FAIL            -> wrap inputs in ts_backfill / group_backfill
    HIGH_SELF_CORRELATION    -> decorrelate: re-rank, shift horizon, change group
    LOW_SUB_UNIVERSE_SHARPE  -> switch neutralization group (finer <-> coarser)
    LOW_SHARPE / LOW_FITNESS -> outlier control (winsorize), backfill, re-rank

Fixes change the *expression* (not just settings) so each repaired variant has a
fresh signature and is not blocked by the all-time dedup registry. Variants are
validated against the operator whitelist before they are ever queued.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from backend.core.data_fields import BRAINDataFields, get_data_fields
from backend.core.expression_normalizer import normalize_brain_expression
from backend.generation.dedup import expression_signature
from backend.selfimprove.evaluator import (
    CONCENTRATED_WEIGHT,
    COVERAGE_FAIL,
    FAILED_CHECKS,
    HIGH_PROD_CORRELATION,
    HIGH_SELF_CORRELATION,
    HIGH_TURNOVER,
    LOW_FITNESS,
    LOW_SHARPE,
    LOW_SUB_UNIVERSE_SHARPE,
    LOW_TURNOVER,
    UNITS,
    Verdict,
)

_GROUPS = ("sector", "industry", "subindustry", "market")

# Regime triggers for trade_when gating (turnover control + Sharpe lift).
_REGIME_TRIGGERS = (
    "ts_rank(ts_std_dev(returns, 22), 252) > 0.5",
    "ts_rank(ts_std_dev(returns, 66), 252) < 0.5",
)
# Risk factors to orthogonalize against for decorrelation (vector_neut).
_RISK_FACTORS = ("cap", "ts_mean(returns, 120)")


@dataclass(frozen=True)
class RepairVariant:
    """One deterministically-repaired candidate derived from a near-miss."""

    expression: str
    fix: str  # human-readable label, e.g. "decay_for_turnover"
    settings: Optional[Dict[str, Any]] = None
    parent_signature: Optional[str] = None


class DeterministicRefiner:
    """Map failure tags to known fixes and emit validated repaired variants."""

    def __init__(self, schema: Optional[BRAINDataFields] = None):
        self.schema = schema or get_data_fields()

    def repair(
        self,
        expression: str,
        verdict: Verdict,
        *,
        settings: Optional[Dict[str, Any]] = None,
        avoid_signatures: Optional[Set[str]] = None,
        max_variants: int = 6,
    ) -> List[RepairVariant]:
        """Return cheap repaired variants for a near-miss, highest-leverage first."""
        base = normalize_brain_expression(expression)
        if not base:
            return []
        parent_sig = expression_signature(base)
        avoid = set(avoid_signatures or set())
        avoid.add(parent_sig)

        failures = list(verdict.failures or [])
        settings = dict(settings or {})
        current_group = str(settings.get("neutralization") or "").strip().lower()

        # (expression, fix-label, settings-override) in priority order:
        #   1. data/validity fixes  2. cheap knobs  3. structural changes.
        proposals: List[tuple] = []

        if COVERAGE_FAIL in failures or FAILED_CHECKS in failures:
            proposals.append((self._wrap(base, "ts_backfill", 120), "backfill_120_for_coverage", None))
            proposals.append((f"winsorize({self._wrap(base, 'ts_backfill', 240)})", "backfill_winsorize_for_coverage", None))

        if HIGH_TURNOVER in failures:
            # Smooth/decay first, then gate trading on a regime, then a hard turnover cap (hump).
            proposals.append((self._wrap(base, "ts_decay_linear", 6), "decay_linear_6_for_turnover",
                              self._with(settings, decay=max(int(settings.get("decay") or 0), 10))))
            proposals.append((self._wrap(base, "ts_decay_linear", 10), "decay_linear_10_for_turnover",
                              self._with(settings, decay=max(int(settings.get("decay") or 0), 20))))
            proposals.append((f"hump({base}, 0.01)", "hump_for_turnover", None))
            proposals.append((f"trade_when({_REGIME_TRIGGERS[0]}, {base}, -1)", "regime_gate_for_turnover", None))

        if LOW_TURNOVER in failures:
            # Too little trading: sharpen/shorten the signal and DROP decay (the opposite of HIGH_TURNOVER).
            proposals.append((self._wrap(base, "ts_zscore", 5), "sharpen_zscore_5_for_low_turnover",
                              self._with(settings, decay=min(int(settings.get("decay") or 4), 2))))
            proposals.append((self._wrap(base, "ts_delta", 1), "delta_1_for_low_turnover", None))

        if HIGH_SELF_CORRELATION in failures or HIGH_PROD_CORRELATION in failures:
            # vector_neut residualization is the canonical BRAIN decorrelator (a - (a.b)b).
            for factor in _RISK_FACTORS:
                proposals.append((f"vector_neut({base}, {factor})", f"vector_neut_for_decorrelation", None))
            proposals.append((f"rank({self._wrap(base, 'ts_decay_linear', 6)})", "rerank_decay_for_self_corr", None))
            proposals.append((self._wrap(base, "ts_zscore", 22), "horizon_shift_for_self_corr", None))
            other = self._other_group(current_group)
            proposals.append((self._group_neutralize(base, other), f"neutralize_{other}_for_self_corr",
                              self._with(settings, neutralization=other.upper())))

        if CONCENTRATED_WEIGHT in failures:
            # A single name holds too much book: clip outliers, re-rank, and tighten truncation.
            proposals.append((self._wrap(base, "winsorize"), "winsorize_for_weight_concentration",
                              self._with(settings, truncation=0.01)))
            proposals.append((self._wrap(base, "rank"), "rank_for_weight_concentration",
                              self._with(settings, truncation=0.01)))

        if UNITS in failures:
            # Dimensional-consistency violation: rank() makes the alpha dimensionless.
            proposals.append((self._wrap(base, "rank"), "rank_for_units", None))
            proposals.append((self._wrap(base, "zscore"), "zscore_for_units", None))

        if LOW_SUB_UNIVERSE_SHARPE in failures:
            for group in self._neighbour_groups(current_group):
                proposals.append((self._group_neutralize(base, group), f"neutralize_{group}_for_sub_universe",
                                  self._with(settings, neutralization=group.upper())))

        if LOW_SHARPE in failures or LOW_FITNESS in failures:
            proposals.append((self._wrap(base, "winsorize"), "winsorize_for_outliers", None))
            proposals.append((f"rank({self._wrap(base, 'ts_backfill', 120)})", "rank_backfill_for_signal", None))

        # If diagnosis was inconclusive, still try the two safest general repairs.
        if not proposals:
            proposals.append((self._wrap(base, "winsorize"), "winsorize_general", None))
            proposals.append((self._wrap(base, "ts_backfill", 120), "backfill_general", None))

        variants: List[RepairVariant] = []
        for expr, fix, override in proposals:
            if len(variants) >= max_variants:
                break
            candidate = normalize_brain_expression(expr)
            if not candidate:
                continue
            sig = expression_signature(candidate)
            if sig in avoid:
                continue
            is_valid, _ = self.schema.validate_expression_basic(candidate)
            if not is_valid:
                continue
            avoid.add(sig)
            variants.append(
                RepairVariant(
                    expression=candidate,
                    fix=fix,
                    settings=override,
                    parent_signature=parent_sig,
                )
            )
        return variants

    # ----- expression builders --------------------------------------------
    @staticmethod
    def _wrap(expression: str, operator: str, *args: Any) -> str:
        if args:
            arg_text = ", ".join(str(arg) for arg in args)
            return f"{operator}({expression}, {arg_text})"
        return f"{operator}({expression})"

    @staticmethod
    def _group_neutralize(expression: str, group: str) -> str:
        return f"group_neutralize({expression}, {group})"

    @staticmethod
    def _with(settings: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
        merged = dict(settings)
        merged.update({key: value for key, value in overrides.items() if value is not None})
        return merged

    @staticmethod
    def _other_group(current: str) -> str:
        # Pick a coarse group different from the current one to break correlation.
        if current in {"subindustry", "industry", "sector"}:
            return "market"
        return "sector"

    @staticmethod
    def _neighbour_groups(current: str) -> List[str]:
        """Groups one step coarser and finer than the current neutralization."""
        order = ["subindustry", "industry", "sector", "market"]
        if current not in order:
            return ["sector", "industry"]
        idx = order.index(current)
        neighbours = []
        if idx + 1 < len(order):
            neighbours.append(order[idx + 1])  # coarser
        if idx - 1 >= 0:
            neighbours.append(order[idx - 1])  # finer
        return neighbours or ["sector"]
