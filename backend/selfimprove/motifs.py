"""Proven alpha motif library (grounds generation in structures that actually work).

The strongest, lowest-correlation alphas on BRAIN follow a small set of motifs: the
101-Alphas negated price-volume reversal (`-ts_corr`, `-rank(ts_corr(rank,rank))`),
the canonical wrapper hierarchy `group_neutralize(decay(rank(ts_op(field,n))))`, and
explicit `vector_neut` decorrelation. The rule-based generator lacked the negation /
ranked-input / decorrelation variants entirely, leaving the highest-yield region of
the design space unexplored. These templates inject those proven structures as seed
candidates; everything is validated against the operator whitelist and the live field
schema before use, so unsupported fields/operators are silently skipped.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from backend.core.data_fields import BRAINDataFields, get_data_fields
from backend.core.expression_normalizer import normalize_brain_expression
from backend.generation.candidates import AlphaCandidate
from backend.generation.dedup import expression_signature

_GROUP_WORDS = {"sector", "industry", "subindustry", "market", "country", "constant"}
_OPERATOR_TOKEN = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")
_WORD_TOKEN = re.compile(r"\b[a-z_][a-z0-9_]*\b")

# focus -> proven expression templates. "{n}" is swept over BRAIN-realistic windows.
PROVEN_MOTIFS = {
    "price_volume": [
        "-ts_corr({p}, volume, {n})",
        "-rank(ts_corr(rank({p}), rank(volume), {n}))",
        "-rank(ts_covariance(rank(close), rank(volume), {n}))",
        "rank(ts_decay_linear(ts_corr(vwap, adv20, {n}), 7))",
        "rank((volume * returns) / (ts_mean(volume, 60) + 0.0001))",
    ],
    "momentum": [
        "group_neutralize(ts_rank(ts_mean(returns, {n}) / (ts_std_dev(returns, {n}) + 0.0001), 120), industry)",
        "rank(ts_rank(close, {n}))",
        "rank(ts_delta(ts_mean(returns, 20), 5) - ts_delta(ts_mean(returns, 60), 5))",
    ],
    "mean_reversion": [
        "-rank(ts_delta(close, {n}))",
        "-ts_corr(open, volume, {n})",
        "-rank(ts_delta(close, 5)) * rank(volume / (ts_mean(volume, 30) + 0.0001))",
    ],
    "reversal": [
        "-rank(ts_delta(close, {n}))",
        "-rank(ts_corr(rank(open), rank(volume), {n}))",
    ],
    "volatility": [
        "-rank(ts_std_dev(ts_std_dev(returns, 5), {n}))",
        "rank(0 - ts_std_dev(returns, 20) / (ts_std_dev(returns, 60) + 0.0001))",
        "group_zscore(0 - (close - open) / (high - low + 0.0001), subindustry)",
    ],
    "liquidity": [
        "group_neutralize(rank(ts_mean(abs(returns) / (volume * close + 0.0001), {n})), subindustry)",
        "-rank(volume / (ts_mean(volume, 20) + 0.0001)) * sign(ts_delta(close, 1))",
    ],
    "quality": [
        "group_neutralize(rank(operating_income / (assets + 0.0001)), subindustry)",
        "group_neutralize(rank(ts_delta(ebit / (sales + 0.0001), 252)), industry)",
        "group_zscore(cashflow_op / (debt + 0.0001), subindustry)",
    ],
    "fundamental": [
        "group_neutralize(ts_rank(equity / (cap + 0.0001), 240), industry)",
        "group_neutralize(rank(0 - ts_delta(debt, 90)), sector)",
    ],
    "analyst": [
        "group_neutralize(rank(ts_delta(ts_backfill(est_eps, 120), 20)), industry)",
        "group_neutralize(rank(ts_rank(ts_backfill(est_eps, 120) / (close + 0.0001), 40)), industry)",
        "group_neutralize(rank(0 - ts_std_dev(ts_backfill(est_eps, 120), 60)), industry)",
    ],
    "sentiment": [
        "group_rank(ts_zscore(ts_backfill(scl12_sentiment, 60), 60), subindustry)",
        "group_rank(ts_zscore(ts_backfill(snt1_score, 60), 60), subindustry)",
    ],
    "options": [
        "-rank(ts_backfill(implied_volatility_put_180, 60) - ts_backfill(implied_volatility_call_180, 60))",
        "group_neutralize(rank(ts_delta(ts_backfill(implied_volatility_call_180, 60), 25)), market)",
    ],
    "model_risk": [
        "group_rank(ts_rank(mdl77_momentum, 60) - ts_rank(mdl77_reversal, 60), sector)",
        "group_neutralize(rank(ts_rank(mdl16_quality_score, 60) + ts_rank(mdl16_value_score, 60)), industry)",
    ],
    "decorrelation": [
        "vector_neut(rank(ts_corr(close, volume, {n})), cap)",
        "vector_neut(rank(operating_income / (assets + 0.0001)), ts_mean(returns, 120))",
        "group_vector_neut(rank(ts_corr(close, volume, {n})), ts_mean(returns, 120), subindustry)",
    ],
}

# Always blend in the strongest, most general (price-volume reversal) motifs.
_ALWAYS = ("price_volume",)

_DEFAULT_WINDOWS = (5, 10, 22, 66, 120)
_PRICE_FIELDS = ("close", "vwap", "open", "high", "low", "returns")


def _operators(expression: str) -> set:
    return set(_OPERATOR_TOKEN.findall(expression.lower()))


def _fields_used(expression: str, schema: BRAINDataFields) -> List[str]:
    operators = _operators(expression)
    fields = []
    for token in _WORD_TOKEN.findall(expression.lower()):
        if token in operators or token in _GROUP_WORDS:
            continue
        if schema.validate_field(token):
            fields.append(token)
    return sorted(set(fields))


def _all_words_resolved(expression: str, schema: BRAINDataFields) -> bool:
    """Every non-operator, non-group word must be a known field (else skip the motif)."""
    operators = _operators(expression)
    for token in set(_WORD_TOKEN.findall(expression.lower())):
        if token in operators or token in _GROUP_WORDS:
            continue
        if not schema.validate_field(token):
            return False
    return True


def motif_candidates(
    focus: Optional[str],
    *,
    schema: Optional[BRAINDataFields] = None,
    windows: Sequence[int] = _DEFAULT_WINDOWS,
    limit: int = 12,
    include_general: bool = True,
) -> List[AlphaCandidate]:
    """Return validated proven-motif candidates for a focus (best-known structures)."""
    schema = schema or get_data_fields()
    keys = []
    if focus and focus in PROVEN_MOTIFS:
        keys.append(focus)
    if include_general:
        for key in _ALWAYS:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = list(_ALWAYS)

    out: List[AlphaCandidate] = []
    seen = set()
    for key in keys:
        for template in PROVEN_MOTIFS.get(key, []):
            window_values = windows if "{n}" in template else (None,)
            price_values = _PRICE_FIELDS if "{p}" in template else (None,)
            for price in price_values:
                for window in window_values:
                    expr = template
                    if price is not None:
                        expr = expr.replace("{p}", price)
                    if window is not None:
                        expr = expr.replace("{n}", str(window))
                    expr = normalize_brain_expression(expr)
                    signature = expression_signature(expr)
                    if signature in seen:
                        continue
                    is_valid, _ = schema.validate_expression_basic(expr)
                    if not is_valid or not _all_words_resolved(expr, schema):
                        continue
                    seen.add(signature)
                    out.append(
                        AlphaCandidate(
                            expression=expr,
                            strategy=key,
                            source_fields=tuple(_fields_used(expr, schema)),
                            operators=tuple(sorted(_operators(expr))),
                            rationale="proven_motif",
                            score=0.62,
                        )
                    )
                    if len(out) >= limit:
                        return out
    return out
