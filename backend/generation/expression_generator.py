"""Rule-based alpha expression generation for Phase 2."""
from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence, Set

from backend.core.data_fields import BRAINDataFields, get_data_fields
from backend.core.dataset_catalog import (
    datasets_for_fields,
    fields_for_datasets,
    focuses_for_datasets,
    normalize_dataset_ids,
)
from backend.core.expression_normalizer import normalize_brain_expression
from backend.generation.candidates import AlphaCandidate
from backend.generation.dedup import ExpressionDeduplicator


STRATEGY_DESCRIPTIONS = {
    "momentum": "Trend-following signals using recent return persistence.",
    "mean_reversion": "Contrarian signals where price or returns are stretched.",
    "price_volume": "Signals combining price movement with trading activity.",
    "liquidity": "Signals based on volume, average volume, and liquidity shifts.",
    "volatility": "Signals using volatility compression or expansion.",
    "size": "Signals using market capitalization or size proxies.",
    "intraday": "Signals using open, high, low, close, and VWAP relationships.",
    "quality": "Fundamental profitability, investment, balance sheet, and cash-flow signals.",
    "fundamental": "Slow-moving fundamental valuation and operating-efficiency signals.",
    "hybrid": "Multi-factor blends that combine price, volume, volatility, and fundamentals.",
    "analyst": "Analyst estimate and target-price expectation signals.",
    "sentiment": "News and social sentiment stability or reaction signals.",
    "options": "Options-implied volatility and open-interest signals.",
    "model_risk": "Model-derived risk, technical, credit, and factor-score signals.",
    "decorrelation": "Variations designed to lower self/prod correlation while preserving the idea.",
}

FOCUS_ALIASES = {
    "momentum": {"momentum", "trend", "trend_following"},
    "mean_reversion": {"mean_reversion", "reversion", "contrarian", "reversal"},
    "price_volume": {"price_volume", "volume_price", "pv", "correlation"},
    "liquidity": {"liquidity", "volume", "adv"},
    "volatility": {"volatility", "risk", "std"},
    "size": {"size", "cap", "market_cap"},
    "intraday": {"intraday", "ohlcv", "vwap"},
    "quality": {"quality", "profitability", "cashflow", "cash_flow"},
    "fundamental": {"fundamental", "fundamentals", "value", "valuation"},
    "hybrid": {"hybrid", "multifactor", "multi_factor", "blend"},
    "analyst": {"analyst", "estimate", "estimates", "target_price"},
    "sentiment": {"sentiment", "news", "buzz", "social"},
    "options": {"options", "option", "iv", "vol_skew"},
    "model_risk": {"model_risk", "risk_model", "risk", "model", "technical_model", "credit"},
    "decorrelation": {"decorrelation", "corr", "correlation", "diversify"},
}


class RuleBasedAlphaGenerator:
    """Generate deterministic, valid-enough FASTEXPR candidates without an LLM."""

    WINDOWS_SHORT = (5, 10, 20)
    WINDOWS_MEDIUM = (20, 40, 60)
    WINDOWS_LONG = (120, 252)
    GROUPS = ("sector", "industry", "subindustry", "market")

    def __init__(
        self,
        schema: Optional[BRAINDataFields] = None,
        seed: Optional[int] = None,
    ):
        self.schema = schema or get_data_fields()
        self.random = random.Random(seed)

    def generate(
        self,
        count: int = 20,
        focus: Optional[str] = None,
        fields: Optional[Sequence[str]] = None,
        dataset_ids: Optional[Sequence[str]] = None,
        neutralize: bool = True,
        existing_expressions: Optional[Iterable[str]] = None,
    ) -> List[AlphaCandidate]:
        """Generate candidate expressions for one or all strategies."""
        count = max(1, min(count, 500))
        selected_datasets = normalize_dataset_ids(dataset_ids)
        allowed_fields = self._allowed_fields(fields, selected_datasets)
        strategies = self._strategy_filter(focus, selected_datasets)
        deduplicator = ExpressionDeduplicator(existing_expressions)

        base_candidates = self._build_candidates(allowed_fields, strategies)
        self.random.shuffle(base_candidates)

        generated: List[AlphaCandidate] = []
        for candidate in base_candidates:
            self._append_if_valid(candidate, allowed_fields, deduplicator, generated)
            if len(generated) >= count:
                return generated

            if neutralize and candidate.strategy != "size":
                wrapped = self._neutralized_variant(candidate)
                self._append_if_valid(wrapped, allowed_fields, deduplicator, generated)
                if len(generated) >= count:
                    return generated

        attempts = 0
        while len(generated) < count and attempts < count * 20:
            attempts += 1
            candidate = self._random_candidate(allowed_fields, strategies)
            self._append_if_valid(candidate, allowed_fields, deduplicator, generated)

        return generated

    def _append_if_valid(
        self,
        candidate: AlphaCandidate,
        allowed_fields: Set[str],
        deduplicator: ExpressionDeduplicator,
        generated: List[AlphaCandidate],
    ) -> None:
        if not set(candidate.source_fields).issubset(allowed_fields):
            return

        expression = normalize_brain_expression(candidate.expression)
        candidate = AlphaCandidate(
            expression=expression,
            strategy=candidate.strategy,
            source_fields=candidate.source_fields,
            dataset_ids=candidate.dataset_ids or datasets_for_fields(candidate.source_fields),
            operators=tuple(
                dict.fromkeys("ts_std_dev" if operator == "ts_std" else operator for operator in candidate.operators)
            ),
            rationale=candidate.rationale,
            score=candidate.score,
        )

        valid, _ = self.schema.validate_expression_basic(candidate.expression)
        if not valid:
            return

        is_unique, _, _ = deduplicator.add(candidate.expression)
        if is_unique:
            generated.append(candidate)

    def _allowed_fields(
        self,
        fields: Optional[Sequence[str]],
        dataset_ids: Optional[Sequence[str]] = None,
    ) -> Set[str]:
        if fields:
            requested = {field.strip().lower() for field in fields if field.strip()}
            valid = {field for field in requested if self.schema.validate_field(field)}
            if valid:
                return valid
        dataset_fields = fields_for_datasets(dataset_ids)
        valid_dataset_fields = {field for field in dataset_fields if self.schema.validate_field(field)}
        if valid_dataset_fields:
            return valid_dataset_fields
        return set(self.schema.fields)

    def _strategy_filter(
        self,
        focus: Optional[str],
        dataset_ids: Optional[Sequence[str]] = None,
    ) -> Set[str]:
        if not focus or focus.lower() in {"all", "any"}:
            dataset_focuses = focuses_for_datasets(dataset_ids)
            return dataset_focuses or set(STRATEGY_DESCRIPTIONS)

        normalized = focus.strip().lower().replace("-", "_").replace(" ", "_")
        matches = {
            strategy
            for strategy, aliases in FOCUS_ALIASES.items()
            if normalized == strategy or normalized in aliases
        }
        return matches or set(STRATEGY_DESCRIPTIONS)

    def _build_candidates(
        self,
        allowed_fields: Set[str],
        strategies: Set[str],
    ) -> List[AlphaCandidate]:
        price = self._preferred(allowed_fields, ("close", "vwap", "open"))
        returns = self._preferred(allowed_fields, ("returns", "ret_0_1", "ret_1_5", price))
        volume = self._preferred(allowed_fields, ("volume", "adv20", "adv5"))
        cap = self._preferred(allowed_fields, ("cap", "mcap"))
        ebit = self._preferred(allowed_fields, ("ebit", "ebitda", "operating_income", "net_income"))
        capex = self._preferred(allowed_fields, ("capex",))
        assets = self._preferred(allowed_fields, ("assets", "bookvalue", "equity"))
        debt = self._preferred(allowed_fields, ("debt", "liabilities"))
        cashflow = self._preferred(allowed_fields, ("cashflow_op", "free_cash_flow", "operating_cash_flow", "cashflow"))
        sales = self._preferred(allowed_fields, ("sales", "revenue"))
        enterprise_value = self._preferred(allowed_fields, ("enterprise_value", "cap", "mcap"))
        est_eps = self._preferred(allowed_fields, ("est_eps", "eps"))
        est_ptp = self._preferred(allowed_fields, ("est_ptp",))
        est_fcf = self._preferred(allowed_fields, ("est_fcf", "free_cash_flow", "cashflow_op"))
        est_cashflow_op = self._preferred(allowed_fields, ("est_cashflow_op", "cashflow_op"))
        est_capex = self._preferred(allowed_fields, ("est_capex", "capex"))
        sentiment_volume = self._preferred(allowed_fields, ("scl12_buzz", "volume"))
        news_attention = self._preferred(allowed_fields, ("news_pct_1min", "scl12_buzz"))
        news_reaction = self._preferred(allowed_fields, ("news_max_up_ret", "rpna_relevance"))
        sentiment_score = self._preferred(
            allowed_fields,
            ("snt1_score", "snt1_sentiment", "scl12_sentiment", "rpna_sentiment"),
        )
        iv_call_120 = self._preferred(allowed_fields, ("implied_volatility_call_120",))
        iv_put_180 = self._preferred(allowed_fields, ("implied_volatility_put_180",))
        iv_call_180 = self._preferred(allowed_fields, ("implied_volatility_call_180",))
        iv_mean_180 = self._preferred(allowed_fields, ("implied_volatility_mean_180",))
        pcr_oi_270 = self._preferred(allowed_fields, ("pcr_oi_270",))
        iv_call_270 = self._preferred(allowed_fields, ("implied_volatility_call_270",))
        iv_put_270 = self._preferred(allowed_fields, ("implied_volatility_put_270",))
        pcr_oi_720 = self._preferred(allowed_fields, ("pcr_oi_720",))
        iv_call_720 = self._preferred(allowed_fields, ("implied_volatility_call_720",))
        iv_put_720 = self._preferred(allowed_fields, ("implied_volatility_put_720",))
        parkinson_vol = self._preferred(allowed_fields, ("parkinson_volatility_120", "volatility"))
        rel_ret_all = self._preferred(allowed_fields, ("rel_ret_all", "returns"))
        model_quality = self._preferred(allowed_fields, ("mdl16_quality_score", "operating_income"))
        model_value = self._preferred(allowed_fields, ("mdl16_value_score", "bookvalue"))
        model_momentum = self._preferred(allowed_fields, ("mdl77_momentum", "momentum"))
        model_reversal = self._preferred(allowed_fields, ("mdl77_reversal", "rsi"))
        model_credit = self._preferred(allowed_fields, ("mdl53_credit_score", "debt"))
        model_default = self._preferred(allowed_fields, ("mdl53_default_risk", "liabilities"))

        candidates: List[AlphaCandidate] = []

        if "momentum" in strategies and returns:
            for window in self.WINDOWS_MEDIUM:
                candidates.append(
                    self._candidate(
                        f"rank(ts_rank({returns}, {window}))",
                        "momentum",
                        (returns,),
                        ("rank", "ts_rank"),
                        f"Ranks {window}-day return persistence cross-sectionally.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(ts_decay_linear({returns}, {window}))",
                        "momentum",
                        (returns,),
                        ("rank", "ts_decay_linear"),
                        f"Applies linear decay to recent {returns} momentum.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(ts_delta(ts_mean({returns}, {window}), 5))",
                        "momentum",
                        (returns,),
                        ("rank", "ts_delta", "ts_mean"),
                        f"Looks for improving {window}-day return trend slope.",
                    )
                )
            for short in self.WINDOWS_SHORT:
                long = self.random.choice(self.WINDOWS_LONG)
                candidates.append(
                    self._candidate(
                        f"rank(ts_mean({returns}, {short}) - ts_mean({returns}, {long}))",
                        "momentum",
                        (returns,),
                        ("rank", "ts_mean"),
                        "Compares short-horizon and long-horizon average returns.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(ts_mean({returns}, {short}) / (ts_std_dev({returns}, {long}) + 0.001))",
                        "momentum",
                        (returns,),
                        ("rank", "ts_mean", "ts_std_dev"),
                        "Scales recent return persistence by long-window volatility.",
                    )
                )

        if "mean_reversion" in strategies and price:
            for window in self.WINDOWS_MEDIUM:
                candidates.append(
                    self._candidate(
                        f"rank(0 - ts_zscore({price}, {window}))",
                        "mean_reversion",
                        (price,),
                        ("rank", "ts_zscore"),
                        f"Favors names whose {price} is below its {window}-day z-score.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank((ts_mean({price}, {window}) - {price}) / ts_std_dev({price}, {window}))",
                        "mean_reversion",
                        (price,),
                        ("rank", "ts_mean", "ts_std_dev"),
                        "Normalizes the gap between spot price and moving average.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"winsorize(rank((ts_mean({price}, {window}) - {price}) / (ts_std_dev({price}, {window}) + 0.001)))",
                        "mean_reversion",
                        (price,),
                        ("winsorize", "rank", "ts_mean", "ts_std_dev"),
                        "Mean-reversion displacement with outlier control.",
                    )
                )
            if {"open", "close"}.issubset(allowed_fields):
                candidates.append(
                    self._candidate(
                        "group_rank(close - open, subindustry)",
                        "mean_reversion",
                        ("close", "open"),
                        ("group_rank",),
                        "Ranks close-open reversal pressure within subindustry.",
                    )
                )

        if "price_volume" in strategies and price and volume:
            for window in self.WINDOWS_SHORT + self.WINDOWS_MEDIUM:
                candidates.append(
                    self._candidate(
                        f"rank(ts_corr({price}, {volume}, {window}))",
                        "price_volume",
                        (price, volume),
                        ("rank", "ts_corr"),
                        f"Ranks {window}-day price-volume correlation.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(ts_rank({volume}, {window}) * ts_rank({price}, {window}))",
                        "price_volume",
                        (price, volume),
                        ("rank", "ts_rank"),
                        "Combines price and volume time-series strength.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(ts_zscore({volume}, {window}) * ts_rank({price}, {window}))",
                        "price_volume",
                        (price, volume),
                        ("rank", "ts_zscore", "ts_rank"),
                        "Combines unusual volume with time-series price rank.",
                    )
                )
            if returns:
                candidates.append(
                    self._candidate(
                        f"rank(ts_corr({returns}, {volume}, 20) - ts_corr({returns}, {volume}, 120))",
                        "price_volume",
                        (returns, volume),
                        ("rank", "ts_corr"),
                        "Looks for changing return-volume coupling.",
                    )
                )

        if "liquidity" in strategies and volume:
            for window in self.WINDOWS_MEDIUM:
                candidates.append(
                    self._candidate(
                        f"rank(ts_zscore({volume}, {window}))",
                        "liquidity",
                        (volume,),
                        ("rank", "ts_zscore"),
                        f"Detects unusual liquidity over {window} days.",
                    )
                )
            if "adv20" in allowed_fields and volume != "adv20":
                candidates.append(
                    self._candidate(
                        f"rank(ts_mean({volume}, 20) / adv20)",
                        "liquidity",
                        (volume, "adv20"),
                        ("rank", "ts_mean"),
                        "Compares current average volume with ADV20.",
                    )
                )

        if "volatility" in strategies and returns:
            for window in self.WINDOWS_MEDIUM:
                candidates.append(
                    self._candidate(
                        f"rank(0 - ts_std_dev({returns}, {window}))",
                        "volatility",
                        (returns,),
                        ("rank", "ts_std_dev"),
                        f"Prefers lower {window}-day realized volatility.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(ts_rank(ts_std_dev({returns}, 10), {window}))",
                        "volatility",
                        (returns,),
                        ("rank", "ts_rank", "ts_std_dev"),
                        "Ranks recent volatility within a longer volatility window.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(0 - (ts_std_dev({returns}, 10) / (ts_std_dev({returns}, {window}) + 0.001)))",
                        "volatility",
                        (returns,),
                        ("rank", "ts_std_dev"),
                        "Favors volatility compression versus a longer regime.",
                    )
                )

        if "size" in strategies and cap:
            candidates.append(
                self._candidate(
                    f"rank(0 - log({cap}))",
                    "size",
                    (cap,),
                    ("rank", "log"),
                    "Builds a simple small-cap tilt.",
                )
            )
            candidates.append(
                self._candidate(
                    f"rank(ts_rank({cap}, 20))",
                    "size",
                    (cap,),
                    ("rank", "ts_rank"),
                    "Ranks recent changes in size proxy.",
                )
            )

        if "intraday" in strategies:
            needed = {"open", "close", "high", "low"} & allowed_fields
            if {"open", "close"}.issubset(allowed_fields):
                candidates.append(
                    self._candidate(
                        "rank((close - open) / open)",
                        "intraday",
                        ("close", "open"),
                        ("rank",),
                        "Ranks open-to-close intraday return.",
                    )
                )
            if {"high", "low", "close", "open"}.issubset(allowed_fields):
                candidates.append(
                    self._candidate(
                        "rank(((close - open) / (high - low + 0.001)))",
                        "intraday",
                        tuple(sorted(needed)),
                        ("rank",),
                        "Normalizes open-to-close movement by the daily range.",
                    )
                )
            if {"close", "vwap"}.issubset(allowed_fields):
                candidates.append(
                    self._candidate(
                        "rank((close - vwap) / vwap)",
                        "intraday",
                        ("close", "vwap"),
                        ("rank",),
                        "Ranks close versus VWAP displacement.",
                    )
                )
            if {"open", "close", "high", "low", "volume"}.issubset(allowed_fields):
                candidates.append(
                    self._candidate(
                        "rank(ts_zscore(volume, 20) * ((close - open) / (high - low + 0.001)))",
                        "intraday",
                        ("open", "close", "high", "low", "volume"),
                        ("rank", "ts_zscore"),
                        "Weights intraday displacement by abnormal volume.",
                    )
                )

        if {"quality", "fundamental"}.intersection(strategies):
            if "fn_liab_fair_val_l1_a" in allowed_fields:
                candidates.append(
                    self._candidate(
                        "rank(0 - ts_rank(fn_liab_fair_val_l1_a, 252))",
                        "fundamental",
                        ("fn_liab_fair_val_l1_a",),
                        ("rank", "ts_rank"),
                        "Shorts rising fair-value liabilities relative to their yearly history.",
                    )
                )
            if ebit and capex:
                candidates.append(
                    self._candidate(
                        f"rank({ebit} / (abs({capex}) + 0.001))",
                        "quality",
                        (ebit, capex),
                        ("rank", "abs"),
                        "Compares operating profit to reinvestment intensity.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(ts_rank({ebit}, 252) - ts_rank(abs({capex}), 252))",
                        "quality",
                        (ebit, capex),
                        ("rank", "ts_rank", "abs"),
                        "Contrasts long-window profit rank against capex rank.",
                    )
                )
            if ebit and assets:
                candidates.append(
                    self._candidate(
                        f"rank({ebit} / (abs({assets}) + 0.001))",
                        "quality",
                        (ebit, assets),
                        ("rank", "abs"),
                        "Ranks operating return on assets.",
                    )
                )
            if cashflow and assets:
                candidates.append(
                    self._candidate(
                        f"rank({cashflow} / (abs({assets}) + 0.001))",
                        "quality",
                        (cashflow, assets),
                        ("rank", "abs"),
                        "Ranks cash-flow yield on asset base.",
                    )
                )
            if cashflow and cap:
                candidates.append(
                    self._candidate(
                        f"group_rank(ts_rank({cashflow} / (abs({cap}) + 0.001), 60), subindustry)",
                        "fundamental",
                        (cashflow, cap),
                        ("group_rank", "ts_rank", "abs"),
                        "Ranks operating cash-flow yield within subindustry.",
                    )
                )
            if debt and assets:
                candidates.append(
                    self._candidate(
                        f"rank(0 - ({debt} / (abs({assets}) + 0.001)))",
                        "quality",
                        (debt, assets),
                        ("rank", "abs"),
                        "Prefers lower leverage after scale normalization.",
                    )
                )
            if sales and assets:
                candidates.append(
                    self._candidate(
                        f"rank(ts_delta({sales} / (abs({assets}) + 0.001), 252))",
                        "fundamental",
                        (sales, assets),
                        ("rank", "ts_delta", "abs"),
                        "Ranks improving asset turnover over a long horizon.",
                    )
                )

        if "analyst" in strategies:
            if est_eps and price:
                candidates.append(
                    self._candidate(
                        f"group_rank(ts_rank({est_eps} / ({price} + 0.001), 60), industry)",
                        "analyst",
                        (est_eps, price),
                        ("group_rank", "ts_rank"),
                        "Ranks analyst earnings yield momentum within industry.",
                    )
                )
            if est_ptp and est_fcf:
                candidates.append(
                    self._candidate(
                        f"rank(0 - ts_corr({est_ptp}, {est_fcf}, 60))",
                        "analyst",
                        (est_ptp, est_fcf),
                        ("rank", "ts_corr"),
                        "Looks for target-price/free-cash-flow estimate overpricing pressure.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"rank(0 - ts_corr(ts_backfill({est_ptp}, 60), ts_backfill({est_fcf}, 60), 120))",
                        "analyst",
                        (est_ptp, est_fcf),
                        ("rank", "ts_corr", "ts_backfill"),
                        "Backfilled analyst estimate correlation over a medium horizon.",
                    )
                )

        if "sentiment" in strategies:
            if sentiment_volume:
                candidates.append(
                    self._candidate(
                        f"rank(0 - ts_std_dev({sentiment_volume}, 10))",
                        "sentiment",
                        (sentiment_volume,),
                        ("rank", "ts_std_dev"),
                        "Prefers stable short-term sentiment attention.",
                    )
                )
                candidates.append(
                    self._candidate(
                        f"winsorize(rank(0 - ts_zscore(ts_std_dev({sentiment_volume}, 10), 60)))",
                        "sentiment",
                        (sentiment_volume,),
                        ("winsorize", "rank", "ts_zscore", "ts_std_dev"),
                        "Normalizes unstable sentiment-volume spikes.",
                    )
                )
            if news_attention:
                candidates.append(
                    self._candidate(
                        f"rank(0 - ts_std_dev(ts_backfill({news_attention}, 60), 10))",
                        "sentiment",
                        (news_attention,),
                        ("rank", "ts_std_dev", "ts_backfill"),
                        "Prefers stable news or social attention after sparse-data backfill.",
                    )
                )
            if sentiment_score:
                candidates.append(
                    self._candidate(
                        f"rank(ts_rank(ts_backfill({sentiment_score}, 60), 20))",
                        "sentiment",
                        (sentiment_score,),
                        ("rank", "ts_rank", "ts_backfill"),
                        "Ranks recent sentiment score after backfilling sparse observations.",
                    )
                )
            if news_attention and news_reaction:
                candidates.append(
                    self._candidate(
                        f"trade_when(ts_backfill({news_attention}, 60) > 0, rank(ts_backfill({news_reaction}, 60)), -1)",
                        "sentiment",
                        (news_attention, news_reaction),
                        ("trade_when", "rank", "ts_backfill"),
                        "Trades post-news reaction only when attention is present.",
                    )
                )
            if {"news_pct_1min", "news_max_up_ret"}.issubset(allowed_fields):
                candidates.append(
                    self._candidate(
                        "winsorize(0 - ts_backfill(news_max_up_ret, 60) * abs(ts_regression(ts_backfill(news_pct_1min, 60), ts_step(1), 5, 2)))",
                        "sentiment",
                        ("news_pct_1min", "news_max_up_ret"),
                        ("winsorize", "ts_backfill", "abs", "ts_regression", "ts_step"),
                        "Flags deteriorating post-news reaction with outlier control.",
                    )
                )

        if "options" in strategies:
            if iv_call_120 and parkinson_vol:
                candidates.append(
                    self._candidate(
                        f"rank(ts_backfill({iv_call_120}, 60) / (ts_backfill({parkinson_vol}, 60) + 0.001))",
                        "options",
                        (iv_call_120, parkinson_vol),
                        ("rank", "ts_backfill"),
                        "Compares implied volatility to realized Parkinson volatility.",
                    )
                )
            if iv_call_180 and iv_put_180 and iv_mean_180:
                candidates.append(
                    self._candidate(
                        f"rank((ts_backfill({iv_call_180}, 60) - ts_backfill({iv_put_180}, 60)) / (ts_backfill({iv_mean_180}, 60) + 0.001))",
                        "options",
                        (iv_call_180, iv_put_180, iv_mean_180),
                        ("rank", "ts_backfill"),
                        "Ranks call-put volatility skew normalized by average implied volatility.",
                    )
                )
            if pcr_oi_270 and iv_call_270 and iv_put_270:
                candidates.append(
                    self._candidate(
                        f"trade_when({pcr_oi_270} < 1, rank(ts_backfill({iv_call_270}, 60) - ts_backfill({iv_put_270}, 60)), -1)",
                        "options",
                        (pcr_oi_270, iv_call_270, iv_put_270),
                        ("trade_when", "rank", "ts_backfill"),
                        "Trades option IV spread only when call-open-interest pressure dominates.",
                    )
                )
            if pcr_oi_720 and iv_call_720 and iv_put_720:
                candidates.append(
                    self._candidate(
                        f"trade_when({pcr_oi_720} < 0.4, rank(ts_backfill({iv_call_720}, 60) - ts_backfill({iv_put_720}, 60)), -1)",
                        "options",
                        (pcr_oi_720, iv_call_720, iv_put_720),
                        ("trade_when", "rank", "ts_backfill"),
                        "Uses long-dated put/call pressure to gate call-put IV skew.",
                    )
                )

        if "model_risk" in strategies:
            if rel_ret_all and returns:
                candidates.append(
                    self._candidate(
                        f"rank(ts_mean({rel_ret_all}, 20) - ts_mean({returns}, 60))",
                        "model_risk",
                        (rel_ret_all, returns),
                        ("rank", "ts_mean"),
                        "Compares model-relative return strength with medium return behavior.",
                    )
                )
            if parkinson_vol:
                candidates.append(
                    self._candidate(
                        f"rank(0 - ts_rank(ts_backfill({parkinson_vol}, 20), 60))",
                        "model_risk",
                        (parkinson_vol,),
                        ("rank", "ts_rank", "ts_backfill"),
                        "Prefers lower realized-risk rank after short backfill.",
                    )
                )
            if model_quality and model_value:
                candidates.append(
                    self._candidate(
                        f"group_rank(ts_rank({model_quality}, 120) + ts_rank({model_value}, 120), subindustry)",
                        "model_risk",
                        (model_quality, model_value),
                        ("group_rank", "ts_rank"),
                        "Combines quality and value model scores within a risk group.",
                    )
                )
            if model_momentum and model_reversal:
                candidates.append(
                    self._candidate(
                        f"rank(ts_rank({model_momentum}, 60) - ts_rank({model_reversal}, 60))",
                        "model_risk",
                        (model_momentum, model_reversal),
                        ("rank", "ts_rank"),
                        "Blends model trend and reversal scores to avoid one-note exposure.",
                    )
                )
            if model_credit and model_default:
                candidates.append(
                    self._candidate(
                        f"rank(ts_rank({model_credit}, 120) - ts_rank({model_default}, 120))",
                        "model_risk",
                        (model_credit, model_default),
                        ("rank", "ts_rank"),
                        "Favors stronger credit score and lower default-risk rank.",
                    )
                )

        if "hybrid" in strategies and price and volume:
            if returns:
                candidates.append(
                    self._candidate(
                        f"zscore(rank((ts_mean({price}, 40) - {price}) / (ts_std_dev({price}, 40) + 0.001)) + rank(ts_corr({returns}, {volume}, 20)))",
                        "hybrid",
                        (price, returns, volume),
                        ("zscore", "rank", "ts_mean", "ts_std_dev", "ts_corr"),
                        "Blends price reversion with short return-volume structure.",
                    )
                )
            if ebit and assets:
                candidates.append(
                    self._candidate(
                        f"zscore(rank({ebit} / (abs({assets}) + 0.001)) + rank(ts_corr({price}, {volume}, 20)))",
                        "hybrid",
                        (price, volume, ebit, assets),
                        ("zscore", "rank", "abs", "ts_corr"),
                        "Blends quality with price-volume behavior.",
                    )
                )

        if "decorrelation" in strategies:
            if price:
                for field in ("open", "high", "low", "vwap"):
                    if field in allowed_fields and field != price:
                        candidates.append(
                            self._candidate(
                                f"rank((ts_mean({field}, 40) - {field}) / (ts_std_dev({field}, 40) + 0.001))",
                                "decorrelation",
                                (field,),
                                ("rank", "ts_mean", "ts_std_dev"),
                                f"Equivalent reversion idea using {field} to reduce correlation to close-based formulas.",
                            )
                        )
            if returns and volume:
                for window in (10, 60, 120):
                    candidates.append(
                        self._candidate(
                            f"group_neutralize(rank(ts_corr({returns}, {volume}, {window})), {self.random.choice(self.GROUPS)})",
                            "decorrelation",
                            (returns, volume),
                            ("group_neutralize", "rank", "ts_corr"),
                            "Changes horizon/grouping to preserve idea while reducing correlation.",
                        )
                    )

        candidates.extend(self._generic_field_candidates(allowed_fields, strategies))
        return candidates

    def _generic_field_candidates(
        self,
        allowed_fields: Set[str],
        strategies: Set[str],
    ) -> List[AlphaCandidate]:
        """Create field-first candidates from live/synced dataset fields."""
        if not strategies.intersection(
            {"fundamental", "quality", "analyst", "sentiment", "options", "model_risk", "hybrid"}
        ):
            return []
        if len(allowed_fields) > 80:
            return []

        fields = [field for field in sorted(allowed_fields) if self._is_transformable_field(field)]
        if len(fields) > 40:
            fields = self.random.sample(fields, 40)

        candidates: List[AlphaCandidate] = []
        for field in fields[:14]:
            field_type = self._field_type(field)
            processed, pre_ops = self._processed_field(field, field_type)
            strategy = self._strategy_from_field(field, strategies)
            candidates.append(
                self._candidate(
                    f"rank(ts_rank({processed}, 60))",
                    strategy,
                    (field,),
                    ("rank", "ts_rank", *pre_ops),
                    f"Field-first transformation of {field} using backfill/vector-safe preprocessing.",
                )
            )
            if field_type != "GROUP":
                candidates.append(
                    self._candidate(
                        f"winsorize(rank(ts_zscore({processed}, 60)))",
                        strategy,
                        (field,),
                        ("winsorize", "rank", "ts_zscore", *pre_ops),
                        f"Normalizes {field} over time with outlier control.",
                    )
                )
        return candidates

    def _random_candidate(
        self,
        allowed_fields: Set[str],
        strategies: Set[str],
    ) -> AlphaCandidate:
        strategy = self.random.choice(sorted(strategies))
        candidates = self._build_candidates(allowed_fields, {strategy})
        if candidates:
            candidate = self.random.choice(candidates)
            window = self.random.choice(self.WINDOWS_SHORT + self.WINDOWS_MEDIUM + self.WINDOWS_LONG)
            expression = candidate.expression.replace(", 20)", f", {window})", 1)
            return AlphaCandidate(
                expression=expression,
                strategy=candidate.strategy,
                source_fields=candidate.source_fields,
                dataset_ids=candidate.dataset_ids,
                operators=candidate.operators,
                rationale=candidate.rationale,
                score=candidate.score,
            )

        field = self._preferred(allowed_fields, ("close", "returns", "volume")) or sorted(allowed_fields)[0]
        fallback_strategy = strategy if strategy in STRATEGY_DESCRIPTIONS else "momentum"
        return self._candidate(
            f"rank(ts_rank({field}, 20))",
            fallback_strategy,
            (field,),
            ("rank", "ts_rank"),
            "Fallback rank of a valid source field.",
        )

    def _neutralized_variant(self, candidate: AlphaCandidate) -> AlphaCandidate:
        group = self.random.choice(self.GROUPS)
        return AlphaCandidate(
            expression=f"group_neutralize({candidate.expression}, {group})",
            strategy=candidate.strategy,
            source_fields=candidate.source_fields,
            dataset_ids=candidate.dataset_ids,
            operators=tuple(dict.fromkeys(candidate.operators + ("group_neutralize",))),
            rationale=f"{candidate.rationale} Neutralized by {group}.",
            score=min(1.0, candidate.score + 0.08),
        )

    def _candidate(
        self,
        expression: str,
        strategy: str,
        source_fields: Sequence[str],
        operators: Sequence[str],
        rationale: str,
    ) -> AlphaCandidate:
        return AlphaCandidate(
            expression=expression,
            strategy=strategy,
            source_fields=tuple(dict.fromkeys(source_fields)),
            dataset_ids=self._dataset_ids_for_fields(source_fields),
            operators=tuple(dict.fromkeys(operators)),
            rationale=rationale,
            score=self._score(strategy, source_fields, operators),
        )

    def _score(
        self,
        strategy: str,
        source_fields: Sequence[str],
        operators: Sequence[str],
    ) -> float:
        score = 0.48
        if strategy in {"momentum", "mean_reversion", "price_volume"}:
            score += 0.10
        if strategy in {"analyst", "sentiment", "options", "model_risk"}:
            score += 0.08
        if len(set(source_fields)) > 1:
            score += 0.08
        if any(operator.startswith("ts_") for operator in operators):
            score += 0.10
        if any(operator.startswith("vec_") for operator in operators):
            score += 0.06
        if "ts_backfill" in operators:
            score += 0.04
        if "group_neutralize" in operators:
            score += 0.08
        return round(min(score, 0.95), 3)

    def _dataset_ids_for_fields(self, source_fields: Sequence[str]) -> tuple[str, ...]:
        dataset_ids = []
        for field in source_fields:
            metadata = self.schema.field_info(field) or {}
            for dataset_id in metadata.get("datasets") or []:
                if dataset_id and dataset_id not in dataset_ids:
                    dataset_ids.append(dataset_id)
        for dataset_id in datasets_for_fields(source_fields):
            if dataset_id not in dataset_ids:
                dataset_ids.append(dataset_id)
        return tuple(dataset_ids)

    def _strategy_from_field(self, field: str, strategies: Set[str]) -> str:
        dataset_ids = self._dataset_ids_for_fields((field,))
        dataset_text = " ".join(dataset_ids)
        text = f"{field} {dataset_text}".lower()
        preferences = [
            ("options", ("implied_volatility", "pcr_", "option")),
            ("analyst", ("est_", "anl", "etz", "analyst")),
            ("sentiment", ("news_", "scl", "snt", "rpna", "sentiment")),
            ("model_risk", ("mdl", "beta", "risk", "model", "volatility")),
            ("quality", ("ebit", "income", "cashflow", "assets", "debt", "fnd", "fundamental")),
        ]
        for strategy, tokens in preferences:
            if strategy in strategies and any(token in text for token in tokens):
                return strategy
        return next(iter(sorted(strategies.intersection(STRATEGY_DESCRIPTIONS) or {"hybrid"})))

    def _field_type(self, field: str) -> str:
        metadata = self.schema.field_info(field) or {}
        return str(metadata.get("type") or "MATRIX").strip().upper()

    @staticmethod
    def _processed_field(field: str, field_type: str) -> tuple[str, tuple[str, ...]]:
        if field_type == "VECTOR":
            return f"vec_avg({field})", ("vec_avg",)
        return f"ts_backfill({field}, 120)", ("ts_backfill",)

    def _is_transformable_field(self, field: str) -> bool:
        field_type = self._field_type(field)
        if field_type in {"GROUP", "UNIVERSE"}:
            return False
        return field not in {"sector", "industry", "subindustry", "market"}

    @staticmethod
    def _preferred(fields: Set[str], choices: Sequence[Optional[str]]) -> Optional[str]:
        for choice in choices:
            if choice and choice in fields:
                return choice
        return None


def available_strategies() -> List[dict]:
    """Return metadata for supported rule-based strategies."""
    return [
        {"name": name, "description": description}
        for name, description in sorted(STRATEGY_DESCRIPTIONS.items())
    ]
