"""Filtering pipeline for generated expressions and backtest results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from backend.core.data_fields import BRAINDataFields, get_data_fields
from backend.generation.dedup import ExpressionDeduplicator
from backend.ml.features import ExpressionFeatureExtractor
from backend.ml.ranker import AlphaRanker
from backend.models import Result


@dataclass(frozen=True)
class ExpressionFilterConfig:
    """Thresholds for expression-level filtering."""

    require_unique: bool = True
    min_ml_probability: float = 0.45
    max_expression_length: int = 1000
    max_nesting_depth: int = 8
    max_operator_count: int = 16
    max_constant_count: int = 20


@dataclass(frozen=True)
class ResultFilterConfig:
    """Thresholds for completed result filtering."""

    min_sharpe: float = 1.0
    min_fitness: float = 0.8
    max_turnover: float = 0.70
    max_self_correlation: float = 0.70
    require_checks_passed: bool = True
    min_ml_probability: float = 0.50


@dataclass(frozen=True)
class FilterDecision:
    """Pass/fail decision for one item."""

    item_id: Optional[int]
    expression: str
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterSummary:
    """Summary of a filtering run."""

    total_count: int
    accepted_count: int
    rejected_count: int
    accepted: List[FilterDecision]
    rejected: List[FilterDecision]


class AlphaFilterPipeline:
    """Apply validation, duplicate, ML, and metric gates."""

    def __init__(
        self,
        schema: Optional[BRAINDataFields] = None,
        ranker: Optional[AlphaRanker] = None,
    ):
        self.schema = schema or get_data_fields()
        self.ranker = ranker or AlphaRanker(schema=self.schema)
        self.extractor = ExpressionFeatureExtractor(self.schema)

    def filter_expressions(
        self,
        expressions: Sequence[str],
        existing_expressions: Optional[Iterable[str]] = None,
        config: Optional[ExpressionFilterConfig] = None,
    ) -> FilterSummary:
        """Filter generated expressions before queueing/submission."""
        config = config or ExpressionFilterConfig()
        deduplicator = ExpressionDeduplicator(existing_expressions)
        accepted: List[FilterDecision] = []
        rejected: List[FilterDecision] = []

        for raw_expression in expressions:
            expression = (raw_expression or "").strip()
            reasons: List[str] = []
            metrics: Dict[str, float | bool | None] = {}

            valid, message = self.schema.validate_expression_basic(expression)
            if not valid:
                reasons.append(message)

            if config.require_unique:
                is_unique, duplicate_of, _ = deduplicator.add(expression)
                if not is_unique:
                    reasons.append(f"Duplicate expression of: {duplicate_of}")

            features = self.extractor.extract(expression)
            operator_count = int(features.values["operator_count"] * 10)
            constant_count = int(features.values["constant_count"] * 10)
            nesting_depth = int(features.values["nesting_depth"] * 10)
            prediction = self.ranker.predict(expression)
            metrics.update(
                {
                    "ml_pass_probability": prediction.pass_probability,
                    "operator_count": operator_count,
                    "constant_count": constant_count,
                    "nesting_depth": nesting_depth,
                    "expression_length": len(expression),
                }
            )

            if len(expression) > config.max_expression_length:
                reasons.append(f"Expression length exceeds {config.max_expression_length}")
            if nesting_depth > config.max_nesting_depth:
                reasons.append(f"Nesting depth exceeds {config.max_nesting_depth}")
            if operator_count > config.max_operator_count:
                reasons.append(f"Operator count exceeds {config.max_operator_count}")
            if constant_count > config.max_constant_count:
                reasons.append(f"Constant count exceeds {config.max_constant_count}")
            if prediction.pass_probability < config.min_ml_probability:
                reasons.append(
                    f"ML probability {prediction.pass_probability:.2f} below {config.min_ml_probability:.2f}"
                )

            decision = FilterDecision(
                item_id=None,
                expression=expression,
                passed=not reasons,
                reasons=reasons,
                metrics=metrics,
            )
            if decision.passed:
                accepted.append(decision)
            else:
                rejected.append(decision)

        return FilterSummary(
            total_count=len(expressions),
            accepted_count=len(accepted),
            rejected_count=len(rejected),
            accepted=accepted,
            rejected=rejected,
        )

    def filter_results(
        self,
        results: Sequence[Result],
        config: Optional[ResultFilterConfig] = None,
    ) -> FilterSummary:
        """Filter completed backtest results by quality gates."""
        config = config or ResultFilterConfig()
        accepted: List[FilterDecision] = []
        rejected: List[FilterDecision] = []

        for result in results:
            reasons: List[str] = []
            metrics = {
                "sharpe": result.sharpe,
                "fitness": result.fitness,
                "turnover": result.turnover,
                "self_correlation": result.self_correlation,
                "all_checks_passed": result.all_checks_passed,
                "ml_pass_probability": result.ml_pass_probability,
                "final_score": result.final_score,
            }

            if (result.sharpe or 0.0) < config.min_sharpe:
                reasons.append(f"Sharpe below {config.min_sharpe}")
            if (result.fitness or 0.0) < config.min_fitness:
                reasons.append(f"Fitness below {config.min_fitness}")
            if result.turnover is not None and result.turnover > config.max_turnover:
                reasons.append(f"Turnover above {config.max_turnover}")
            if result.self_correlation is not None and result.self_correlation > config.max_self_correlation:
                reasons.append(f"Self-correlation above {config.max_self_correlation}")
            if config.require_checks_passed and result.all_checks_passed is not True:
                reasons.append("BRAIN checks did not pass")

            ml_probability = result.ml_pass_probability
            if ml_probability is None:
                prediction = self.ranker.predict(
                    result.expression,
                    metrics={
                        "sharpe": result.sharpe,
                        "fitness": result.fitness,
                        "turnover": result.turnover,
                        "self_correlation": result.self_correlation,
                        "all_checks_passed": result.all_checks_passed,
                    },
                )
                ml_probability = prediction.pass_probability
                metrics["ml_pass_probability"] = ml_probability

            if ml_probability < config.min_ml_probability:
                reasons.append(f"ML probability {ml_probability:.2f} below {config.min_ml_probability:.2f}")

            decision = FilterDecision(
                item_id=result.id,
                expression=result.expression,
                passed=not reasons,
                reasons=reasons,
                metrics=metrics,
            )
            if decision.passed:
                accepted.append(decision)
            else:
                rejected.append(decision)

        return FilterSummary(
            total_count=len(results),
            accepted_count=len(accepted),
            rejected_count=len(rejected),
            accepted=accepted,
            rejected=rejected,
        )
