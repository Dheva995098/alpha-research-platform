"""Lightweight alpha pass-probability ranker."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence

from backend.core.data_fields import BRAINDataFields, get_data_fields
from backend.ml.features import ExpressionFeatureExtractor, FEATURE_NAMES


@dataclass(frozen=True)
class PredictionResult:
    """Prediction for one expression."""

    expression: str
    pass_probability: float
    score: float
    feature_values: Dict[str, float]
    reasons: List[str]


@dataclass(frozen=True)
class TrainingExample:
    """Labeled training example."""

    expression: str
    label: int
    metrics: Optional[Dict] = None
    weight: float = 1.0


@dataclass(frozen=True)
class TrainingResult:
    """Training summary."""

    trained: bool
    example_count: int
    positive_count: int
    negative_count: int
    accuracy: Optional[float]
    message: str


class AlphaRanker:
    """Heuristic-first, trainable logistic alpha ranker."""

    DEFAULT_WEIGHTS = {
        "operator_count": 0.35,
        "time_series_operator_count": 0.55,
        "group_operator_count": 0.35,
        "window_count": 0.20,
        "has_group_neutralize": 0.32,
        "has_price_volume_pair": 0.25,
        "has_fundamental_pair": 0.30,
        "has_reversion_shape": 0.24,
        "has_outlier_control": 0.18,
        "has_trade_when": 0.18,
        "has_backfill": 0.16,
        "has_group_rank": 0.20,
        "has_options_data": 0.10,
        "has_analyst_data": 0.14,
        "has_sentiment_data": 0.10,
        "has_model_risk_data": 0.16,
        "has_alternative_data": 0.12,
        "nesting_depth": -0.20,
        "expression_length": -0.18,
        "constant_count": -0.10,
        "result_sharpe": 0.95,
        "result_fitness": 0.90,
        "result_turnover": -0.55,
        "result_self_correlation": -0.45,
        "result_checks_passed": 1.10,
        "result_check_pass_rate": 1.35,
        "result_check_fail_count": -1.15,
        "result_check_pending_count": -0.25,
        "failed_low_sharpe": -1.05,
        "failed_low_fitness": -0.95,
        "failed_low_sub_universe_sharpe": -0.55,
        "failed_turnover": -0.60,
        "pending_self_correlation": -0.15,
        "result_grade_score": 0.85,
        "setting_decay": 0.10,
        "setting_truncation": 0.22,
        "setting_delay": 0.04,
        "setting_region_usa": 0.02,
        "setting_region_chn": 0.02,
        "setting_universe_top3000": 0.04,
        "setting_universe_top1000": 0.06,
        "setting_universe_top500": 0.08,
        "setting_universe_top200": 0.10,
        "setting_neutralization_subindustry": 0.18,
        "setting_neutralization_sector": 0.08,
        "setting_neutralization_industry": 0.12,
        "setting_neutralization_market": 0.05,
        "setting_neutralization_none": 0.02,
        "setting_max_trade_off": 0.14,
        "setting_options_profile": 0.28,
    }

    def __init__(
        self,
        schema: Optional[BRAINDataFields] = None,
        feature_names: Sequence[str] = FEATURE_NAMES,
        weights: Optional[Dict[str, float]] = None,
        bias: float = -0.20,
    ):
        self.schema = schema or get_data_fields()
        self.extractor = ExpressionFeatureExtractor(self.schema)
        self.feature_names = list(feature_names)
        self.weights = {name: float((weights or self.DEFAULT_WEIGHTS).get(name, 0.0)) for name in self.feature_names}
        self.bias = float(bias)

    def predict(self, expression: str, metrics: Optional[Dict] = None) -> PredictionResult:
        """Predict pass probability for one expression."""
        features = self.extractor.extract(expression, metrics=metrics)
        logit = self.bias
        for name in self.feature_names:
            logit += self.weights.get(name, 0.0) * features.values.get(name, 0.0)
        probability = self._sigmoid(logit)
        score = self._score(probability, features.values)

        return PredictionResult(
            expression=expression,
            pass_probability=round(probability, 4),
            score=round(score, 4),
            feature_values=features.values,
            reasons=self._reasons(features.values),
        )

    def predict_many(self, expressions: Iterable[str]) -> List[PredictionResult]:
        """Predict and rank expressions descending by score."""
        predictions = [self.predict(expression) for expression in expressions]
        return sorted(predictions, key=lambda prediction: prediction.score, reverse=True)

    def train(
        self,
        examples: Sequence[TrainingExample],
        epochs: int = 700,
        learning_rate: float = 0.05,
    ) -> TrainingResult:
        """Train logistic weights with weighted gradient descent."""
        valid_examples = [example for example in examples if example.expression and example.label in {0, 1}]
        positive_count = sum(example.label for example in valid_examples)
        negative_count = len(valid_examples) - positive_count

        if len(valid_examples) < 5 or positive_count == 0 or negative_count == 0:
            return TrainingResult(
                trained=False,
                example_count=len(valid_examples),
                positive_count=positive_count,
                negative_count=negative_count,
                accuracy=None,
                message="Need at least 5 labeled examples with both positive and negative labels",
            )

        for _ in range(max(1, epochs)):
            bias_gradient = 0.0
            gradients = {name: 0.0 for name in self.feature_names}
            total_weight = 0.0
            for example in valid_examples:
                features = self.extractor.extract(example.expression, metrics=example.metrics)
                prediction = self._predict_raw(features.values)
                weight = max(float(example.weight or 1.0), 0.05)
                total_weight += weight
                error = (prediction - example.label) * weight
                bias_gradient += error
                for name in self.feature_names:
                    gradients[name] += error * features.values.get(name, 0.0)

            size = max(total_weight, 1.0)
            self.bias -= learning_rate * (bias_gradient / size)
            for name in self.feature_names:
                self.weights[name] -= learning_rate * (gradients[name] / size)

        accuracy = self._accuracy(valid_examples)
        return TrainingResult(
            trained=True,
            example_count=len(valid_examples),
            positive_count=positive_count,
            negative_count=negative_count,
            accuracy=round(accuracy, 4),
            message="Trained logistic alpha ranker",
        )

    def state_dict(self) -> Dict:
        """Serialize model state."""
        return {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "model_type": "logistic",
        }

    @classmethod
    def from_state(cls, state: Dict, schema: Optional[BRAINDataFields] = None) -> "AlphaRanker":
        """Create a ranker from serialized state."""
        return cls(
            schema=schema,
            feature_names=state.get("feature_names", FEATURE_NAMES),
            weights=state.get("weights", cls.DEFAULT_WEIGHTS),
            bias=state.get("bias", -0.20),
        )

    def _predict_raw(self, feature_values: Dict[str, float]) -> float:
        logit = self.bias
        for name in self.feature_names:
            logit += self.weights.get(name, 0.0) * feature_values.get(name, 0.0)
        return self._sigmoid(logit)

    def _accuracy(self, examples: Sequence[TrainingExample]) -> float:
        correct = 0
        for example in examples:
            probability = self.predict(example.expression, metrics=example.metrics).pass_probability
            label = 1 if probability >= 0.5 else 0
            if label == example.label:
                correct += 1
        return correct / len(examples)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1 / (1 + z)
        z = math.exp(value)
        return z / (1 + z)

    @staticmethod
    def _score(probability: float, feature_values: Dict[str, float]) -> float:
        result_bonus = (
            feature_values.get("result_sharpe", 0.0) * 0.20
            + feature_values.get("result_fitness", 0.0) * 0.20
            + feature_values.get("result_checks_passed", 0.0) * 0.10
        )
        turnover_penalty = max(feature_values.get("result_turnover", 0.0) - 0.70, 0.0) * 0.15
        return probability + result_bonus - turnover_penalty

    @staticmethod
    def _reasons(feature_values: Dict[str, float]) -> List[str]:
        reasons: List[str] = []
        if feature_values.get("time_series_operator_count", 0.0) > 0:
            reasons.append("uses time-series structure")
        if feature_values.get("has_group_neutralize", 0.0) > 0:
            reasons.append("includes group neutralization")
        if feature_values.get("has_price_volume_pair", 0.0) > 0:
            reasons.append("combines price and volume information")
        if feature_values.get("has_fundamental_pair", 0.0) > 0:
            reasons.append("uses fundamental scale normalization")
        if feature_values.get("has_reversion_shape", 0.0) > 0:
            reasons.append("uses normalized mean-reversion structure")
        if feature_values.get("has_outlier_control", 0.0) > 0:
            reasons.append("controls extreme values")
        if feature_values.get("has_trade_when", 0.0) > 0:
            reasons.append("uses event-gated trading")
        if feature_values.get("has_backfill", 0.0) > 0:
            reasons.append("handles sparse data with backfill")
        if feature_values.get("has_group_rank", 0.0) > 0:
            reasons.append("ranks within a risk group")
        if feature_values.get("has_analyst_data", 0.0) > 0:
            reasons.append("uses analyst estimate data")
        if feature_values.get("has_options_data", 0.0) > 0:
            reasons.append("uses options-implied information")
        if feature_values.get("has_sentiment_data", 0.0) > 0:
            reasons.append("uses news or sentiment information")
        if feature_values.get("has_model_risk_data", 0.0) > 0:
            reasons.append("uses model or risk dataset fields")
        if feature_values.get("has_alternative_data", 0.0) > 0 and feature_values.get("has_backfill", 0.0) > 0:
            reasons.append("handles sparse alternative data")
        if feature_values.get("result_checks_passed", 0.0) > 0:
            reasons.append("historical checks passed")
        if feature_values.get("result_check_pass_rate", 0.0) >= 0.75:
            reasons.append("most live checks passed")
        if feature_values.get("failed_low_sharpe", 0.0) > 0:
            reasons.append("live history failed Sharpe")
        if feature_values.get("failed_low_fitness", 0.0) > 0:
            reasons.append("live history failed fitness")
        if feature_values.get("failed_low_sub_universe_sharpe", 0.0) > 0:
            reasons.append("live history failed sub-universe Sharpe")
        if feature_values.get("result_grade_score", 0.0) < 0:
            reasons.append("weak BRAIN grade history")
        if feature_values.get("setting_truncation", 0.0) >= 0.75:
            reasons.append("uses aggressive truncation setting")
        if feature_values.get("setting_options_profile", 0.0) > 0:
            reasons.append("matches learned options-data settings")
        if feature_values.get("setting_universe_top500", 0.0) > 0 or feature_values.get("setting_universe_top200", 0.0) > 0:
            reasons.append("uses a more liquid universe setting")
        if feature_values.get("setting_neutralization_industry", 0.0) > 0:
            reasons.append("uses industry neutralization")
        if feature_values.get("setting_neutralization_market", 0.0) > 0:
            reasons.append("uses market neutralization")
        if feature_values.get("nesting_depth", 0.0) > 0.6:
            reasons.append("complex expression depth adds risk")
        return reasons or ["baseline expression-quality estimate"]
