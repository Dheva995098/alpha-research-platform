"""Database-backed ML ranking service."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from backend.ml.features import FEATURE_NAMES
from backend.ml.ranker import AlphaRanker, PredictionResult, TrainingExample, TrainingResult
from backend.models import AlphaLibrary, AttemptMemory, LeaderboardAlpha, MLModelRecord, Result


class MLRankingService:
    """Score expressions and train/load lightweight ranker state."""

    MODEL_NAME = "alpha_ranker"

    def __init__(self, db: Optional[Session] = None, ranker: Optional[AlphaRanker] = None):
        self.db = db
        self.ranker = ranker or self._load_ranker(db)

    def score_expression(self, expression: str, metrics: Optional[Dict] = None) -> PredictionResult:
        """Score one expression."""
        return self.ranker.predict(expression, metrics=metrics)

    def score_expressions(self, expressions: Iterable[str]) -> List[PredictionResult]:
        """Score and rank expressions."""
        return self.ranker.predict_many(expressions)

    def score_results(self, limit: int = 500, only_unscored: bool = True) -> List[Result]:
        """Update ml_pass_probability/final_score on stored results."""
        if self.db is None:
            raise RuntimeError("Database session is required to score stored results")

        query = self.db.query(Result).order_by(Result.id.desc()).limit(limit)
        if only_unscored:
            query = self.db.query(Result).filter(Result.ml_pass_probability.is_(None)).order_by(Result.id.desc()).limit(limit)

        results = query.all()
        for result in results:
            prediction = self.ranker.predict(result.expression, metrics=self.metrics_from_result(result))
            result.ml_pass_probability = prediction.pass_probability
            heuristic_score = result.final_score or 0.0
            result.final_score = round(heuristic_score * 0.50 + prediction.score * 0.50, 4)

        self.db.commit()
        for result in results:
            self.db.refresh(result)
        return results

    def train_from_db(self, min_examples: int = 5) -> TrainingResult:
        """Train from leaderboard and completed result rows."""
        if self.db is None:
            raise RuntimeError("Database session is required to train from stored examples")

        examples = self._training_examples_from_db()
        if len(examples) < min_examples:
            return TrainingResult(
                trained=False,
                example_count=len(examples),
                positive_count=sum(example.label for example in examples),
                negative_count=len(examples) - sum(example.label for example in examples),
                accuracy=None,
                message=f"Need at least {min_examples} training examples",
            )

        training_result = self.ranker.train(examples)
        if training_result.trained:
            self._save_ranker(training_result)
        return training_result

    def model_status(self) -> Dict:
        """Return current model metadata."""
        state = self.ranker.state_dict()
        active_record = None
        if self.db is not None:
            active_record = (
                self.db.query(MLModelRecord)
                .filter(MLModelRecord.name == self.MODEL_NAME, MLModelRecord.is_active == True)
                .order_by(MLModelRecord.id.desc())
                .first()
            )

        return {
            "model_name": self.MODEL_NAME,
            "model_type": state["model_type"],
            "feature_names": state["feature_names"],
            "trained": active_record is not None,
            "trained_on_count": active_record.trained_on_count if active_record else 0,
            "metrics": active_record.metrics if active_record else {},
        }

    def ranked_results(self, limit: int = 100) -> List[Result]:
        """Return results ordered by final score, highest first."""
        if self.db is None:
            raise RuntimeError("Database session is required to list ranked results")
        return (
            self.db.query(Result)
            .order_by(Result.final_score.desc().nullslast(), Result.id.desc())
            .limit(limit)
            .all()
        )

    def _training_examples_from_db(self) -> List[TrainingExample]:
        examples: List[TrainingExample] = []
        leaderboard_rows = self.db.query(LeaderboardAlpha).all()
        for row in leaderboard_rows:
            examples.append(
                TrainingExample(
                    expression=row.expression,
                    label=1 if row.passes_checks else 0,
                    metrics={
                        "sharpe": row.sharpe,
                        "fitness": row.fitness,
                        "turnover": row.turnover,
                        "self_correlation": row.self_correlation,
                        "all_checks_passed": row.passes_checks,
                    },
                    weight=1.25,
                )
            )

        result_rows = self.db.query(Result).filter(Result.all_checks_passed.isnot(None)).all()
        for row in result_rows:
            raw_metrics = row.raw_metrics if isinstance(row.raw_metrics, dict) else {}
            if raw_metrics.get("dry_run") or raw_metrics.get("source") == "dry_run":
                continue
            label = self._label_from_result(row)
            examples.append(
                TrainingExample(
                    expression=row.expression,
                    label=label,
                    metrics=self.metrics_from_result(row),
                    weight=self._result_training_weight(row),
                )
            )

        examples.extend(self._self_improving_examples())
        return examples

    def _self_improving_examples(self) -> List[TrainingExample]:
        """Positives from the confirmed-win library + hard negatives from near-misses.

        Library winners are the system's own ground truth of good alphas; near-misses
        are the most information-dense negatives (looked good, still failed), so they
        sharpen the ranker's precision exactly where quota is wasted. Inert until the
        self-improving loop has accumulated data.
        """
        out: List[TrainingExample] = []
        try:
            for row in self.db.query(AlphaLibrary).all():
                if not row.expression:
                    continue
                out.append(
                    TrainingExample(
                        expression=row.expression,
                        label=1,
                        metrics={
                            "sharpe": row.sharpe,
                            "fitness": row.fitness,
                            "turnover": row.turnover,
                            "self_correlation": row.self_correlation,
                            "all_checks_passed": True,
                        },
                        weight=1.5,
                    )
                )
            near_rows = (
                self.db.query(AttemptMemory)
                .filter(AttemptMemory.outcome == "near")
                .order_by(AttemptMemory.score.desc())
                .limit(200)
                .all()
            )
            for row in near_rows:
                if not row.expression:
                    continue
                out.append(
                    TrainingExample(
                        expression=row.expression,
                        label=0,  # hard negative: confidently near, still not a pass
                        metrics={
                            "sharpe": row.sharpe,
                            "fitness": row.fitness,
                            "turnover": row.turnover,
                            "self_correlation": row.self_correlation,
                            "all_checks_passed": False,
                        },
                        weight=2.0,
                    )
                )
        except Exception:
            return out
        return out

    def _save_ranker(self, training_result: TrainingResult) -> MLModelRecord:
        self.db.query(MLModelRecord).filter(MLModelRecord.name == self.MODEL_NAME).update({"is_active": False})
        state = self.ranker.state_dict()
        record = MLModelRecord(
            name=self.MODEL_NAME,
            version="v1",
            model_type=state["model_type"],
            feature_names=state["feature_names"],
            weights=state["weights"],
            bias=state["bias"],
            metrics={
                "accuracy": training_result.accuracy,
                "positive_count": training_result.positive_count,
                "negative_count": training_result.negative_count,
            },
            trained_on_count=training_result.example_count,
            is_active=True,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    @classmethod
    def _load_ranker(cls, db: Optional[Session]) -> AlphaRanker:
        if db is None:
            return AlphaRanker()

        record = (
            db.query(MLModelRecord)
            .filter(MLModelRecord.name == cls.MODEL_NAME, MLModelRecord.is_active == True)
            .order_by(MLModelRecord.id.desc())
            .first()
        )
        if record is None:
            return AlphaRanker()

        stored_weights = record.weights if isinstance(record.weights, dict) else {}
        upgraded_weights = {**AlphaRanker.DEFAULT_WEIGHTS, **stored_weights}
        return AlphaRanker.from_state(
            {
                "feature_names": FEATURE_NAMES,
                "weights": upgraded_weights,
                "bias": record.bias,
            }
        )

    @staticmethod
    def _label_from_result(result: Result) -> int:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        if raw_metrics.get("source") == "training_seed":
            return 1 if result.all_checks_passed is True else 0
        if result.human_approved and MLRankingService._metrics_clear_quality_bar(result):
            return 0 if MLRankingService._has_failed_checks(result) else 1
        if result.all_checks_passed is False:
            return 0
        if result.all_checks_passed is True and MLRankingService._metrics_clear_quality_bar(result):
            return 1
        return 0

    @staticmethod
    def metrics_from_result(result: Result) -> Dict:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        is_block = raw_metrics.get("is") if isinstance(raw_metrics.get("is"), dict) else {}
        checks = is_block.get("checks") or raw_metrics.get("checks")
        return {
            "sharpe": result.sharpe,
            "fitness": result.fitness,
            "turnover": result.turnover,
            "self_correlation": result.self_correlation,
            "all_checks_passed": result.all_checks_passed,
            "raw_metrics": raw_metrics,
            "checks": checks,
            "grade": raw_metrics.get("grade"),
            "settings": raw_metrics.get("settings"),
        }

    @staticmethod
    def _result_training_weight(result: Result) -> float:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        is_block = raw_metrics.get("is") if isinstance(raw_metrics.get("is"), dict) else {}
        checks = is_block.get("checks") or raw_metrics.get("checks") or []
        failed_checks = [
            check
            for check in checks
            if isinstance(check, dict) and str(check.get("result") or "").upper() == "FAIL"
        ]
        weight = 4.0 if raw_metrics.get("source") == "training_seed" else 2.0
        if result.all_checks_passed is True:
            weight += 1.25
        if result.human_approved:
            weight += 1.0
        weight += min(len(failed_checks) * 0.30, 1.50)
        if result.sharpe is not None:
            weight += min(abs(float(result.sharpe)) * 0.20, 0.80)
        if result.fitness is not None:
            weight += min(abs(float(result.fitness)) * 0.20, 0.80)
        return round(weight, 4)

    @staticmethod
    def _metrics_clear_quality_bar(result: Result) -> bool:
        return bool(
            (result.sharpe or 0) >= 1.25
            and (result.fitness or 0) >= 1.0
            and (result.turnover is None or result.turnover <= 0.70)
            and (result.self_correlation is None or result.self_correlation <= 0.70)
        )

    @staticmethod
    def _has_failed_checks(result: Result) -> bool:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        is_block = raw_metrics.get("is") if isinstance(raw_metrics.get("is"), dict) else {}
        checks = is_block.get("checks") or raw_metrics.get("checks") or []
        return any(
            isinstance(check, dict) and str(check.get("result") or "").upper() == "FAIL"
            for check in checks
        )
