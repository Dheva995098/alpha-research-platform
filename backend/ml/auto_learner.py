"""Automatic feedback loop for model training and result learning."""
from __future__ import annotations

from collections import defaultdict
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from backend.ml.service import MLRankingService
from backend.models import Result


@dataclass(frozen=True)
class LearnedFocusStat:
    """Aggregated result quality for one expression family."""

    name: str
    count: int
    passed: int
    pass_rate: float
    avg_sharpe: Optional[float]
    avg_fitness: Optional[float]
    avg_score: Optional[float]


@dataclass(frozen=True)
class LearnedSettingsStat:
    """Aggregated result quality for one simulation-settings cluster."""

    settings: Dict[str, Any]
    count: int
    passed: int
    pass_rate: float
    avg_sharpe: Optional[float]
    avg_fitness: Optional[float]
    avg_score: Optional[float]


class AutoLearningService:
    """Train, rescore, and summarize the local alpha feedback loop."""

    def __init__(self, db: Session):
        self.db = db

    def run_once(
        self,
        limit: int = 500,
        min_examples: int = 5,
    ) -> Dict[str, Any]:
        """Retrain from stored examples, refresh result scores, and return status."""
        ranking = MLRankingService(self.db)
        training_result = ranking.train_from_db(min_examples=min_examples)
        scored_results = ranking.score_results(limit=limit, only_unscored=False)
        summary = self.status(limit=limit)
        summary.update(
            {
                "trained": training_result.trained,
                "training": asdict(training_result),
                "scored_count": len(scored_results),
                "message": self._message(training_result.trained, summary),
            }
        )
        return summary

    def status(self, limit: int = 500) -> Dict[str, Any]:
        """Return current model and learned-pattern summary without retraining."""
        ranking = MLRankingService(self.db)
        model_status = ranking.model_status()
        results = (
            self.db.query(Result)
            .order_by(Result.id.desc())
            .limit(max(1, min(limit, 5000)))
            .all()
        )
        training_seed_results = [result for result in results if self._is_training_seed(result)]
        live_results = [
            result
            for result in results
            if not self._is_dry_run(result) and not self._is_training_seed(result)
        ]
        dry_results = [result for result in results if self._is_dry_run(result)]
        positive_results = [result for result in live_results if self._passes_quality_bar(result)]

        summary = {
            "model": model_status,
            "trained": bool(model_status.get("trained")),
            "training": None,
            "scored_count": 0,
            "total_result_count": len(results),
            "live_result_count": len(live_results),
            "dry_result_count": len(dry_results),
            "training_seed_count": len(training_seed_results),
            "positive_result_count": len(positive_results),
            "accept_rate": round(len(positive_results) / len(live_results), 4) if live_results else 0.0,
            "check_summary": self._check_summary(live_results),
            "top_failed_checks": self._top_failed_checks(live_results),
            "best_focuses": [asdict(item) for item in self._focus_stats(live_results)[:5]],
            "best_settings": [asdict(item) for item in self._settings_stats(live_results)[:5]],
            "best_training_seed_focuses": [asdict(item) for item in self._focus_stats(training_seed_results)[:8]],
            "best_training_seed_settings": [asdict(item) for item in self._settings_stats(training_seed_results)[:8]],
        }
        summary["message"] = self._message(bool(model_status.get("trained")), summary)
        return summary

    @classmethod
    def _message(cls, trained: bool, summary: Optional[Dict[str, Any]]) -> str:
        if summary and not summary.get("live_result_count"):
            return "Need live BRAIN results before the learner can trust performance patterns."
        if summary and not summary.get("positive_result_count"):
            return "Model updated, but no live alpha has crossed the quality bar yet."
        if not trained:
            return "Need at least 5 labeled examples with both winners and losers to train."
        return "Learner updated from stored examples and result history."

    @staticmethod
    def _is_dry_run(result: Result) -> bool:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        return bool(
            raw_metrics.get("dry_run")
            or raw_metrics.get("source") == "dry_run"
            or str(result.brain_alpha_id or "").startswith("dry-run")
        )

    @staticmethod
    def _is_training_seed(result: Result) -> bool:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        return raw_metrics.get("source") == "training_seed"

    @staticmethod
    def _passes_quality_bar(result: Result) -> bool:
        if result.all_checks_passed is not True:
            if not result.human_approved or AutoLearningService._has_failed_checks(result):
                return False
        if result.human_approved and AutoLearningService._has_failed_checks(result):
            return False
        if (result.sharpe or 0.0) < 1.25:
            return False
        if (result.fitness or 0.0) < 1.0:
            return False
        if result.turnover is not None and result.turnover > 0.70:
            return False
        if result.self_correlation is not None and result.self_correlation > 0.70:
            return False
        return True

    @staticmethod
    def _has_failed_checks(result: Result) -> bool:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        is_block = raw_metrics.get("is") if isinstance(raw_metrics.get("is"), dict) else {}
        checks = is_block.get("checks") or raw_metrics.get("checks") or []
        return any(
            isinstance(check, dict) and str(check.get("result") or "").upper() == "FAIL"
            for check in checks
        )

    def _focus_stats(self, results: Sequence[Result]) -> List[LearnedFocusStat]:
        groups: Dict[str, List[Result]] = defaultdict(list)
        for result in results:
            groups[self._classify_focus(result.expression)].append(result)

        stats = [
            LearnedFocusStat(
                name=name,
                count=len(rows),
                passed=sum(1 for row in rows if self._passes_quality_bar(row)),
                pass_rate=self._pass_rate(rows),
                avg_sharpe=self._avg(row.sharpe for row in rows),
                avg_fitness=self._avg(row.fitness for row in rows),
                avg_score=self._avg(row.final_score for row in rows),
            )
            for name, rows in groups.items()
        ]
        return sorted(stats, key=lambda item: (item.pass_rate, item.count, item.avg_score or 0.0), reverse=True)

    def _settings_stats(self, results: Sequence[Result]) -> List[LearnedSettingsStat]:
        groups: Dict[Tuple[Tuple[str, Any], ...], List[Result]] = defaultdict(list)
        settings_by_key: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = {}
        for result in results:
            settings = self._settings_for_result(result)
            key = tuple(sorted(settings.items()))
            groups[key].append(result)
            settings_by_key[key] = settings

        stats = [
            LearnedSettingsStat(
                settings=settings_by_key[key],
                count=len(rows),
                passed=sum(1 for row in rows if self._passes_quality_bar(row)),
                pass_rate=self._pass_rate(rows),
                avg_sharpe=self._avg(row.sharpe for row in rows),
                avg_fitness=self._avg(row.fitness for row in rows),
                avg_score=self._avg(row.final_score for row in rows),
            )
            for key, rows in groups.items()
        ]
        return sorted(stats, key=lambda item: (item.pass_rate, item.count, item.avg_score or 0.0), reverse=True)

    @staticmethod
    def _settings_for_result(result: Result) -> Dict[str, Any]:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        settings = raw_metrics.get("settings") if isinstance(raw_metrics, dict) else None
        if not settings and result.simulation is not None:
            settings = result.simulation.settings
        settings = settings if isinstance(settings, dict) else {}
        keys = ("region", "universe", "delay", "decay", "neutralization", "truncation", "testPeriod", "language")
        return {key: settings[key] for key in keys if key in settings}

    @staticmethod
    def _classify_focus(expression: str) -> str:
        text = (expression or "").lower()
        if "implied_volatility" in text or "pcr_" in text:
            return "options"
        if "est_" in text:
            return "analyst"
        if "news_" in text or "scl" in text:
            return "sentiment"
        if "mdl" in text or any(field in text for field in ("beta", "rel_ret_all", "parkinson_volatility")):
            return "model_risk"
        if any(field in text for field in ("ebit", "capex", "cashflow", "assets", "debt", "sales", "revenue")):
            return "quality"
        if "ts_corr" in text and any(field in text for field in ("volume", "adv20", "adv5")):
            return "price_volume"
        if "ts_std_dev" in text or "ts_zscore" in text:
            return "mean_reversion"
        if "ts_rank" in text or "ts_decay_linear" in text:
            return "momentum"
        if any(field in text for field in ("open", "high", "low", "vwap")):
            return "intraday"
        return "hybrid"

    @staticmethod
    def _checks_for_result(result: Result) -> List[Dict[str, Any]]:
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        is_block = raw_metrics.get("is") if isinstance(raw_metrics.get("is"), dict) else {}
        checks = is_block.get("checks") or raw_metrics.get("checks") or []
        return [check for check in checks if isinstance(check, dict)]

    def _check_summary(self, results: Sequence[Result]) -> Dict[str, Any]:
        pass_count = 0
        fail_count = 0
        pending_count = 0
        for result in results:
            checks = self._checks_for_result(result)
            if not checks and result.all_checks_passed is not None:
                pass_count += 1 if result.all_checks_passed else 0
                fail_count += 0 if result.all_checks_passed else 1
                continue
            for check in checks:
                outcome = str(check.get("result") or "").upper()
                if outcome == "PASS":
                    pass_count += 1
                elif outcome == "FAIL":
                    fail_count += 1
                elif outcome == "PENDING":
                    pending_count += 1
        total = pass_count + fail_count + pending_count
        return {
            "pass": pass_count,
            "fail": fail_count,
            "pending": pending_count,
            "pass_rate": round(pass_count / total, 4) if total else 0.0,
        }

    def _top_failed_checks(self, results: Sequence[Result]) -> List[Dict[str, Any]]:
        counts = Counter()
        for result in results:
            for check in self._checks_for_result(result):
                if str(check.get("result") or "").upper() == "FAIL":
                    counts[str(check.get("name") or "UNKNOWN")] += 1
        return [{"name": name, "count": count} for name, count in counts.most_common(8)]

    def _pass_rate(self, rows: Sequence[Result]) -> float:
        return round(sum(1 for row in rows if self._passes_quality_bar(row)) / len(rows), 4) if rows else 0.0

    @staticmethod
    def _avg(values) -> Optional[float]:
        numeric = [float(value) for value in values if value is not None]
        return round(mean(numeric), 4) if numeric else None
