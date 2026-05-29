"""Autopilot generation loop for keeping live simulations fed."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import logging
import random
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from backend.config import settings
from backend.core.dataset_catalog import (
    DatasetProfile,
    dataset_settings_overrides,
    list_dataset_profiles,
)
from backend.core.field_intelligence import schema_with_persisted_fields, top_field_names
from backend.core.simulation_settings import merge_simulation_settings
from backend.generation.candidates import AlphaCandidate
from backend.generation.expression_generator import RuleBasedAlphaGenerator, STRATEGY_DESCRIPTIONS
from backend.generation.openai_advisor import OpenAIAlphaAdvisor
from backend.ml.auto_learner import AutoLearningService
from backend.ml.service import MLRankingService
from backend.models import Account, Simulation
from backend.orchestration.service import RUNNING_STATUSES, SimulationOrchestrator
from backend.utils.time import utc_now

logger = logging.getLogger(__name__)


@dataclass
class SpecialBatch:
    """One random generation batch selected by the special autopilot."""

    seed: int
    dataset_id: str
    focus: str
    settings: Dict[str, Any]
    generated_count: int
    queued_count: int
    expressions: List[str]
    top_predictions: List[Dict[str, Any]]
    openai_assist: bool = False
    openai_advised_count: int = 0


class SpecialAutopilot:
    """Generate, score, queue, and submit random candidates under a running cap."""

    DECAYS = (2, 4, 6, 8, 10, 12, 16, 20, 40, 60)
    TRUNCATIONS = (0.01, 0.02, 0.04, 0.08)
    NEUTRALIZATIONS = ("SUBINDUSTRY", "INDUSTRY", "SECTOR")

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)

    def tick(
        self,
        db: Session,
        orchestrator: SimulationOrchestrator,
        *,
        dry_run: bool,
        universe: str,
        batch_size: int = 5,
        target_running: int = 5,
        max_running: int = 6,
        refill_pending_below: int = 10,
        max_pending: int = 15,
        stale_running_minutes: int = 240,
        openai_assist: bool = False,
        account_ids: Optional[Sequence[int]] = None,
        poll_first: bool = True,
        submit_batch_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one autopilot control tick."""
        batch_size = max(1, min(batch_size, 20))
        target_running = max(1, min(target_running, 25))
        max_running = max(target_running, min(max_running, 30))
        refill_pending_below = max(0, min(refill_pending_below, 500))
        max_pending = max(refill_pending_below, min(max_pending, 1000))
        stale_running_minutes = max(15, min(stale_running_minutes, 24 * 60))
        selected_account_ids = self._worker_account_ids(db, account_ids)
        if not selected_account_ids:
            return {
                "poll_message": "No worker-enabled accounts available",
                "poll_errors": ["No worker-enabled accounts available"],
                "stale_reaped_count": 0,
                "running_before": 0,
                "running_after": 0,
                "pending_before": 0,
                "pending_after": 0,
                "pending_refill_below": refill_pending_below,
                "pending_max": max_pending,
                "refill_skipped": True,
                "batch": None,
                "submitted_ids": [],
                "submit_errors": ["No worker-enabled accounts available"],
                "account_ids": [],
            }

        if poll_first:
            poll_result = orchestrator.poll_running(
                db,
                limit=max(25, max_running * 4),
                account_ids=selected_account_ids,
                per_account_limit=1,
            )
            poll_message = poll_result.message
            poll_errors = poll_result.errors
        else:
            poll_message = "Skipped on submit tick; worker poll interval owns BRAIN polling"
            poll_errors = []
        stale_count = self.reap_stale_running(
            db,
            max_age_minutes=stale_running_minutes,
            account_ids=selected_account_ids,
        )
        rebalance_result = orchestrator.rebalance_pending(
            db,
            account_ids=selected_account_ids,
            require_worker_enabled=True,
        )
        lane_before = self._lane_snapshot(db, selected_account_ids)
        running_before = self._running_count(db, selected_account_ids)
        pending_before = self._pending_count(db, selected_account_ids)
        batch: Optional[SpecialBatch] = None

        pending_room = max(0, max_pending - pending_before)
        lane_needs_work = any(
            lane["running"] < min(lane["max_running"], max_running)
            and (lane["running"] + lane["pending"]) < min(lane["max_running"], max_running)
            for lane in lane_before
        )
        should_refill_pending = (
            pending_before < refill_pending_below
            and running_before < max_running
            and (running_before < target_running or lane_needs_work)
        )
        if should_refill_pending and pending_room > 0:
            batch = self.queue_random_batch(
                db,
                orchestrator,
                batch_size=min(batch_size, pending_room),
                openai_assist=openai_assist,
                account_ids=selected_account_ids,
            )

        submitted = []
        submit_errors: List[str] = []
        running_now = self._running_count(db, selected_account_ids)
        slots = max(0, max_running - running_now)
        attempts = 0
        max_submissions = slots if submit_batch_limit is None else max(1, min(slots, int(submit_batch_limit or 1)))
        max_attempts = max(max_submissions * 4, max_submissions)
        while slots > 0 and len(submitted) < max_submissions and attempts < max_attempts:
            attempts += 1
            result = orchestrator.submit_next(
                db,
                universe=universe,
                dry_run=dry_run,
                account_ids=selected_account_ids,
                require_worker_enabled=True,
            )
            if result.errors:
                waiting_errors = [
                    error
                    for error in result.errors
                    if "waiting" in str(error).lower() and "live submit" in str(error).lower()
                ]
                if len(waiting_errors) != len(result.errors):
                    submit_errors.extend(result.errors)
                break
                continue
            if not result.simulations:
                break
            submitted.extend(
                simulation.id
                for simulation in result.simulations
                if simulation.status in {"running", "completed"}
            )
            running_now = self._running_count(db, selected_account_ids)
            slots = max(0, max_running - running_now)
            if running_now >= max_running:
                break

        return {
            "poll_message": poll_message,
            "poll_errors": poll_errors,
            "poll_first": poll_first,
            "stale_reaped_count": stale_count,
            "running_before": running_before,
            "running_after": self._running_count(db, selected_account_ids),
            "pending_before": pending_before,
            "pending_after": self._pending_count(db, selected_account_ids),
            "pending_refill_below": refill_pending_below,
            "pending_max": max_pending,
            "refill_skipped": not should_refill_pending,
            "rebalance": rebalance_result.metadata,
            "lanes": self._lane_snapshot(db, selected_account_ids),
            "batch": asdict(batch) if batch else None,
            "submitted_ids": submitted,
            "submit_errors": submit_errors,
            "submit_batch_limit": max_submissions,
            "account_ids": selected_account_ids,
        }

    def queue_random_batch(
        self,
        db: Session,
        orchestrator: SimulationOrchestrator,
        *,
        batch_size: int = 5,
        openai_assist: bool = False,
        account_ids: Optional[Sequence[int]] = None,
    ) -> SpecialBatch:
        """Generate one random learner-scored batch and queue the best rows."""
        profile = self._choose_dataset(db)
        focus = self._choose_focus(db, profile)
        seed = self.random.randrange(1, 2_147_483_647)
        settings = self._learned_settings(db, profile, focus) or self._random_settings(profile, focus)
        candidate_pool = self._candidate_pool(db, profile, focus, seed, batch_size)
        ranked, openai_count = self._rank_candidates(
            db,
            candidate_pool,
            settings=settings,
            focus=focus,
            dataset_id=profile.id,
            openai_assist=openai_assist,
        )
        selected = ranked[:batch_size]
        expressions = [item["candidate"].expression for item in selected]

        queued_count = 0
        if expressions:
            result = orchestrator.enqueue_expressions(
                db,
                expressions,
                account_ids=account_ids,
                validate=True,
                settings=settings,
                require_worker_enabled=True,
            )
            queued_count = len(result.simulations) if result.ok else 0

        return SpecialBatch(
            seed=seed,
            dataset_id=profile.id,
            focus=focus,
            settings=settings,
            generated_count=len(candidate_pool),
            queued_count=queued_count,
            expressions=expressions,
            top_predictions=[
                {
                    "expression": item["candidate"].expression,
                    "pass_probability": item["pass_probability"],
                    "score": item["score"],
                    "candidate_score": item["candidate"].score,
                    "openai_score": item.get("openai_score"),
                }
                for item in selected
            ],
            openai_assist=openai_count > 0,
            openai_advised_count=openai_count,
        )

    def _candidate_pool(
        self,
        db: Session,
        profile: DatasetProfile,
        focus: str,
        seed: int,
        batch_size: int,
    ) -> List[AlphaCandidate]:
        fields = top_field_names(db, dataset_ids=[profile.id], limit=120)
        existing = [row[0] for row in db.query(Simulation.expression).all()]
        schema = schema_with_persisted_fields(db)
        generator = RuleBasedAlphaGenerator(schema=schema, seed=seed)
        candidates = generator.generate(
            count=max(batch_size * 8, 32),
            focus=focus,
            fields=fields or None,
            dataset_ids=[profile.id],
            neutralize=self.random.random() > 0.15,
            existing_expressions=existing,
        )
        if len(candidates) >= batch_size:
            return candidates

        fallback = RuleBasedAlphaGenerator(schema=schema, seed=seed)
        candidates.extend(
            fallback.generate(
                count=max(batch_size * 8, 32),
                focus=None,
                dataset_ids=["pv1"],
                neutralize=True,
                existing_expressions=existing + [candidate.expression for candidate in candidates],
            )
        )
        if len(candidates) >= batch_size:
            return candidates

        broad = RuleBasedAlphaGenerator(schema=schema, seed=seed + 1)
        candidates.extend(
            broad.generate(
                count=max(batch_size * 8, 32),
                focus=None,
                neutralize=True,
                existing_expressions=existing + [candidate.expression for candidate in candidates],
            )
        )
        return candidates

    def _rank_candidates(
        self,
        db: Session,
        candidates: Sequence[AlphaCandidate],
        *,
        settings: Dict[str, Any],
        focus: str,
        dataset_id: str,
        openai_assist: bool,
    ) -> tuple[List[Dict[str, Any]], int]:
        ranker = MLRankingService(db)
        ranked = []
        for candidate in candidates:
            prediction = ranker.score_expression(candidate.expression, metrics={"settings": settings})
            failure_penalty = self._failure_shape_penalty(candidate.expression, focus, settings)
            blended = (
                prediction.score * 0.55
                + prediction.pass_probability * 0.30
                + candidate.score * 0.15
                + self.random.random() * 0.025
                - failure_penalty
            )
            ranked.append(
                {
                    "candidate": candidate,
                    "pass_probability": round(prediction.pass_probability, 4),
                    "score": round(blended, 4),
                }
            )
        ranked = sorted(ranked, key=lambda item: item["score"], reverse=True)
        if not openai_assist:
            return ranked, 0

        try:
            advice = OpenAIAlphaAdvisor().advise(
                [item["candidate"] for item in ranked[:30]],
                settings=settings,
                focus=focus,
                dataset_id=dataset_id,
                limit=30,
            )
        except Exception:
            logger.info("OpenAI special autopilot advice unavailable", exc_info=True)
            return ranked, 0

        advice_by_expression = {item.expression: item for item in advice}
        if not advice_by_expression:
            return ranked, 0

        for item in ranked:
            candidate = item["candidate"]
            advice_item = advice_by_expression.get(candidate.expression)
            if advice_item is None:
                continue
            item["openai_score"] = round(advice_item.score, 4)
            item["score"] = round(item["score"] * 0.80 + advice_item.score * 0.20, 4)
        return sorted(ranked, key=lambda item: item["score"], reverse=True), len(advice_by_expression)

    def _choose_dataset(self, db: Session) -> DatasetProfile:
        profiles = list_dataset_profiles()
        learned = set(self._learned_focuses(db))
        weights = []
        for profile in profiles:
            preferred = set(profile.preferred_focuses or ())
            if profile.category in {"price_volume", "fundamental", "model_risk"}:
                weight = 1.35
            elif profile.category in {"options", "news_sentiment", "analyst"}:
                weight = 1.15
            else:
                weight = 1.0
            if learned and preferred.intersection(learned):
                weight *= 2.25
            if "analyst" in preferred and "analyst" in learned:
                weight *= 1.75
            weights.append(weight)
        return self.random.choices(profiles, weights=weights, k=1)[0]

    def _choose_focus(self, db: Session, profile: DatasetProfile) -> str:
        preferred = list(profile.preferred_focuses or ())
        learned = self._learned_focuses(db)
        learned_preferred = [item for item in learned if item in preferred]
        if "analyst" in learned_preferred and self.random.random() < 0.55:
            return "analyst"
        if learned_preferred and self.random.random() < 0.80:
            return self.random.choice(learned_preferred)
        if preferred:
            return self.random.choice(preferred)
        return self.random.choice(sorted(STRATEGY_DESCRIPTIONS))

    def _learned_focuses(self, db: Session) -> List[str]:
        try:
            status = AutoLearningService(db).status(limit=500)
        except Exception:
            return []
        live_rows = status.get("best_focuses") or []
        live_focuses = [
            str(row.get("name"))
            for row in live_rows
            if row.get("name") and (row.get("passed") or 0) > 0
        ]
        if live_focuses:
            return live_focuses
        seed_rows = status.get("best_training_seed_focuses") or []
        return [
            str(row.get("name"))
            for row in seed_rows
            if row.get("name") and (row.get("passed") or 0) > 0
        ]

    def _learned_settings(self, db: Session, profile: DatasetProfile, focus: str) -> Optional[Dict[str, Any]]:
        try:
            status = AutoLearningService(db).status(limit=1000)
        except Exception:
            return None

        best_live = [
            row
            for row in status.get("best_settings") or []
            if (row.get("passed") or 0) > 0 and isinstance(row.get("settings"), dict)
        ]
        if focus == "analyst" and self.random.random() < 0.75:
            settings = {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "decay": 4,
                "neutralization": "NONE",
                "truncation": 0.02,
                "testPeriod": "P5Y",
                "language": "FASTEXPR",
            }
            if profile.default_universe:
                settings["universe"] = profile.default_universe
            return merge_simulation_settings(settings)

        if best_live and self.random.random() < 0.55:
            settings = dict(best_live[0]["settings"])
            if profile.default_universe and self.random.random() < 0.35:
                settings["universe"] = profile.default_universe
            return merge_simulation_settings(settings)

        return None

    def _random_settings(self, profile: DatasetProfile, focus: str) -> Dict[str, Any]:
        settings = merge_simulation_settings(dataset_settings_overrides([profile.id]))
        settings["decay"] = self._decay_for_focus(focus)
        settings["truncation"] = self.random.choice(self.TRUNCATIONS)
        settings["neutralization"] = settings.get("neutralization") or self.random.choice(self.NEUTRALIZATIONS)
        if self.random.random() < 0.20:
            settings["neutralization"] = self.random.choice(self.NEUTRALIZATIONS)
        if profile.default_universe:
            settings["universe"] = profile.default_universe
        if profile.category in {"options", "news_sentiment"}:
            settings["maxTrade"] = "OFF"
            settings["truncation"] = max(float(settings["truncation"]), 0.04)
        return merge_simulation_settings(settings)

    @staticmethod
    def _failure_shape_penalty(expression: str, focus: str, settings: Dict[str, Any]) -> float:
        text = (expression or "").lower()
        penalty = 0.0
        if focus in {"momentum", "price_volume"} and "ts_rank" in text and "group_" not in text:
            penalty += 0.06
        if str(settings.get("neutralization") or "").upper() in {"SUBINDUSTRY", "INDUSTRY"} and text.startswith("group_neutralize("):
            penalty += 0.05
        try:
            decay = int(settings.get("decay") or 0)
        except (TypeError, ValueError):
            decay = 0
        if decay >= 20 and focus in {"momentum", "quality"}:
            penalty += 0.04
        if "ts_zscore" in text and "winsorize" not in text:
            penalty += 0.03
        return penalty

    def _decay_for_focus(self, focus: str) -> int:
        if focus in {"fundamental", "quality"}:
            return self.random.choice((10, 16, 20, 40, 60))
        if focus in {"options", "sentiment", "intraday"}:
            return self.random.choice((2, 4, 6, 8))
        if focus in {"momentum", "model_risk"}:
            return self.random.choice((8, 10, 12, 16, 20))
        return self.random.choice(self.DECAYS)

    @staticmethod
    def _running_count(db: Session, account_ids: Optional[Sequence[int]] = None) -> int:
        query = db.query(Simulation).filter(Simulation.status.in_(list(RUNNING_STATUSES)))
        if account_ids:
            query = query.filter(Simulation.account_id.in_(list(account_ids)))
        return query.count()

    @staticmethod
    def _pending_count(db: Session, account_ids: Optional[Sequence[int]] = None) -> int:
        query = db.query(Simulation).filter(Simulation.status == "pending")
        if account_ids:
            query = query.filter(Simulation.account_id.in_(list(account_ids)))
        return query.count()

    @staticmethod
    def _worker_account_ids(db: Session, account_ids: Optional[Sequence[int]] = None) -> List[int]:
        query = db.query(Account).filter(Account.is_active == True).filter(Account.worker_enabled == True)
        if bool(settings.single_account_mode):
            primary_id = settings.primary_account_id
            if primary_id:
                primary = query.filter(Account.id == int(primary_id)).first()
                return [int(primary.id)] if primary else []
            first = query.order_by(Account.id.asc()).first()
            return [int(first.id)] if first else []
        if account_ids:
            query = query.filter(Account.id.in_(list(account_ids)))
        return [int(account.id) for account in query.order_by(Account.id.asc()).all()]

    @staticmethod
    def _lane_snapshot(db: Session, account_ids: Sequence[int]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        accounts = (
            db.query(Account)
            .filter(Account.id.in_(list(account_ids)))
            .order_by(Account.id.asc())
            .all()
        )
        for account in accounts:
            running = (
                db.query(Simulation)
                .filter(Simulation.account_id == account.id)
                .filter(Simulation.status.in_(list(RUNNING_STATUSES)))
                .count()
            )
            pending = (
                db.query(Simulation)
                .filter(Simulation.account_id == account.id)
                .filter(Simulation.status == "pending")
                .count()
            )
            rows.append(
                {
                    "account_id": int(account.id),
                    "running": int(running),
                    "pending": int(pending),
                    "max_running": int(account.max_running or 1),
                    "max_pending": int(account.max_pending or 0),
                    "cooldown_until": account.cooldown_until.isoformat() if account.cooldown_until else None,
                    "last_worker_error": account.last_worker_error,
                }
            )
        return rows

    @staticmethod
    def reap_stale_running(
        db: Session,
        *,
        max_age_minutes: int = 240,
        account_ids: Optional[Sequence[int]] = None,
    ) -> int:
        """Fail running/submitting rows that are too old to keep blocking slots."""
        cutoff = utc_now() - timedelta(minutes=max_age_minutes)
        query = (
            db.query(Simulation)
            .filter(Simulation.status.in_(list(RUNNING_STATUSES)))
            .filter(Simulation.submitted_at.isnot(None))
            .filter(Simulation.submitted_at < cutoff)
        )
        if account_ids:
            query = query.filter(Simulation.account_id.in_(list(account_ids)))
        stale_rows = query.all()
        for simulation in stale_rows:
            simulation.status = "failed"
            simulation.completed_at = utc_now()
            simulation.error_message = (
                f"Stale running timeout after {max_age_minutes} minutes; "
                f"BRAIN id preserved: {simulation.brain_simulation_id}"
            )
        if stale_rows:
            db.commit()
        return len(stale_rows)
