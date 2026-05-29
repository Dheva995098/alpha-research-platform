"""Quota-aware simulation queue and polling orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.brain_api import BRAINAuthenticationError, BRAINRateLimitError, BRAINSession
from backend.config import settings
from backend.core.data_fields import BRAINDataFields, get_data_fields
from backend.core.expression_normalizer import clean_brain_error_message, normalize_brain_expression
from backend.core.simulation_settings import merge_simulation_settings
from backend.generation.dedup import ExpressionDeduplicator, expression_signature
from backend.models import Account, AlphaRegistry, Result, Simulation
from backend.orchestration.quota import (
    AccountQuota,
    quota_for_account,
    reset_daily_quota_if_needed,
)
from backend.security import decrypt_credential
from backend.utils.time import utc_now

logger = logging.getLogger(__name__)


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
RUNNING_STATUSES = {"submitting", "running"}
QUEUE_STATUSES = {"pending", "submitting", "running", "completed", "failed", "cancelled"}


@dataclass
class OrchestrationResult:
    """Structured result returned by orchestration operations."""

    action: str
    simulations: List[Simulation] = field(default_factory=list)
    message: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


class BrainGateway:
    """Small live BRAIN gateway used by the orchestrator.

    Tests can provide a fake with the same methods, so queue behavior is
    verifiable without WorldQuant credentials.
    """

    _session_cache: Dict[int, BRAINSession] = {}
    _session_lock = threading.Lock()
    _request_lock = threading.Lock()
    _last_request_at = 0.0
    _cooldown_until_monotonic = 0.0

    def submit_expression(
        self,
        account: Account,
        password: str,
        expression: str,
        universe: str = "default",
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        session = self._authenticated_session(account, password)
        try:
            self._pace_request()
            simulation_id = session.submit_expression(
                normalize_brain_expression(expression),
                universe=universe,
                settings=settings,
            )
        except BRAINRateLimitError as exc:
            self._record_rate_limit(exc)
            raise
        except BRAINAuthenticationError:
            self._drop_cached_session(account.id)
            raise
        if not simulation_id:
            raise RuntimeError("BRAIN did not return a simulation id")

        return simulation_id

    def get_status(self, account: Account, password: str, brain_simulation_id: str) -> Optional[Dict[str, Any]]:
        session = self._authenticated_session(account, password)
        try:
            self._pace_request()
            return session.get_simulation_status(brain_simulation_id)
        except BRAINRateLimitError as exc:
            self._record_rate_limit(exc)
            raise
        except BRAINAuthenticationError:
            self._drop_cached_session(account.id)
            raise

    def get_results(self, account: Account, password: str, brain_simulation_id: str) -> Optional[Dict[str, Any]]:
        session = self._authenticated_session(account, password)
        try:
            self._pace_request()
            return session.get_alpha_results(brain_simulation_id)
        except BRAINRateLimitError as exc:
            self._record_rate_limit(exc)
            raise
        except BRAINAuthenticationError:
            self._drop_cached_session(account.id)
            raise

    def _authenticated_session(self, account: Account, password: str) -> BRAINSession:
        with self._session_lock:
            cached = self._session_cache.get(account.id)
            if (
                cached is not None
                and cached.is_authenticated
                and cached.email == account.brain_email
                and cached.password == password
            ):
                return cached

        session = BRAINSession(account.brain_email, password, session_name=f"account-{account.id}")
        self._pace_request()
        if not session.authenticate():
            retry_after = session.last_retry_after
            session.close()
            if session.last_status_code == 429:
                exc = BRAINRateLimitError("BRAIN authentication is rate limited; retry polling later", retry_after)
                self._record_rate_limit(exc)
                raise exc
            raise RuntimeError("BRAIN authentication failed")

        with self._session_lock:
            old_session = self._session_cache.get(account.id)
            if old_session is not None and old_session is not session:
                old_session.close()
            self._session_cache[account.id] = session
        return session

    @classmethod
    def _drop_cached_session(cls, account_id: int) -> None:
        with cls._session_lock:
            session = cls._session_cache.pop(int(account_id), None)
        if session is not None:
            session.close()

    @classmethod
    def _pace_request(cls) -> None:
        """Serialize BRAIN traffic across accounts to avoid same-IP bursts."""
        min_interval = max(0.0, float(settings.brain_request_interval_seconds or 0.0))
        with cls._request_lock:
            now = time.monotonic()
            wait_until = max(cls._cooldown_until_monotonic, cls._last_request_at + min_interval)
            wait_seconds = max(0.0, wait_until - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.monotonic()
            cls._last_request_at = now

    @classmethod
    def _record_rate_limit(cls, exc: BRAINRateLimitError) -> None:
        retry_after = getattr(exc, "retry_after", None)
        try:
            wait_seconds = int(float(retry_after or 0))
        except (TypeError, ValueError):
            wait_seconds = 0
        max_global_pause = max(0, int(settings.brain_rate_limit_cooldown_seconds or 0))
        wait_seconds = min(max(wait_seconds, 1), max_global_pause) if max_global_pause else 0
        with cls._request_lock:
            cls._cooldown_until_monotonic = max(
                cls._cooldown_until_monotonic,
                time.monotonic() + wait_seconds,
            )

    @classmethod
    def global_cooldown_remaining(cls) -> float:
        return max(0.0, cls._cooldown_until_monotonic - time.monotonic())


class SimulationOrchestrator:
    """Manage a durable queue of pending simulations."""

    _last_live_submit_by_account: Dict[int, datetime] = {}
    _last_live_submit_at: Optional[datetime] = None

    def __init__(
        self,
        schema: Optional[BRAINDataFields] = None,
        gateway: Optional[BrainGateway] = None,
    ):
        self.schema = schema or get_data_fields()
        self.gateway = gateway or BrainGateway()

    def enqueue_expressions(
        self,
        db: Session,
        expressions: Sequence[str],
        account_ids: Optional[Sequence[int]] = None,
        validate: bool = True,
        settings: Optional[Dict[str, Any]] = None,
        require_worker_enabled: bool = False,
    ) -> OrchestrationResult:
        """Create pending Simulation rows for unique expressions."""
        errors: List[str] = []
        skipped: List[Dict[str, str]] = []
        clean_expressions = [
            normalize_brain_expression(expression)
            for expression in expressions
            if expression and expression.strip()
        ]

        if not clean_expressions:
            return OrchestrationResult(
                action="enqueue",
                message="No expressions provided",
                errors=["At least one non-empty expression is required"],
            )

        existing = [normalize_brain_expression(row[0]) for row in db.query(Simulation.expression).all()]
        dedupe_result = ExpressionDeduplicator(existing).dedupe(clean_expressions)

        accounts = self._available_accounts(
            db,
            account_ids,
            require_worker_enabled=require_worker_enabled,
        )
        if not accounts:
            return OrchestrationResult(
                action="enqueue",
                message="No active accounts with remaining quota",
                errors=["No active accounts with remaining quota are available"],
            )

        registry_signatures = {
            row[0]
            for row in db.query(AlphaRegistry.expression_signature).all()
            if row[0]
        }
        created: List[Simulation] = []
        for expression in dedupe_result.unique:
            signature = expression_signature(expression)
            if signature in registry_signatures:
                skipped.append({"expression": expression, "reason": "Duplicate alpha already exists in registry"})
                continue

            if validate:
                valid, message = self.schema.validate_expression_basic(expression)
                if not valid:
                    skipped.append({"expression": expression, "reason": message})
                    continue

            account = self._select_account(db, accounts, created)
            if account is None:
                skipped.append({"expression": expression, "reason": "No selected account has pending capacity"})
                continue

            simulation_settings = self._settings_for_expression(expression, settings)
            simulation = Simulation(
                account_id=account.id,
                expression_signature=signature,
                expression=expression,
                settings=simulation_settings,
                status="pending",
                progress=0.0,
                submitted_at=utc_now(),
            )
            db.add(simulation)
            db.flush()
            db.add(
                AlphaRegistry(
                    expression_signature=signature,
                    expression=expression,
                    settings_signature=self._settings_signature(simulation_settings),
                    first_account_id=account.id,
                    first_simulation_id=simulation.id,
                    status="queued",
                    source="queue",
                )
            )
            registry_signatures.add(signature)
            created.append(simulation)

        db.commit()
        for simulation in created:
            db.refresh(simulation)

        return OrchestrationResult(
            action="enqueue",
            simulations=created,
            message=f"Queued {len(created)} simulation(s)",
            metadata={
                "duplicate_count": len(dedupe_result.duplicates)
                + sum(1 for item in skipped if item.get("reason") == "Duplicate alpha already exists in registry"),
                "skipped": skipped,
            },
        )

    def submit_next(
        self,
        db: Session,
        universe: str = "default",
        dry_run: bool = False,
        account_ids: Optional[Sequence[int]] = None,
        require_worker_enabled: bool = False,
    ) -> OrchestrationResult:
        """Submit the oldest pending simulation whose account has capacity."""
        account_ids = self._single_account_ids(db, account_ids)
        pending_query = db.query(Simulation).filter(Simulation.status == "pending")
        if account_ids:
            pending_query = pending_query.filter(Simulation.account_id.in_(list(account_ids)))
        pending_simulations = pending_query.order_by(Simulation.id.asc()).limit(100).all()
        if not pending_simulations:
            return OrchestrationResult(action="submit_next", message="No pending simulations")

        simulation: Optional[Simulation] = None
        account: Optional[Account] = None
        skipped: List[str] = []

        for candidate in pending_simulations:
            candidate_account = db.query(Account).filter(Account.id == candidate.account_id).first()
            if candidate_account is None or not candidate_account.is_active:
                candidate.status = "failed"
                candidate.error_message = "Assigned account is missing or inactive"
                skipped.append(f"Simulation {candidate.id}: {candidate.error_message}")
                continue
            if require_worker_enabled and not bool(candidate_account.worker_enabled):
                skipped.append(f"Simulation {candidate.id}: assigned account worker lane is disabled")
                continue
            if self._account_is_cooling_down(candidate_account):
                skipped.append(f"Simulation {candidate.id}: assigned account is cooling down")
                continue

            reset_daily_quota_if_needed(candidate_account)
            quota = quota_for_account(candidate_account)
            if not quota.has_capacity:
                skipped.append(f"Simulation {candidate.id}: assigned account has no remaining quota")
                continue
            submit_wait_seconds = (
                self._live_submit_wait_seconds(db, candidate_account.id)
                if not dry_run and type(self.gateway) is BrainGateway
                else 0.0
            )
            if not dry_run and submit_wait_seconds > 0:
                skipped.append(
                    f"Simulation {candidate.id}: waiting {int(submit_wait_seconds)}s before next live submit"
                )
                continue
            running_count = self._running_count_for_account(db, candidate_account.id)
            if running_count >= max(1, int(candidate_account.max_running or 1)):
                skipped.append(
                    f"Simulation {candidate.id}: assigned account already has {running_count} running"
                )
                continue

            simulation = candidate
            account = candidate_account
            break

        if simulation is None or account is None:
            db.commit()
            return OrchestrationResult(
                action="submit_next",
                message="No pending simulations have available account quota",
                errors=skipped or ["No pending simulations have available account quota"],
                metadata={"skipped": skipped},
            )

        simulation.status = "submitting"
        simulation.expression = normalize_brain_expression(simulation.expression)
        simulation.expression_signature = expression_signature(simulation.expression)
        simulation.settings = self._settings_for_expression(simulation.expression, simulation.settings)
        simulation.error_message = None
        db.commit()

        if dry_run:
            simulation.brain_simulation_id = f"dry-run-{simulation.id}"
            simulation.status = "completed"
            simulation.progress = 100.0
            simulation.completed_at = utc_now()
            simulation.error_message = None
            self._upsert_result(db, simulation, self._dry_run_result(simulation))
            self._sync_registry_status(db, simulation)
            db.commit()
            db.refresh(simulation)
            return OrchestrationResult(
                action="submit_next",
                simulations=[simulation],
                message="Dry-run completed locally",
                metadata={"dry_run": True, "skipped": skipped},
            )

        try:
            password = decrypt_credential(account.brain_password_encrypted)
            brain_simulation_id = self.gateway.submit_expression(
                account=account,
                password=password,
                expression=simulation.expression,
                universe=universe,
                settings=self._simulation_settings(simulation.settings),
            )
            simulation.brain_simulation_id = brain_simulation_id
            simulation.status = "running"
            simulation.progress = 0.0
            simulation.submitted_at = utc_now()
            simulation.error_message = None
            account.submissions_today = (account.submissions_today or 0) + 1
            account.last_worker_error = None
            account.cooldown_until = None
            self._record_live_submit(account.id)
            self._sync_registry_status(db, simulation)
            db.commit()
            db.refresh(simulation)
            return OrchestrationResult(
                action="submit_next",
                simulations=[simulation],
                message=f"Submitted simulation {simulation.id}",
                metadata={"skipped": skipped},
            )
        except BRAINRateLimitError as exc:
            message = self._rate_limit_message(exc)
            logger.warning("Live submit for simulation %s delayed: %s", simulation.id, message)
            simulation.status = "pending"
            simulation.error_message = message
            self._mark_account_error(account, message, retry_after=exc.retry_after)
            db.commit()
            db.refresh(simulation)
            return OrchestrationResult(
                action="submit_next",
                simulations=[simulation],
                message="Submission delayed by BRAIN rate limit",
                errors=[message],
                metadata={"skipped": skipped, "transient": True, "retry_after": exc.retry_after},
            )
        except Exception as exc:
            logger.exception("Failed to submit simulation %s", simulation.id)
            simulation.status = "failed"
            simulation.error_message = clean_brain_error_message(str(exc))
            invalid_field = self._unknown_variable_from_error(simulation.error_message)
            disabled_count = self._fail_pending_with_field(db, invalid_field) if invalid_field else 0
            if invalid_field:
                account.last_worker_error = None
                account.cooldown_until = None
            else:
                self._mark_account_error(account, simulation.error_message)
            self._sync_registry_status(db, simulation)
            db.commit()
            db.refresh(simulation)
            return OrchestrationResult(
                action="submit_next",
                simulations=[simulation],
                message="Submission failed",
                errors=[simulation.error_message or str(exc)],
                metadata={"skipped": skipped, "invalid_field": invalid_field, "disabled_pending_count": disabled_count},
            )

    def submit_result_live(
        self,
        db: Session,
        result_id: int,
        universe: str = "default",
        settings: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Create and submit a live BRAIN simulation from a stored result row."""
        source_result = db.query(Result).filter(Result.id == result_id).first()
        if source_result is None:
            return OrchestrationResult(
                action="submit_result_live",
                errors=[f"Result {result_id} not found"],
            )

        account = db.query(Account).filter(Account.id == source_result.account_id).first()
        if account is None or not account.is_active:
            return OrchestrationResult(
                action="submit_result_live",
                errors=["Result account is missing or inactive"],
            )

        reset_daily_quota_if_needed(account)
        quota = quota_for_account(account)
        if not quota.has_capacity:
            return OrchestrationResult(
                action="submit_result_live",
                errors=["Assigned account has no remaining quota"],
                metadata={"result_id": result_id},
            )

        expression = normalize_brain_expression(source_result.expression)
        simulation_settings = self._settings_for_expression(
            expression,
            self._settings_from_result(source_result, overrides=settings),
        )
        simulation = Simulation(
            account_id=account.id,
            expression_signature=expression_signature(expression),
            expression=expression,
            settings=simulation_settings,
            status="submitting",
            progress=0.0,
            submitted_at=utc_now(),
        )
        db.add(simulation)
        db.commit()
        db.refresh(simulation)

        try:
            password = decrypt_credential(account.brain_password_encrypted)
            brain_simulation_id = self.gateway.submit_expression(
                account=account,
                password=password,
                expression=simulation.expression,
                universe=universe,
                settings=simulation_settings,
            )
            simulation.brain_simulation_id = brain_simulation_id
            simulation.status = "running"
            simulation.progress = 0.0
            simulation.error_message = None
            account.submissions_today = (account.submissions_today or 0) + 1
            account.last_worker_error = None
            account.cooldown_until = None
            db.commit()
            db.refresh(simulation)
            return OrchestrationResult(
                action="submit_result_live",
                simulations=[simulation],
                message=f"Submitted result {result_id} to live BRAIN simulation",
                metadata={
                    "source_result_id": result_id,
                    "dry_run": False,
                    "settings": simulation_settings,
                },
            )
        except BRAINRateLimitError as exc:
            message = self._rate_limit_message(exc)
            logger.warning("Live submit for result %s delayed: %s", result_id, message)
            simulation.status = "pending"
            simulation.error_message = message
            self._mark_account_error(account, message, retry_after=exc.retry_after)
            db.commit()
            db.refresh(simulation)
            return OrchestrationResult(
                action="submit_result_live",
                simulations=[simulation],
                message="Live submission delayed by BRAIN rate limit",
                errors=[message],
                metadata={"source_result_id": result_id, "transient": True, "retry_after": exc.retry_after},
            )
        except Exception as exc:
            logger.exception("Failed to live-submit result %s", result_id)
            simulation.status = "failed"
            simulation.error_message = clean_brain_error_message(str(exc))
            invalid_field = self._unknown_variable_from_error(simulation.error_message)
            disabled_count = self._fail_pending_with_field(db, invalid_field) if invalid_field else 0
            if invalid_field:
                account.last_worker_error = None
                account.cooldown_until = None
            else:
                self._mark_account_error(account, simulation.error_message)
            db.commit()
            db.refresh(simulation)
            return OrchestrationResult(
                action="submit_result_live",
                simulations=[simulation],
                message="Live submission failed",
                errors=[simulation.error_message or str(exc)],
                metadata={
                    "source_result_id": result_id,
                    "invalid_field": invalid_field,
                    "disabled_pending_count": disabled_count,
                },
            )

    def poll_running(
        self,
        db: Session,
        limit: int = 25,
        account_ids: Optional[Sequence[int]] = None,
        per_account_limit: Optional[int] = None,
    ) -> OrchestrationResult:
        """Poll running simulations and persist completed results."""
        account_ids = self._single_account_ids(db, account_ids)
        query = (
            db.query(Simulation)
            .filter(Simulation.status.in_(list(RUNNING_STATUSES)))
            .filter(Simulation.brain_simulation_id.isnot(None))
        )
        if account_ids:
            query = query.filter(Simulation.account_id.in_(list(account_ids)))
        if per_account_limit:
            simulations = self._poll_candidates_by_account(
                db,
                query,
                account_ids=account_ids,
                per_account_limit=per_account_limit,
                limit=limit,
            )
        else:
            simulations = query.order_by(Simulation.id.asc()).limit(limit).all()

        updated: List[Simulation] = []
        errors: List[str] = []
        rate_limited = False
        for simulation in simulations:
            account = simulation.account
            if account is None:
                simulation.status = "failed"
                simulation.error_message = "Assigned account is missing"
                updated.append(simulation)
                db.commit()
                db.refresh(simulation)
                continue
            if self._account_is_cooling_down(account):
                simulation.error_message = "Assigned account is cooling down after a BRAIN API limit/error"
                updated.append(simulation)
                db.commit()
                db.refresh(simulation)
                continue

            try:
                if simulation.brain_simulation_id.startswith("dry-run-"):
                    simulation.status = "completed"
                    simulation.progress = 100.0
                    simulation.completed_at = utc_now()
                    simulation.error_message = None
                    self._upsert_result(db, simulation, self._dry_run_result(simulation))
                    account.last_worker_error = None
                    account.cooldown_until = None
                    self._sync_registry_status(db, simulation)
                    updated.append(simulation)
                    db.commit()
                    db.refresh(simulation)
                    continue

                password = decrypt_credential(account.brain_password_encrypted)
                status_payload = self.gateway.get_status(
                    account=account,
                    password=password,
                    brain_simulation_id=simulation.brain_simulation_id,
                )
                if not status_payload:
                    continue

                self._apply_status(simulation, status_payload)
                if simulation.status == "completed":
                    result_payload = self.gateway.get_results(
                        account=account,
                        password=password,
                        brain_simulation_id=simulation.brain_simulation_id,
                    )
                    if result_payload is not None:
                        self._upsert_result(db, simulation, result_payload)
                    account.last_worker_error = None
                    account.cooldown_until = None

                self._sync_registry_status(db, simulation)
                updated.append(simulation)
            except BRAINRateLimitError as exc:
                message = self._rate_limit_message(exc)
                logger.warning("Polling simulation %s delayed: %s", simulation.id, message)
                simulation.error_message = message
                self._mark_account_error(account, message, retry_after=exc.retry_after)
                errors.append(message)
                updated.append(simulation)
                rate_limited = True
                break
            except Exception as exc:
                logger.exception("Failed to poll simulation %s", simulation.id)
                simulation.status = "failed"
                simulation.error_message = clean_brain_error_message(str(exc))
                self._mark_account_error(account, simulation.error_message)
                self._sync_registry_status(db, simulation)
                errors.append(simulation.error_message or str(exc))
                updated.append(simulation)

            db.commit()
            db.refresh(simulation)

        if not simulations:
            db.commit()

        return OrchestrationResult(
            action="poll_running",
            simulations=updated,
            message=f"Updated {len(updated)} simulation(s)",
            errors=errors,
            metadata={"transient": rate_limited, "rate_limited": rate_limited},
        )

    def clear_terminal(
        self,
        db: Session,
        statuses: Optional[Sequence[str]] = None,
    ) -> OrchestrationResult:
        """Delete failed/cancelled local queue rows."""
        statuses = list(statuses or ["failed", "cancelled"])
        simulations = db.query(Simulation).filter(Simulation.status.in_(statuses)).all()
        count = len(simulations)
        for simulation in simulations:
            db.delete(simulation)
        db.commit()
        return OrchestrationResult(
            action="clear_terminal",
            message=f"Cleared {count} simulation(s)",
            metadata={"cleared_count": count, "statuses": statuses},
        )

    def clear_pending(
        self,
        db: Session,
        keep_latest: int = 0,
    ) -> OrchestrationResult:
        """Delete local pending rows, optionally keeping the newest N."""
        keep_latest = max(0, int(keep_latest or 0))
        query = db.query(Simulation).filter(Simulation.status == "pending")
        if keep_latest:
            keep_ids = [
                row.id
                for row in query.order_by(Simulation.id.desc()).limit(keep_latest).all()
            ]
            simulations = query.filter(~Simulation.id.in_(keep_ids)).all()
        else:
            simulations = query.all()
        count = len(simulations)
        for simulation in simulations:
            db.delete(simulation)
        db.commit()
        return OrchestrationResult(
            action="clear_pending",
            message=f"Cleared {count} pending simulation(s)",
            metadata={"cleared_count": count, "keep_latest": keep_latest},
        )

    def rebalance_pending(
        self,
        db: Session,
        account_ids: Optional[Sequence[int]] = None,
        require_worker_enabled: bool = True,
        limit: int = 200,
    ) -> OrchestrationResult:
        """Move pending rows away from cooling/disabled/saturated account lanes."""
        account_ids = self._single_account_ids(db, account_ids)
        target_accounts = self._available_accounts(
            db,
            account_ids=account_ids,
            require_worker_enabled=require_worker_enabled,
        )
        if not target_accounts:
            return OrchestrationResult(
                action="rebalance_pending",
                message="No available account lanes for pending rebalance",
                metadata={"rebalanced_count": 0, "moves": []},
            )

        pending_query = db.query(Simulation).filter(Simulation.status == "pending")
        if account_ids:
            pending_query = pending_query.filter(Simulation.account_id.in_(list(account_ids)))
        pending_rows = pending_query.order_by(Simulation.id.asc()).limit(max(1, min(limit, 1000))).all()

        changed: List[Simulation] = []
        moves: List[Dict[str, Any]] = []
        running_counts = self._status_counts_by_account(db, list(RUNNING_STATUSES))
        pending_counts = self._status_counts_by_account(db, ["pending"])
        for simulation in pending_rows:
            current_account = db.query(Account).filter(Account.id == simulation.account_id).first()
            current_is_healthy = self._pending_assignment_is_healthy(db, current_account, require_worker_enabled)
            current_lane_pressure = float("inf")
            if current_account is not None:
                current_capacity = max(1, int(current_account.max_running or 1))
                current_lane_pressure = (
                    running_counts.get(current_account.id, 0)
                    + pending_counts.get(current_account.id, 0)
                ) / current_capacity

            target_account = self._select_account(db, target_accounts, changed)
            target_lane_pressure = float("inf")
            if target_account is not None:
                target_capacity = max(1, int(target_account.max_running or 1))
                local_reassignments = sum(1 for item in changed if item.account_id == target_account.id)
                target_lane_pressure = (
                    running_counts.get(target_account.id, 0)
                    + pending_counts.get(target_account.id, 0)
                    + local_reassignments
                ) / target_capacity

            if (
                current_is_healthy
                and target_account is not None
                and target_account.id == simulation.account_id
            ):
                continue
            if current_is_healthy and target_lane_pressure >= current_lane_pressure:
                continue
            if target_account is None or target_account.id == simulation.account_id:
                continue

            old_account_id = simulation.account_id
            simulation.account_id = target_account.id
            simulation.error_message = None
            changed.append(simulation)
            moves.append(
                {
                    "simulation_id": simulation.id,
                    "from_account_id": old_account_id,
                    "to_account_id": target_account.id,
                }
            )

        if changed:
            db.commit()
            for simulation in changed:
                db.refresh(simulation)

        return OrchestrationResult(
            action="rebalance_pending",
            simulations=changed,
            message=f"Rebalanced {len(changed)} pending simulation(s)",
            metadata={"rebalanced_count": len(changed), "moves": moves},
        )

    def cancel_pending(self, db: Session, simulation_id: int) -> OrchestrationResult:
        """Cancel a pending simulation before it is submitted."""
        simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
        if simulation is None:
            return OrchestrationResult(
                action="cancel_pending",
                errors=[f"Simulation {simulation_id} not found"],
            )

        if simulation.status != "pending":
            return OrchestrationResult(
                action="cancel_pending",
                simulations=[simulation],
                errors=["Only pending simulations can be cancelled"],
            )

        simulation.status = "cancelled"
        simulation.completed_at = utc_now()
        self._sync_registry_status(db, simulation)
        db.commit()
        db.refresh(simulation)
        return OrchestrationResult(
            action="cancel_pending",
            simulations=[simulation],
            message=f"Cancelled simulation {simulation_id}",
        )

    def queue_summary(self, db: Session) -> Dict[str, Any]:
        """Return status counts and account quota summaries."""
        active_account_ids = self._single_account_ids(db)
        statuses = {}
        for status in sorted(QUEUE_STATUSES):
            status_query = db.query(Simulation).filter(Simulation.status == status)
            if active_account_ids:
                status_query = status_query.filter(Simulation.account_id.in_(active_account_ids))
            else:
                status_query = status_query.filter(False)
            statuses[status] = status_query.count()

        accounts_query = db.query(Account).filter(Account.is_active == True)
        if active_account_ids:
            accounts_query = accounts_query.filter(Account.id.in_(active_account_ids))
        else:
            accounts_query = accounts_query.filter(False)
        accounts = accounts_query.order_by(Account.id.asc()).all()
        for account in accounts:
            reset_daily_quota_if_needed(account)
            if account.cooldown_until and account.cooldown_until <= utc_now():
                account.cooldown_until = None
                if account.last_worker_error and "rate limit" in account.last_worker_error.lower():
                    account.last_worker_error = None
        db.commit()

        pending_counts = self._status_counts_by_account(db, ["pending"])
        running_counts = self._status_counts_by_account(db, list(RUNNING_STATUSES))
        quotas = [quota_for_account(account) for account in accounts]
        return {
            "statuses": statuses,
            "accounts": [
                {
                    "account_id": quota.account_id,
                    "daily_quota": quota.daily_quota,
                    "submissions_today": quota.submissions_today,
                    "remaining": quota.remaining,
                    "is_active": quota.is_active,
                    "worker_enabled": bool(account.worker_enabled),
                    "max_running": int(account.max_running or 1),
                    "max_pending": int(account.max_pending or 0),
                    "cooldown_until": account.cooldown_until.isoformat()
                    if account.cooldown_until
                    else None,
                    "last_worker_error": account.last_worker_error,
                    "pending": pending_counts.get(quota.account_id, 0),
                    "running": running_counts.get(quota.account_id, 0),
                }
                for account, quota in zip(accounts, quotas)
            ],
        }

    def list_simulations(
        self,
        db: Session,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Simulation]:
        """List queued simulations, optionally filtered by status."""
        query = db.query(Simulation)
        if status:
            query = query.filter(Simulation.status == status)
        return query.order_by(Simulation.id.desc()).limit(limit).all()

    def _available_accounts(
        self,
        db: Session,
        account_ids: Optional[Sequence[int]] = None,
        require_worker_enabled: bool = False,
    ) -> List[Account]:
        account_ids = self._single_account_ids(db, account_ids)
        query = db.query(Account).filter(Account.is_active == True)
        if account_ids:
            query = query.filter(Account.id.in_(list(account_ids)))
        if require_worker_enabled:
            query = query.filter(Account.worker_enabled == True)

        accounts = query.order_by(Account.id.asc()).all()
        for account in accounts:
            reset_daily_quota_if_needed(account)
        db.commit()

        return [
            account
            for account in accounts
            if quota_for_account(account).has_capacity and not self._account_is_cooling_down(account)
        ]

    @staticmethod
    def _single_account_ids(db: Session, account_ids: Optional[Sequence[int]] = None) -> Optional[List[int]]:
        """Force old single-account behavior even if callers pass multiple lanes."""
        if not bool(settings.single_account_mode):
            return list(account_ids) if account_ids else None

        primary_id = settings.primary_account_id
        query = db.query(Account.id).filter(Account.is_active == True)
        if primary_id:
            primary = query.filter(Account.id == int(primary_id)).first()
            if primary:
                return [int(primary[0])]

        first = query.order_by(Account.id.asc()).first()
        return [int(first[0])] if first else []

    def _select_account(
        self,
        db: Session,
        accounts: Sequence[Account],
        created: Sequence[Simulation],
    ) -> Optional[Account]:
        assigned_counts: Dict[int, int] = {}
        for simulation in created:
            assigned_counts[simulation.account_id] = assigned_counts.get(simulation.account_id, 0) + 1

        pending_counts = self._status_counts_by_account(db, ["pending"])

        candidates: List[Account] = []
        for account in accounts:
            quota = quota_for_account(account)
            local_assigned = assigned_counts.get(account.id, 0)
            if bool(settings.single_account_mode):
                pending_room = max(0, quota.remaining - local_assigned)
            else:
                pending_room = max(0, int(account.max_pending or 0) - pending_counts.get(account.id, 0) - local_assigned)
            remaining = quota.remaining - local_assigned
            if pending_room > 0 and remaining > 0:
                candidates.append(account)

        if not candidates:
            return None

        def capacity_after_local_assignments(account: Account) -> tuple[int, int, int, int]:
            quota = quota_for_account(account)
            local_assigned = assigned_counts.get(account.id, 0)
            if bool(settings.single_account_mode):
                pending_room = quota.remaining - local_assigned
            else:
                pending_room = int(account.max_pending or 0) - pending_counts.get(account.id, 0) - local_assigned
            remaining = quota.remaining - local_assigned
            running_room = max(0, int(account.max_running or 1) - self._running_count_for_account(db, account.id))
            return (running_room, pending_room, remaining, -account.id)

        selected = max(candidates, key=capacity_after_local_assignments)
        return selected

    @staticmethod
    def _status_counts_by_account(db: Session, statuses: Sequence[str]) -> Dict[int, int]:
        rows = (
            db.query(Simulation.account_id, func.count(Simulation.id))
            .filter(Simulation.status.in_(list(statuses)))
            .group_by(Simulation.account_id)
            .all()
        )
        return {int(account_id): int(count) for account_id, count in rows if account_id is not None}

    @staticmethod
    def _running_count_for_account(db: Session, account_id: int) -> int:
        return (
            db.query(Simulation)
            .filter(Simulation.account_id == account_id)
            .filter(Simulation.status.in_(list(RUNNING_STATUSES)))
            .count()
        )

    @staticmethod
    def _poll_candidates_by_account(
        db: Session,
        base_query,
        *,
        account_ids: Optional[Sequence[int]],
        per_account_limit: int,
        limit: int,
    ) -> List[Simulation]:
        per_account_limit = max(1, int(per_account_limit or 1))
        account_id_rows: Sequence[int]
        if account_ids:
            account_id_rows = [int(account_id) for account_id in account_ids]
        else:
            account_id_rows = [
                int(row[0])
                for row in db.query(Simulation.account_id)
                .filter(Simulation.status.in_(list(RUNNING_STATUSES)))
                .filter(Simulation.brain_simulation_id.isnot(None))
                .filter(Simulation.account_id.isnot(None))
                .distinct()
                .order_by(Simulation.account_id.asc())
                .all()
            ]

        rows: List[Simulation] = []
        seen_ids: set[int] = set()
        for account_id in account_id_rows:
            account_rows = (
                base_query.filter(Simulation.account_id == account_id)
                .order_by(Simulation.id.asc())
                .limit(per_account_limit)
                .all()
            )
            for simulation in account_rows:
                if simulation.id in seen_ids:
                    continue
                rows.append(simulation)
                seen_ids.add(simulation.id)
                if len(rows) >= limit:
                    return rows
        return rows

    def _pending_assignment_is_healthy(
        self,
        db: Session,
        account: Optional[Account],
        require_worker_enabled: bool,
    ) -> bool:
        if account is None or not account.is_active:
            return False
        if require_worker_enabled and not bool(account.worker_enabled):
            return False
        if self._account_is_cooling_down(account):
            return False
        reset_daily_quota_if_needed(account)
        if not quota_for_account(account).has_capacity:
            return False
        running_count = self._running_count_for_account(db, account.id)
        return running_count < max(1, int(account.max_running or 1))

    @staticmethod
    def _account_is_cooling_down(account: Account) -> bool:
        return bool(account.cooldown_until and account.cooldown_until > utc_now())

    @staticmethod
    def _mark_account_error(
        account: Account,
        message: Optional[str],
        retry_after: Optional[float] = None,
    ) -> None:
        clean_message = clean_brain_error_message(message or "Worker account error")
        account.last_worker_error = clean_message
        try:
            retry_seconds = int(float(retry_after or 0))
        except (TypeError, ValueError):
            retry_seconds = 0
        lower_message = clean_message.lower()
        if retry_seconds > 0:
            account.cooldown_until = utc_now() + timedelta(seconds=retry_seconds)
        elif any(marker in lower_message for marker in ("rate limit", "authentication", "login", "429")):
            account.cooldown_until = utc_now() + timedelta(
                seconds=max(60, int(settings.brain_rate_limit_cooldown_seconds or 120))
            )
        else:
            account.cooldown_until = utc_now() + timedelta(minutes=3)

    @staticmethod
    def _unknown_variable_from_error(message: Optional[str]) -> Optional[str]:
        match = re.search(r'unknown variable\s+"([^"]+)"', str(message or ""), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
        return None

    @staticmethod
    def _fail_pending_with_field(db: Session, field: Optional[str]) -> int:
        normalized = str(field or "").strip().lower()
        if not normalized:
            return 0
        pattern = re.compile(rf"\b{re.escape(normalized)}\b", flags=re.IGNORECASE)
        disabled = 0
        rows = db.query(Simulation).filter(Simulation.status == "pending").all()
        for row in rows:
            if not pattern.search(row.expression or ""):
                continue
            row.status = "failed"
            row.error_message = f'Live BRAIN rejected unknown variable "{normalized}"; field disabled for future generation'
            row.completed_at = utc_now()
            disabled += 1
        return disabled

    @classmethod
    def _record_live_submit(cls, account_id: int) -> None:
        now = utc_now()
        cls._last_live_submit_by_account[int(account_id)] = now
        cls._last_live_submit_at = now

    @classmethod
    def _live_submit_wait_seconds(cls, db: Session, account_id: int) -> float:
        now = utc_now()
        account_min_interval = max(0.0, float(settings.brain_submit_interval_seconds or 0.0))
        global_min_interval = max(0.0, float(settings.brain_submit_global_interval_seconds or 0.0))

        account_last = cls._last_live_submit_by_account.get(int(account_id))
        if account_last is None and account_min_interval > 0:
            account_last = (
                db.query(func.max(Simulation.submitted_at))
                .filter(Simulation.account_id == int(account_id))
                .filter(Simulation.brain_simulation_id.isnot(None))
                .scalar()
            )

        global_last = cls._last_live_submit_at
        if global_last is None and global_min_interval > 0:
            global_last = (
                db.query(func.max(Simulation.submitted_at))
                .filter(Simulation.brain_simulation_id.isnot(None))
                .scalar()
            )

        waits = []
        if account_last is not None and account_min_interval > 0:
            waits.append(account_min_interval - (now - account_last).total_seconds())
        if global_last is not None and global_min_interval > 0:
            waits.append(global_min_interval - (now - global_last).total_seconds())
        return max(0.0, *waits) if waits else 0.0

    @staticmethod
    def _sync_registry_status(db: Session, simulation: Simulation) -> None:
        signature = simulation.expression_signature or expression_signature(simulation.expression or "")
        if not signature:
            return
        registry = db.query(AlphaRegistry).filter(AlphaRegistry.expression_signature == signature).first()
        if registry is None:
            registry = AlphaRegistry(
                expression_signature=signature,
                expression=normalize_brain_expression(simulation.expression or ""),
                first_account_id=simulation.account_id,
                first_simulation_id=simulation.id,
                source="queue",
            )
            db.add(registry)
        registry.status = simulation.status
        registry.updated_at = utc_now()

    @staticmethod
    def _settings_signature(settings: Optional[Dict[str, Any]]) -> str:
        normalized = json.dumps(SimulationOrchestrator._simulation_settings(settings), sort_keys=True)
        return expression_signature(normalized)

    def _apply_status(self, simulation: Simulation, payload: Dict[str, Any]) -> None:
        status = str(payload.get("status") or "").lower()
        progress = payload.get("progress")
        if progress is not None:
            simulation.progress = float(progress)

        if status in {"completed", "complete", "done"} or payload.get("alpha") or simulation.progress >= 100:
            simulation.status = "completed"
            simulation.progress = 100.0
            simulation.completed_at = utc_now()
            simulation.error_message = None
        elif status in {"failed", "error", "cancelled"}:
            simulation.status = "failed"
            simulation.completed_at = utc_now()
            simulation.error_message = clean_brain_error_message(
                str(payload.get("error") or payload.get("message") or "Simulation failed")
            )
        else:
            simulation.status = "running"
            retry_after = payload.get("retry_after")
            simulation.error_message = (
                f"BRAIN is still processing; retry after about {int(float(retry_after))}s"
                if retry_after
                else None
            )

    def _upsert_result(self, db: Session, simulation: Simulation, payload: Dict[str, Any]) -> Result:
        payload = dict(payload)
        simulation.expression_signature = simulation.expression_signature or expression_signature(simulation.expression or "")
        payload.setdefault("settings", self._simulation_settings(simulation.settings))
        payload.setdefault("source", "dry_run" if payload.get("dry_run") else "live")
        requested_alpha_id = self._result_alpha_id(payload, simulation)

        result = db.query(Result).filter(Result.simulation_id == simulation.id).first()
        if result is None:
            result = Result(
                account_id=simulation.account_id,
                simulation_id=simulation.id,
                expression=normalize_brain_expression(simulation.expression),
            )
            db.add(result)

        result.brain_alpha_id = self._unique_result_alpha_id(db, result, requested_alpha_id, simulation, payload)
        result.expression = normalize_brain_expression(simulation.expression)
        result.sharpe = self._as_float(self._payload_value(payload, "sharpe"))
        result.fitness = self._as_float(self._payload_value(payload, "fitness"))
        result.turnover = self._as_float(self._payload_value(payload, "turnover"))
        result.self_correlation = self._as_float(
            self._payload_value(payload, "self_correlation", "selfCorrelation")
        )
        result.all_checks_passed = self._as_bool(
            self._payload_value(payload, "all_checks_passed", "checksPassed", "checks")
        )
        result.raw_metrics = payload
        result.final_score = self._score_result(result)
        try:
            from backend.ml.service import MLRankingService

            ranking = MLRankingService(db)
            prediction = ranking.score_expression(result.expression, metrics=ranking.metrics_from_result(result))
            result.ml_pass_probability = prediction.pass_probability
            heuristic_score = result.final_score or 0.0
            result.final_score = round(heuristic_score * 0.50 + prediction.score * 0.50, 4)
        except Exception:
            logger.exception("Failed to apply ML score for simulation %s", simulation.id)
        return result

    @staticmethod
    def _result_alpha_id(payload: Dict[str, Any], simulation: Simulation) -> str:
        return str(
            payload.get("alpha_id")
            or payload.get("alphaId")
            or payload.get("alpha")
            or payload.get("id")
            or simulation.brain_simulation_id
        )

    @staticmethod
    def _unique_result_alpha_id(
        db: Session,
        result: Result,
        requested_alpha_id: str,
        simulation: Simulation,
        payload: Dict[str, Any],
    ) -> str:
        """Keep Result rows one-per-simulation even when BRAIN reuses an alpha id."""
        candidate = requested_alpha_id
        suffix = 0
        while True:
            pending_collision = any(
                isinstance(item, Result)
                and item is not result
                and getattr(item, "brain_alpha_id", None) == candidate
                for item in list(db.new) + list(db.dirty)
            )
            existing = None if pending_collision else db.query(Result).filter(Result.brain_alpha_id == candidate).first()
            if existing is None or (result.id is not None and existing.id == result.id):
                if not pending_collision:
                    if candidate != requested_alpha_id:
                        payload["brain_alpha_id_original"] = requested_alpha_id
                        payload["brain_alpha_id_deduped"] = candidate
                    return candidate
            if candidate != requested_alpha_id:
                payload["brain_alpha_id_original"] = requested_alpha_id
                payload["brain_alpha_id_deduped"] = candidate
                return candidate
            suffix += 1
            candidate = f"{requested_alpha_id}#sim-{simulation.id}" if suffix == 1 else f"{requested_alpha_id}#sim-{simulation.id}-{suffix}"

    @staticmethod
    def _dry_run_result(simulation: Simulation) -> Dict[str, Any]:
        """Generate deterministic local metrics for dry-run workflow checks."""
        expression_score = (sum(ord(char) for char in simulation.expression) % 100) / 100
        return {
            "alpha_id": simulation.brain_simulation_id,
            "sharpe": round(0.5 + expression_score, 3),
            "fitness": round(0.4 + expression_score * 0.8, 3),
            "turnover": round(0.25 + expression_score * 0.35, 3),
            "self_correlation": round(0.15 + expression_score * 0.30, 3),
            "all_checks_passed": expression_score >= 0.35,
            "dry_run": True,
            "source": "dry_run",
            "status": "DRY_RUN_LOCAL_ONLY",
            "grade": "LOCAL",
            "settings": SimulationOrchestrator._simulation_settings(simulation.settings),
        }

    @staticmethod
    def _settings_from_result(
        result: Result,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        settings = BRAINSession.DEFAULT_SIMULATION_SETTINGS.copy()
        raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
        raw_settings = raw_metrics.get("settings") if isinstance(raw_metrics, dict) else None
        if isinstance(raw_settings, dict):
            settings.update(raw_settings)
        if overrides:
            settings.update(overrides)
        return SimulationOrchestrator._simulation_settings(settings)

    @staticmethod
    def _simulation_settings(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return merge_simulation_settings(settings)

    @staticmethod
    def _settings_for_expression(
        expression: str,
        base_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        settings = merge_simulation_settings(base_settings)
        normalized = (expression or "").strip().lower()
        if normalized.startswith("group_neutralize("):
            settings["neutralization"] = "NONE"
        if "trade_when(" in normalized:
            settings["decay"] = min(int(settings.get("decay") or 0), 4)
        return settings

    @staticmethod
    def _payload_value(payload: Dict[str, Any], *keys: str) -> Any:
        """Read metric values from top-level or common BRAIN nested metric blocks."""
        containers = [payload]
        for nested_key in ("is", "metrics", "summary"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                containers.append(nested)

        checks = payload.get("checks")
        if isinstance(checks, dict):
            containers.append(checks)

        for container in containers:
            for key in keys:
                if key in container:
                    value = container[key]
                    if key == "checks" and isinstance(value, list):
                        return all(item.get("result") == "PASS" for item in value if isinstance(item, dict))
                    return value
        return None

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "pass", "passed"}
        return bool(value)

    @staticmethod
    def _score_result(result: Result) -> Optional[float]:
        if result.sharpe is None and result.fitness is None:
            return None

        sharpe = result.sharpe or 0.0
        fitness = result.fitness or 0.0
        turnover_penalty = 0.0
        if result.turnover is not None and result.turnover > 0.70:
            turnover_penalty = min(result.turnover - 0.70, 1.0)

        check_bonus = 0.10 if result.all_checks_passed else 0.0
        return round(sharpe * 0.45 + fitness * 0.45 + check_bonus - turnover_penalty, 4)

    @staticmethod
    def _rate_limit_message(exc: BRAINRateLimitError) -> str:
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            return f"{clean_brain_error_message(str(exc))} (retry after about {int(retry_after)}s)"
        return clean_brain_error_message(str(exc))
