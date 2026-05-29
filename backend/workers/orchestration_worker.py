"""In-process background worker for submission orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import time
from typing import List, Optional, Sequence

from backend.automation.special_runner import SpecialAutopilot
from backend.config import settings
from backend.ml.auto_learner import AutoLearningService
from backend.models import Account, SessionLocal, Simulation
from backend.orchestration.service import BrainGateway, RUNNING_STATUSES, SimulationOrchestrator
from backend.utils.time import utc_now

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """Current background worker state."""

    running: bool = False
    dry_run: bool = True
    universe: str = "default"
    submit_interval_seconds: int = 30
    poll_interval_seconds: int = 20
    auto_learn: bool = False
    learning_interval_seconds: int = 300
    special_auto: bool = False
    special_batch_size: int = 5
    special_target_running: int = 5
    special_max_running: int = 6
    special_refill_pending_below: int = 10
    special_max_pending: int = 15
    special_stale_running_minutes: int = 240
    openai_assist: bool = False
    account_ids: Optional[List[int]] = None
    iterations: int = 0
    special_runs: int = 0
    special_queued: int = 0
    started_at: Optional[datetime] = None
    last_tick_at: Optional[datetime] = None
    last_learning_at: Optional[datetime] = None
    last_learning_message: Optional[str] = None
    last_special_at: Optional[datetime] = None
    last_special_message: Optional[str] = None
    last_error: Optional[str] = None


class OrchestrationWorker:
    """Simple daemon thread that submits pending work and polls running work."""

    def __init__(self, orchestrator: Optional[SimulationOrchestrator] = None):
        self.orchestrator = orchestrator or SimulationOrchestrator()
        self.autopilot = SpecialAutopilot()
        self.state = WorkerState()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._poll_due_by_account: dict[int, float] = {}
        self._last_poll_lane_at = 0.0

    def start(
        self,
        submit_interval_seconds: int = 30,
        poll_interval_seconds: int = 20,
        universe: str = "default",
        dry_run: bool = True,
        auto_learn: bool = False,
        learning_interval_seconds: int = 300,
        special_auto: bool = False,
        special_batch_size: int = 5,
        special_target_running: int = 5,
        special_max_running: int = 6,
        special_refill_pending_below: int = 10,
        special_max_pending: int = 15,
        special_stale_running_minutes: int = 240,
        openai_assist: bool = False,
        account_ids: Optional[Sequence[int]] = None,
    ) -> WorkerState:
        """Start the worker if it is not already running."""
        with self._lock:
            if self.state.running:
                return self.state

            self._stop_event.clear()
            self._poll_due_by_account = {}
            self._last_poll_lane_at = 0.0
            selected_account_ids = self._sanitize_account_ids(account_ids)
            effective_submit_interval = max(1, int(submit_interval_seconds))
            if not dry_run:
                live_min_interval = float(settings.brain_submit_interval_seconds or 0)
                effective_submit_interval = max(
                    effective_submit_interval,
                    int(live_min_interval),
                )
            self.state = WorkerState(
                running=True,
                dry_run=dry_run,
                universe=universe,
                submit_interval_seconds=effective_submit_interval,
                poll_interval_seconds=max(1, poll_interval_seconds),
                auto_learn=auto_learn,
                learning_interval_seconds=max(60, learning_interval_seconds),
                special_auto=special_auto,
                special_batch_size=max(1, min(special_batch_size, 20)),
                special_target_running=max(1, min(special_target_running, 25)),
                special_max_running=max(special_target_running, min(special_max_running, 30)),
                special_refill_pending_below=max(0, min(special_refill_pending_below, 500)),
                special_max_pending=max(0, min(special_max_pending, 1000)),
                special_stale_running_minutes=max(15, min(special_stale_running_minutes, 24 * 60)),
                openai_assist=openai_assist,
                account_ids=selected_account_ids,
                started_at=utc_now(),
            )
            self._thread = threading.Thread(target=self._run, name="orchestration-worker", daemon=True)
            self._thread.start()
            return self.state

    def stop(self, timeout_seconds: int = 5) -> WorkerState:
        """Stop the worker and wait briefly for the thread to exit."""
        with self._lock:
            self._stop_event.set()
            thread = self._thread

        if thread and thread.is_alive():
            thread.join(timeout=timeout_seconds)

        with self._lock:
            self.state.running = False
            self._thread = None
            return self.state

    def snapshot(self) -> WorkerState:
        """Return a snapshot of the current state."""
        with self._lock:
            return WorkerState(**self.state.__dict__)

    @staticmethod
    def _sanitize_account_ids(account_ids: Optional[Sequence[int]]) -> Optional[List[int]]:
        if bool(settings.single_account_mode):
            return [int(settings.primary_account_id or 1)]
        if not account_ids:
            return None
        selected = sorted({int(account_id) for account_id in account_ids if int(account_id) > 0})
        return selected or None

    def _run(self) -> None:
        next_submit_at = 0.0
        next_learning_at = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            db = SessionLocal()
            try:
                global_cooldown = BrainGateway.global_cooldown_remaining()
                if global_cooldown > 0:
                    with self._lock:
                        self.state.last_error = f"BRAIN global cooldown active for about {int(global_cooldown)}s"
                    self._stop_event.wait(timeout=min(5, max(1, global_cooldown)))
                    continue

                if self.state.special_auto:
                    poll_account_ids = self._poll_accounts_due(db, now)
                    if poll_account_ids:
                        self.orchestrator.poll_running(
                            db,
                            limit=max(1, len(poll_account_ids) * 3),
                            account_ids=poll_account_ids,
                            per_account_limit=3,
                        )
                        self.autopilot.reap_stale_running(
                            db,
                            max_age_minutes=self.state.special_stale_running_minutes,
                            account_ids=self.state.account_ids,
                        )
                        self._mark_polled_accounts(db, poll_account_ids, time.monotonic())

                    if now >= next_submit_at:
                        metadata = self.autopilot.tick(
                            db,
                            self.orchestrator,
                            dry_run=self.state.dry_run,
                            universe=self.state.universe,
                            batch_size=self.state.special_batch_size,
                            target_running=self.state.special_target_running,
                            max_running=self.state.special_max_running,
                            refill_pending_below=self.state.special_refill_pending_below,
                            max_pending=self.state.special_max_pending,
                            stale_running_minutes=self.state.special_stale_running_minutes,
                            openai_assist=self.state.openai_assist,
                            account_ids=self.state.account_ids,
                            poll_first=False,
                            submit_batch_limit=self._submit_batch_limit(),
                        )
                        batch = metadata.get("batch") or {}
                        queued = int(batch.get("queued_count") or 0)
                        submitted = len(metadata.get("submitted_ids") or [])
                        with self._lock:
                            self.state.special_runs += 1
                            self.state.special_queued += queued
                            self.state.last_special_at = utc_now()
                            rebalanced = int((metadata.get("rebalance") or {}).get("rebalanced_count") or 0)
                            errors = len(metadata.get("submit_errors") or [])
                            self.state.last_special_message = (
                                f"queued {queued}, submitted {submitted}, "
                                f"rebalanced {rebalanced}, "
                                f"pending {metadata.get('pending_after')}, "
                                f"running {metadata.get('running_after')}, "
                                f"errors {errors}"
                        )
                        next_submit_at = now + self.state.submit_interval_seconds
                else:
                    if now >= next_submit_at:
                        self._submit_until_full(db)
                        next_submit_at = now + self.state.submit_interval_seconds

                    poll_account_ids = self._poll_accounts_due(db, now)
                    if poll_account_ids:
                        self.orchestrator.poll_running(
                            db,
                            limit=max(1, len(poll_account_ids) * 3),
                            account_ids=poll_account_ids,
                            per_account_limit=3,
                        )
                        self._mark_polled_accounts(db, poll_account_ids, time.monotonic())

                if self.state.auto_learn and now >= next_learning_at:
                    learning = AutoLearningService(db).run_once(limit=500)
                    with self._lock:
                        self.state.last_learning_at = utc_now()
                        self.state.last_learning_message = learning.get("message")
                    next_learning_at = now + self.state.learning_interval_seconds

                with self._lock:
                    self.state.iterations += 1
                    self.state.last_tick_at = utc_now()
                    self.state.last_error = None
            except Exception as exc:
                logger.exception("Orchestration worker tick failed")
                with self._lock:
                    self.state.last_error = str(exc)
            finally:
                db.close()

            self._stop_event.wait(timeout=1)

        with self._lock:
            self.state.running = False

    def _submit_batch_limit(self) -> int:
        if bool(settings.single_account_mode):
            return 1
        selected_count = len(self.state.account_ids or [])
        if selected_count <= 0:
            return 1
        return max(1, min(self.state.special_batch_size, selected_count))

    def _submit_until_full(self, db) -> None:
        max_submissions = self._submit_batch_limit()
        attempts = 0
        submitted_count = 0
        while attempts < max(max_submissions * 3, max_submissions):
            attempts += 1
            result = self.orchestrator.submit_next(
                db,
                universe=self.state.universe,
                dry_run=self.state.dry_run,
                account_ids=self.state.account_ids,
                require_worker_enabled=True,
            )
            if result.errors:
                if result.metadata.get("transient"):
                    break
                if not result.simulations:
                    break
                continue
            if not result.simulations:
                break
            submitted_count += len(
                [
                    simulation
                    for simulation in result.simulations
                    if simulation.status in {"running", "completed"}
                ]
            )
            if submitted_count >= max_submissions:
                break

    def _poll_accounts_due(self, db, now: float) -> List[int]:
        account_ids = self._running_poll_account_ids(db)
        if not account_ids:
            self._poll_due_by_account.clear()
            return []

        gap = self._poll_account_gap_seconds()
        if self._last_poll_lane_at and now < self._last_poll_lane_at + gap:
            return []

        account_id_set = set(account_ids)
        for account_id in list(self._poll_due_by_account):
            if account_id not in account_id_set:
                self._poll_due_by_account.pop(account_id, None)

        latest_due = max(self._poll_due_by_account.values(), default=now - gap)
        for index, account_id in enumerate(account_ids):
            if account_id not in self._poll_due_by_account:
                if not self._poll_due_by_account and index == 0:
                    self._poll_due_by_account[account_id] = now
                else:
                    latest_due = max(latest_due, now) + gap
                    self._poll_due_by_account[account_id] = latest_due

        due_accounts = [
            account_id
            for account_id in account_ids
            if self._poll_due_by_account.get(account_id, float("inf")) <= now
        ]
        if not due_accounts:
            return []
        return sorted(due_accounts, key=lambda account_id: (self._poll_due_by_account.get(account_id, now), account_id))

    def _poll_account_due(self, db, now: float) -> Optional[int]:
        due_accounts = self._poll_accounts_due(db, now)
        return due_accounts[0] if due_accounts else None

    def _mark_polled_accounts(self, db, account_ids: Sequence[int], now: float) -> None:
        self._last_poll_lane_at = now
        gap = self._poll_account_gap_seconds()
        cycle_seconds = max(float(self.state.poll_interval_seconds), gap)
        for account_id in account_ids:
            self._poll_due_by_account[int(account_id)] = now + cycle_seconds

    def _mark_polled_account(self, db, account_id: int, now: float) -> None:
        self._mark_polled_accounts(db, [account_id], now)

    def _running_poll_account_ids(self, db) -> List[int]:
        query = (
            db.query(Simulation.account_id)
            .join(Account, Account.id == Simulation.account_id)
            .filter(Simulation.status.in_(list(RUNNING_STATUSES)))
            .filter(Simulation.brain_simulation_id.isnot(None))
            .filter(Account.is_active == True)
            .filter(Account.worker_enabled == True)
        )
        if self.state.account_ids:
            query = query.filter(Simulation.account_id.in_(list(self.state.account_ids)))
        rows = query.group_by(Simulation.account_id).order_by(Simulation.account_id.asc()).all()
        return [int(account_id) for (account_id,) in rows if account_id is not None]

    @staticmethod
    def _poll_account_gap_seconds() -> float:
        return float(max(1, int(settings.brain_poll_account_gap_seconds or 30)))


worker = OrchestrationWorker()
