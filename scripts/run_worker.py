"""Standalone always-on autopilot worker (for Render / Railway / a VPS).

The Vercel deployment is serverless, so it cannot run the Special Auto loop 24/7.
This script runs that loop in a long-lived process: generate -> submit -> poll ->
learn, forever. Point it at the SAME database as the Vercel UI (DATABASE_URL =
your Postgres) and use the SAME AES_KEY, so it can decrypt the stored BRAIN
account and everything it produces (results, attempt memory, library, trained
model) persists and shows up in the live dashboard.

Run:  python scripts/run_worker.py

Required env (must match the Vercel project):
  DATABASE_URL   postgresql://...   (the shared Postgres)
  AES_KEY        <same as Vercel>   (to decrypt the saved BRAIN account)
Optional env:
  WORKER_DRY_RUN=false  WORKER_AUTO_LEARN=true  WORKER_ACCOUNT_ID=1
  WORKER_SUBMIT_INTERVAL=60  WORKER_POLL_INTERVAL=90  WORKER_LEARN_INTERVAL=600
  WORKER_BATCH_SIZE=5  WORKER_TARGET_RUNNING=4  WORKER_MAX_RUNNING=6
  WORKER_REFILL_BELOW=10  WORKER_MAX_PENDING=15  WORKER_HEARTBEAT=60
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Allow running as `python scripts/run_worker.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models import init_db  # noqa: E402
from backend.workers.orchestration_worker import OrchestrationWorker  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
log = logging.getLogger("run_worker")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _start(worker: OrchestrationWorker) -> None:
    worker.start(
        dry_run=_env_bool("WORKER_DRY_RUN", False),
        auto_learn=_env_bool("WORKER_AUTO_LEARN", True),
        special_auto=True,
        submit_interval_seconds=_env_int("WORKER_SUBMIT_INTERVAL", 60),
        poll_interval_seconds=_env_int("WORKER_POLL_INTERVAL", 90),
        learning_interval_seconds=_env_int("WORKER_LEARN_INTERVAL", 600),
        special_batch_size=_env_int("WORKER_BATCH_SIZE", 5),
        special_target_running=_env_int("WORKER_TARGET_RUNNING", 4),
        special_max_running=_env_int("WORKER_MAX_RUNNING", 6),
        special_refill_pending_below=_env_int("WORKER_REFILL_BELOW", 10),
        special_max_pending=_env_int("WORKER_MAX_PENDING", 15),
        account_ids=[_env_int("WORKER_ACCOUNT_ID", 1)],
    )


def main() -> None:
    log.info("Initializing database (create tables + load invalid-field quarantine)...")
    init_db()

    worker = OrchestrationWorker()
    _start(worker)
    mode = "DRY-RUN" if _env_bool("WORKER_DRY_RUN", False) else "LIVE"
    log.info("Special Auto worker started in %s mode. Looping forever (Ctrl+C to stop).", mode)

    heartbeat = max(15, _env_int("WORKER_HEARTBEAT", 60))
    try:
        while True:
            time.sleep(heartbeat)
            snap = worker.snapshot()
            log.info(
                "heartbeat: ticks=%s special_runs=%s queued=%s last='%s' error='%s'",
                snap.iterations, snap.special_runs, snap.special_queued,
                snap.last_special_message, snap.last_error,
            )
            # Self-restart the inner loop if it ever died (resilience).
            if not snap.running:
                log.warning("Worker thread not running; restarting it.")
                _start(worker)
    except KeyboardInterrupt:
        log.info("Stopping worker...")
        worker.stop()


if __name__ == "__main__":
    main()
