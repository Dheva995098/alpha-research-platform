"""API routes for simulation queue orchestration."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from backend.models import Simulation, get_db
from backend.orchestration.service import QUEUE_STATUSES, SimulationOrchestrator
from backend.workers.orchestration_worker import worker

router = APIRouter()


class SimulationResponse(BaseModel):
    """Simulation queue item response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    account_id: int
    brain_simulation_id: Optional[str] = None
    expression: str
    settings: Optional[Dict[str, Any]] = None
    status: str
    progress: float
    error_message: Optional[str] = None
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class QueueRequest(BaseModel):
    """Request to enqueue expressions."""

    model_config = ConfigDict(populate_by_name=True)

    expressions: List[str] = Field(min_length=1)
    account_ids: Optional[List[int]] = None
    validate_expressions: bool = Field(default=True, alias="validate")
    settings: Optional[Dict[str, Any]] = None


class QueueResponse(BaseModel):
    """Queue operation response."""

    message: str
    queued_count: int
    simulations: List[SimulationResponse]
    duplicate_count: int = 0
    skipped: List[Dict[str, str]] = Field(default_factory=list)


class SubmitNextRequest(BaseModel):
    """Request to submit the next pending simulation."""

    universe: str = "default"
    dry_run: bool = False


class SubmitResultLiveRequest(BaseModel):
    """Request to submit one stored result to live BRAIN."""

    universe: str = "default"
    settings: Optional[Dict[str, Any]] = None


class PollRequest(BaseModel):
    """Request to poll running simulations."""

    limit: int = Field(default=25, ge=1, le=200)
    per_account_limit: Optional[int] = Field(default=None, ge=1, le=10)


class OperationResponse(BaseModel):
    """Generic orchestration operation response."""

    action: str
    ok: bool
    message: str
    errors: List[str] = Field(default_factory=list)
    simulations: List[SimulationResponse] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClearQueueRequest(BaseModel):
    """Request to clear terminal queue rows."""

    statuses: List[str] = Field(default_factory=lambda: ["failed", "cancelled"])


class ClearPendingRequest(BaseModel):
    """Request to clear local pending queue rows."""

    keep_latest: int = Field(default=0, ge=0, le=1000)


class WorkerStartRequest(BaseModel):
    """Request to start the background worker."""

    submit_interval_seconds: int = Field(default=30, ge=1, le=3600)
    poll_interval_seconds: int = Field(default=20, ge=1, le=3600)
    universe: str = "default"
    dry_run: bool = True
    auto_learn: bool = False
    learning_interval_seconds: int = Field(default=300, ge=60, le=86400)
    special_auto: bool = False
    special_batch_size: int = Field(default=5, ge=1, le=20)
    special_target_running: int = Field(default=5, ge=1, le=25)
    special_max_running: int = Field(default=6, ge=1, le=30)
    special_refill_pending_below: int = Field(default=10, ge=0, le=500)
    special_max_pending: int = Field(default=15, ge=0, le=1000)
    special_stale_running_minutes: int = Field(default=240, ge=15, le=1440)
    openai_assist: bool = False
    account_ids: Optional[List[int]] = None


class WorkerStateResponse(BaseModel):
    """Background worker state response."""

    running: bool
    dry_run: bool
    universe: str
    submit_interval_seconds: int
    poll_interval_seconds: int
    auto_learn: bool
    learning_interval_seconds: int
    special_auto: bool
    special_batch_size: int
    special_target_running: int
    special_max_running: int
    special_refill_pending_below: int
    special_max_pending: int
    special_stale_running_minutes: int
    openai_assist: bool
    account_ids: Optional[List[int]] = None
    iterations: int
    special_runs: int
    special_queued: int
    started_at: Optional[datetime] = None
    last_tick_at: Optional[datetime] = None
    last_learning_at: Optional[datetime] = None
    last_learning_message: Optional[str] = None
    last_special_at: Optional[datetime] = None
    last_special_message: Optional[str] = None
    last_error: Optional[str] = None


@router.post("/queue", response_model=QueueResponse, status_code=status.HTTP_201_CREATED, tags=["orchestration"])
def enqueue_simulations(
    request: QueueRequest,
    db: Session = Depends(get_db),
) -> QueueResponse:
    """Queue alpha expressions as pending simulations."""
    result = SimulationOrchestrator().enqueue_expressions(
        db,
        expressions=request.expressions,
        account_ids=request.account_ids,
        validate=request.validate_expressions,
        settings=request.settings,
    )
    if result.errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.errors)

    return QueueResponse(
        message=result.message,
        queued_count=len(result.simulations),
        simulations=result.simulations,
        duplicate_count=result.metadata.get("duplicate_count", 0),
        skipped=result.metadata.get("skipped", []),
    )


@router.get("/queue", response_model=List[SimulationResponse], tags=["orchestration"])
def list_queue(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> List[Simulation]:
    """List queued simulations."""
    if status_filter and status_filter not in QUEUE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown status: {status_filter}",
        )
    return SimulationOrchestrator().list_simulations(db, status=status_filter, limit=limit)


@router.get("/summary", tags=["orchestration"])
def orchestration_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return queue status counts and account quotas."""
    return SimulationOrchestrator().queue_summary(db)


@router.post("/submit-next", response_model=OperationResponse, tags=["orchestration"])
def submit_next(
    request: SubmitNextRequest,
    db: Session = Depends(get_db),
) -> OperationResponse:
    """Submit the next pending simulation."""
    result = SimulationOrchestrator().submit_next(
        db,
        universe=request.universe,
        dry_run=request.dry_run,
    )
    return _operation_response(result)


@router.post("/results/{result_id}/live-submit", response_model=OperationResponse, tags=["orchestration"])
def submit_result_live(
    result_id: int,
    request: SubmitResultLiveRequest,
    db: Session = Depends(get_db),
) -> OperationResponse:
    """Submit a stored result expression as a fresh live BRAIN simulation."""
    result = SimulationOrchestrator().submit_result_live(
        db,
        result_id=result_id,
        universe=request.universe,
        settings=request.settings,
    )
    if result.errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.errors)
    return _operation_response(result)


@router.post("/poll", response_model=OperationResponse, tags=["orchestration"])
def poll_running(
    request: PollRequest,
    db: Session = Depends(get_db),
) -> OperationResponse:
    """Poll running simulations and persist completed results."""
    result = SimulationOrchestrator().poll_running(
        db,
        limit=request.limit,
        per_account_limit=request.per_account_limit,
    )
    return _operation_response(result)


@router.post("/queue/{simulation_id}/cancel", response_model=OperationResponse, tags=["orchestration"])
def cancel_pending(
    simulation_id: int,
    db: Session = Depends(get_db),
) -> OperationResponse:
    """Cancel a pending simulation."""
    result = SimulationOrchestrator().cancel_pending(db, simulation_id)
    if result.errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.errors)
    return _operation_response(result)


@router.post("/queue/clear-terminal", response_model=OperationResponse, tags=["orchestration"])
def clear_terminal(
    request: ClearQueueRequest,
    db: Session = Depends(get_db),
) -> OperationResponse:
    """Clear failed/cancelled queue rows."""
    invalid_statuses = [item for item in request.statuses if item not in {"failed", "cancelled", "completed"}]
    if invalid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only failed, cancelled, and completed rows can be cleared: {invalid_statuses}",
        )
    result = SimulationOrchestrator().clear_terminal(db, statuses=request.statuses)
    return _operation_response(result)


@router.post("/queue/clear-pending", response_model=OperationResponse, tags=["orchestration"])
def clear_pending(
    request: ClearPendingRequest,
    db: Session = Depends(get_db),
) -> OperationResponse:
    """Clear local pending queue rows without touching running/completed results."""
    result = SimulationOrchestrator().clear_pending(db, keep_latest=request.keep_latest)
    return _operation_response(result)


@router.post("/worker/start", response_model=WorkerStateResponse, tags=["orchestration"])
def start_worker(request: WorkerStartRequest) -> WorkerStateResponse:
    """Start the in-process orchestration worker."""
    return WorkerStateResponse(
        **worker.start(
            submit_interval_seconds=request.submit_interval_seconds,
            poll_interval_seconds=request.poll_interval_seconds,
            universe=request.universe,
            dry_run=request.dry_run,
            auto_learn=request.auto_learn,
            learning_interval_seconds=request.learning_interval_seconds,
            special_auto=request.special_auto,
            special_batch_size=request.special_batch_size,
            special_target_running=request.special_target_running,
            special_max_running=request.special_max_running,
            special_refill_pending_below=request.special_refill_pending_below,
            special_max_pending=request.special_max_pending,
            special_stale_running_minutes=request.special_stale_running_minutes,
            openai_assist=request.openai_assist,
            account_ids=request.account_ids,
        ).__dict__
    )


@router.post("/worker/stop", response_model=WorkerStateResponse, tags=["orchestration"])
def stop_worker() -> WorkerStateResponse:
    """Stop the in-process orchestration worker."""
    return WorkerStateResponse(**worker.stop().__dict__)


@router.get("/worker/status", response_model=WorkerStateResponse, tags=["orchestration"])
def worker_status() -> WorkerStateResponse:
    """Return the in-process orchestration worker state."""
    return WorkerStateResponse(**worker.snapshot().__dict__)


def _operation_response(result) -> OperationResponse:
    return OperationResponse(
        action=result.action,
        ok=result.ok,
        message=result.message,
        errors=result.errors,
        simulations=result.simulations,
        metadata=result.metadata,
    )
