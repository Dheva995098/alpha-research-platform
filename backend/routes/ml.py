"""API routes for ML ranking."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from backend.ml.service import MLRankingService
from backend.ml.auto_learner import AutoLearningService
from backend.models import Result, get_db

router = APIRouter()


class ScoreExpressionRequest(BaseModel):
    """Request to score alpha expressions."""

    expressions: List[str] = Field(min_length=1)
    metrics: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    """ML prediction response."""

    expression: str
    pass_probability: float
    score: float
    feature_values: Dict[str, float]
    reasons: List[str]


class ScoreExpressionResponse(BaseModel):
    """Ranked predictions response."""

    predictions: List[PredictionResponse]


class TrainingResponse(BaseModel):
    """Training summary response."""

    trained: bool
    example_count: int
    positive_count: int
    negative_count: int
    accuracy: Optional[float]
    message: str


class ScoreResultsRequest(BaseModel):
    """Request to score stored result rows."""

    limit: int = Field(default=500, ge=1, le=5000)
    only_unscored: bool = True


class AutoLearnRequest(BaseModel):
    """Request to run the automatic learner."""

    limit: int = Field(default=500, ge=1, le=5000)
    min_examples: int = Field(default=5, ge=5, le=5000)


class ResultScoreResponse(BaseModel):
    """Stored result score response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    brain_alpha_id: Optional[str] = None
    expression: str
    sharpe: Optional[float]
    fitness: Optional[float]
    turnover: Optional[float]
    self_correlation: Optional[float]
    all_checks_passed: Optional[bool]
    raw_metrics: Optional[Dict[str, Any]] = None
    ml_pass_probability: Optional[float]
    final_score: Optional[float]
    human_approved: bool = False

@router.get("/status", tags=["ml"])
def model_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return ML model metadata and feature names."""
    return MLRankingService(db).model_status()


@router.get("/learning-status", tags=["ml"])
def learning_status(
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Return current auto-learning status and recommendations."""
    return AutoLearningService(db).status(limit=limit)


@router.post("/auto-learn", tags=["ml"])
def auto_learn(
    request: AutoLearnRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Train, rescore, and summarize learned alpha patterns."""
    return AutoLearningService(db).run_once(
        limit=request.limit,
        min_examples=request.min_examples,
    )


@router.post("/score", response_model=ScoreExpressionResponse, tags=["ml"])
def score_expressions(request: ScoreExpressionRequest, db: Session = Depends(get_db)) -> ScoreExpressionResponse:
    """Score expressions and return predictions ranked by score."""
    service = MLRankingService(db)
    if request.metrics and len(request.expressions) == 1:
        predictions = [service.score_expression(request.expressions[0], metrics=request.metrics)]
    else:
        predictions = service.score_expressions(request.expressions)
    return ScoreExpressionResponse(
        predictions=[PredictionResponse(**prediction.__dict__) for prediction in predictions]
    )


@router.post("/train", response_model=TrainingResponse, tags=["ml"])
def train_ranker(db: Session = Depends(get_db)) -> TrainingResponse:
    """Train the ranker from leaderboard and result history."""
    result = MLRankingService(db).train_from_db()
    return TrainingResponse(**result.__dict__)


@router.post("/score-results", response_model=List[ResultScoreResponse], tags=["ml"])
def score_results(
    request: ScoreResultsRequest,
    db: Session = Depends(get_db),
) -> List[Result]:
    """Score stored results with the active ML ranker."""
    return MLRankingService(db).score_results(
        limit=request.limit,
        only_unscored=request.only_unscored,
    )


@router.get("/results", response_model=List[ResultScoreResponse], tags=["ml"])
def ranked_results(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> List[Result]:
    """List stored results ordered by final score."""
    return MLRankingService(db).ranked_results(limit=limit)


@router.post("/results/{result_id}/approve-good", response_model=ResultScoreResponse, tags=["ml"])
def approve_good_alpha(
    result_id: int,
    db: Session = Depends(get_db),
) -> Result:
    """Manually promote a live review result into the Good Live vault."""
    result = db.query(Result).filter(Result.id == result_id).first()
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result {result_id} not found")
    if _is_dry_result(result) or not _is_live_result(result):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only live BRAIN results can be approved")
    if _has_failed_checks(result):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Results with failed checks cannot be approved")
    if (result.sharpe or 0.0) < 1.25 or (result.fitness or 0.0) < 1.0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Result does not meet Sharpe/Fitness minimums")
    if result.turnover is not None and result.turnover > 0.70:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Result turnover is above the vault limit")
    if result.self_correlation is not None and result.self_correlation > 0.70:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Result self-correlation is above the vault limit")

    result.human_approved = True
    db.commit()
    db.refresh(result)
    return result


@router.get("/good-alphas", tags=["ml"])
def good_alphas(
    limit: int = Query(default=100, ge=1, le=500),
    min_sharpe: float = Query(default=1.25, ge=-10, le=20),
    min_fitness: float = Query(default=1.0, ge=-10, le=20),
    max_turnover: float = Query(default=0.70, ge=0, le=10),
    live_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Return persisted live results that clear the local good-alpha bar."""
    rows = db.query(Result).order_by(Result.id.desc()).limit(max(limit * 10, 500)).all()
    good_rows = [
        result
        for result in rows
        if _is_good_alpha(
            result,
            min_sharpe=min_sharpe,
            min_fitness=min_fitness,
            max_turnover=max_turnover,
            live_only=live_only,
        )
    ]
    good_rows = sorted(
        good_rows,
        key=lambda item: (
            item.final_score or 0.0,
            item.ml_pass_probability or 0.0,
            item.sharpe or 0.0,
            item.fitness or 0.0,
        ),
        reverse=True,
    )[:limit]
    live_results = [result for result in rows if _is_live_result(result)]
    dry_results = [result for result in rows if _is_dry_result(result)]
    training_seed_results = [result for result in rows if _result_source(result) == "training_seed"]
    return {
        "summary": {
            "good_count": len(good_rows),
            "live_result_count": len(live_results),
            "dry_result_count": len(dry_results),
            "training_seed_count": len(training_seed_results),
            "accept_rate": round(len(good_rows) / len(live_results), 4) if live_results else 0.0,
            "thresholds": {
                "min_sharpe": min_sharpe,
                "min_fitness": min_fitness,
                "max_turnover": max_turnover,
                "live_only": live_only,
            },
        },
        "alphas": [_good_alpha_payload(result) for result in good_rows],
    }


def _is_good_alpha(
    result: Result,
    *,
    min_sharpe: float,
    min_fitness: float,
    max_turnover: float,
    live_only: bool,
) -> bool:
    if live_only and _is_dry_result(result):
        return False
    if live_only and not _is_live_result(result):
        return False
    if result.human_approved:
        if _has_failed_checks(result):
            return False
        if result.self_correlation is not None and result.self_correlation > 0.70:
            return False
    elif result.all_checks_passed is not True:
        return False
    if (result.sharpe or 0.0) < min_sharpe:
        return False
    if (result.fitness or 0.0) < min_fitness:
        return False
    if result.turnover is not None and result.turnover > max_turnover:
        return False
    return True


def _is_dry_result(result: Result) -> bool:
    alpha_id = str(result.brain_alpha_id or "")
    return bool(_raw_metrics(result).get("dry_run") or _result_source(result) == "dry_run" or alpha_id.startswith("dry-run"))


def _is_live_result(result: Result) -> bool:
    return _result_source(result) == "live" and not _is_dry_result(result)


def _result_source(result: Result) -> str:
    return str(_raw_metrics(result).get("source") or "").strip().lower()


def _raw_metrics(result: Result) -> Dict[str, Any]:
    return result.raw_metrics if isinstance(result.raw_metrics, dict) else {}


def _has_failed_checks(result: Result) -> bool:
    raw_metrics = _raw_metrics(result)
    is_block = raw_metrics.get("is") if isinstance(raw_metrics.get("is"), dict) else {}
    checks = is_block.get("checks") or raw_metrics.get("checks") or []
    return any(
        isinstance(check, dict) and str(check.get("result") or "").upper() == "FAIL"
        for check in checks
    )


def _good_alpha_payload(result: Result) -> Dict[str, Any]:
    raw_metrics = result.raw_metrics if isinstance(result.raw_metrics, dict) else {}
    settings = raw_metrics.get("settings")
    if not isinstance(settings, dict) and result.simulation is not None:
        settings = result.simulation.settings
    payload = {
        "id": result.id,
        "brain_alpha_id": result.brain_alpha_id,
        "expression": result.expression,
        "sharpe": result.sharpe,
        "fitness": result.fitness,
        "turnover": result.turnover,
        "self_correlation": result.self_correlation,
        "all_checks_passed": result.all_checks_passed,
        "human_approved": result.human_approved,
        "ml_pass_probability": result.ml_pass_probability,
        "final_score": result.final_score,
        "settings": settings or {},
        "raw_metrics": raw_metrics,
        "created_at": result.created_at,
    }
    payload["copy_text"] = _good_alpha_copy_text(payload)
    return payload


def _good_alpha_copy_text(payload: Dict[str, Any]) -> str:
    settings = payload.get("settings") or {}
    setting_lines = [
        ("Instrument Type", settings.get("instrumentType") or "EQUITY"),
        ("Region", settings.get("region")),
        ("Universe", settings.get("universe")),
        ("Language", settings.get("language")),
        ("Decay", settings.get("decay")),
        ("Delay", settings.get("delay")),
        ("Truncation", settings.get("truncation")),
        ("Neutralization", settings.get("neutralization")),
        ("Pasteurization", settings.get("pasteurization")),
        ("NaN Handling", settings.get("nanHandling")),
        ("Unit Handling", settings.get("unitHandling")),
        ("Max Trade", settings.get("maxTrade")),
        ("Test Period", settings.get("testPeriod")),
    ]
    metrics = (
        f"Sharpe: {payload.get('sharpe')}\n"
        f"Fitness: {payload.get('fitness')}\n"
        f"Turnover: {payload.get('turnover')}\n"
        f"Self Correlation: {payload.get('self_correlation')}\n"
        f"All Checks Passed: {payload.get('all_checks_passed')}"
    )
    rendered_settings = "\n".join(
        f"{label}: {value}" for label, value in setting_lines if value is not None
    )
    return (
        "Alpha Code\n"
        "----------\n"
        f"{payload.get('expression')}\n\n"
        "Simulation Settings\n"
        "-------------------\n"
        f"{rendered_settings}\n\n"
        "Result Metrics\n"
        "--------------\n"
        f"{metrics}\n"
    )
