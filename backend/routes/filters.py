"""API routes for alpha filtering."""
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.filters.pipeline import (
    AlphaFilterPipeline,
    ExpressionFilterConfig,
    FilterDecision,
    ResultFilterConfig,
)
from backend.models import Result, get_db

router = APIRouter()


class FilterDecisionResponse(BaseModel):
    """Filter decision response."""

    item_id: Optional[int]
    expression: str
    passed: bool
    reasons: List[str]
    metrics: Dict


class FilterSummaryResponse(BaseModel):
    """Filter summary response."""

    total_count: int
    accepted_count: int
    rejected_count: int
    accepted: List[FilterDecisionResponse]
    rejected: List[FilterDecisionResponse]


class FilterExpressionsRequest(BaseModel):
    """Request to filter expressions."""

    expressions: List[str] = Field(min_length=1)
    existing_expressions: List[str] = Field(default_factory=list)
    require_unique: bool = True
    min_ml_probability: float = Field(default=0.45, ge=0.0, le=1.0)
    max_expression_length: int = Field(default=1000, ge=1, le=10000)
    max_nesting_depth: int = Field(default=8, ge=1, le=50)
    max_operator_count: int = Field(default=16, ge=1, le=200)
    max_constant_count: int = Field(default=20, ge=1, le=200)


class FilterResultsRequest(BaseModel):
    """Request to filter stored results."""

    limit: int = Field(default=500, ge=1, le=5000)
    min_sharpe: float = 1.0
    min_fitness: float = 0.8
    max_turnover: float = 0.70
    max_self_correlation: float = 0.70
    require_checks_passed: bool = True
    min_ml_probability: float = Field(default=0.50, ge=0.0, le=1.0)


@router.get("/rules", tags=["filters"])
def filter_rules() -> Dict:
    """Return default filter thresholds."""
    return {
        "expression": ExpressionFilterConfig().__dict__,
        "result": ResultFilterConfig().__dict__,
    }


@router.post("/expressions", response_model=FilterSummaryResponse, tags=["filters"])
def filter_expressions(request: FilterExpressionsRequest) -> FilterSummaryResponse:
    """Filter generated expressions."""
    summary = AlphaFilterPipeline().filter_expressions(
        expressions=request.expressions,
        existing_expressions=request.existing_expressions,
        config=ExpressionFilterConfig(
            require_unique=request.require_unique,
            min_ml_probability=request.min_ml_probability,
            max_expression_length=request.max_expression_length,
            max_nesting_depth=request.max_nesting_depth,
            max_operator_count=request.max_operator_count,
            max_constant_count=request.max_constant_count,
        ),
    )
    return _summary_response(summary)


@router.post("/results", response_model=FilterSummaryResponse, tags=["filters"])
def filter_results(
    request: FilterResultsRequest,
    db: Session = Depends(get_db),
) -> FilterSummaryResponse:
    """Filter stored result rows by quality gates."""
    results = db.query(Result).order_by(Result.id.desc()).limit(request.limit).all()
    summary = AlphaFilterPipeline().filter_results(
        results=results,
        config=ResultFilterConfig(
            min_sharpe=request.min_sharpe,
            min_fitness=request.min_fitness,
            max_turnover=request.max_turnover,
            max_self_correlation=request.max_self_correlation,
            require_checks_passed=request.require_checks_passed,
            min_ml_probability=request.min_ml_probability,
        ),
    )
    return _summary_response(summary)


@router.get("/results/accepted", response_model=List[FilterDecisionResponse], tags=["filters"])
def accepted_results(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> List[FilterDecision]:
    """Return accepted stored results using default gates."""
    results = db.query(Result).order_by(Result.id.desc()).limit(limit).all()
    return AlphaFilterPipeline().filter_results(results).accepted


def _summary_response(summary) -> FilterSummaryResponse:
    return FilterSummaryResponse(
        total_count=summary.total_count,
        accepted_count=summary.accepted_count,
        rejected_count=summary.rejected_count,
        accepted=[FilterDecisionResponse(**decision.__dict__) for decision in summary.accepted],
        rejected=[FilterDecisionResponse(**decision.__dict__) for decision in summary.rejected],
    )
