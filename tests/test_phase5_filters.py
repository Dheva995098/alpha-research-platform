"""Tests for Phase 5 filtering pipeline."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.filters.pipeline import (
    AlphaFilterPipeline,
    ExpressionFilterConfig,
    ResultFilterConfig,
)
from backend.main import app
from backend.models import Base, Result
from backend.routes.filters import FilterExpressionsRequest, filter_expressions


def make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory()


def test_expression_filter_accepts_structured_alpha():
    pipeline = AlphaFilterPipeline()

    summary = pipeline.filter_expressions(
        ["group_neutralize(rank(ts_corr(close, volume, 20)), sector)"],
        config=ExpressionFilterConfig(min_ml_probability=0.40),
    )

    assert summary.accepted_count == 1
    assert summary.rejected_count == 0


def test_expression_filter_rejects_duplicates_and_invalid_expression():
    pipeline = AlphaFilterPipeline()

    summary = pipeline.filter_expressions(
        ["rank(close)", " rank( close ) ", "rank(close); DROP"],
        config=ExpressionFilterConfig(min_ml_probability=0.0),
    )

    assert summary.accepted_count == 1
    assert summary.rejected_count == 2
    assert any("Duplicate expression" in reason for decision in summary.rejected for reason in decision.reasons)
    assert any("Suspicious pattern" in reason for decision in summary.rejected for reason in decision.reasons)


def test_expression_filter_rejects_low_ml_probability():
    pipeline = AlphaFilterPipeline()

    summary = pipeline.filter_expressions(
        ["rank(close)"],
        config=ExpressionFilterConfig(min_ml_probability=0.99),
    )

    assert summary.accepted_count == 0
    assert summary.rejected_count == 1
    assert any("ML probability" in reason for reason in summary.rejected[0].reasons)


def test_result_filter_accepts_good_result_and_rejects_bad_result():
    pipeline = AlphaFilterPipeline()
    good = Result(
        id=1,
        expression="group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
        sharpe=1.5,
        fitness=1.2,
        turnover=0.4,
        self_correlation=0.2,
        all_checks_passed=True,
        ml_pass_probability=0.75,
    )
    bad = Result(
        id=2,
        expression="rank(close)",
        sharpe=0.2,
        fitness=0.1,
        turnover=1.2,
        self_correlation=0.9,
        all_checks_passed=False,
        ml_pass_probability=0.25,
    )

    summary = pipeline.filter_results(
        [good, bad],
        config=ResultFilterConfig(min_ml_probability=0.50),
    )

    assert summary.accepted_count == 1
    assert summary.rejected_count == 1
    assert summary.accepted[0].item_id == 1
    assert summary.rejected[0].item_id == 2


def test_filter_routes_are_registered():
    route_paths = {route.path for route in app.routes}

    assert "/api/filters/rules" in route_paths
    assert "/api/filters/expressions" in route_paths
    assert "/api/filters/results" in route_paths


def test_filter_expression_route_handler_returns_summary():
    response = filter_expressions(
        FilterExpressionsRequest(
            expressions=[
                "group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
                "rank(close); DROP",
            ],
            min_ml_probability=0.0,
        )
    )

    assert response.total_count == 2
    assert response.accepted_count == 1
    assert response.rejected_count == 1
