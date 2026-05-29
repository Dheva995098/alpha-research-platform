"""Tests for Phase 4 ML ranking."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.ml.curated_alpha_seeds import (
    CURATED_PERFECT_ALPHA_SEEDS,
    upsert_curated_perfect_alpha_seeds,
)
from backend.ml.features import ExpressionFeatureExtractor
from backend.ml.ranker import AlphaRanker, TrainingExample
from backend.ml.service import MLRankingService
from backend.models import Base, LeaderboardAlpha, Result
from backend.routes.ml import ScoreExpressionRequest, score_expressions
from scripts.seed_public_alpha_research import parse_jglazar_submitted_alphas


def make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory()


def test_feature_extractor_detects_structure():
    extractor = ExpressionFeatureExtractor()

    features = extractor.extract("group_neutralize(rank(ts_corr(close, volume, 20)), sector)")

    assert "close" in features.fields
    assert "volume" in features.fields
    assert "ts_corr" in features.operators
    assert features.values["has_group_neutralize"] == 1.0
    assert features.values["has_price_volume_pair"] == 1.0
    assert features.windows == [20]


def test_feature_extractor_uses_simulation_settings():
    extractor = ExpressionFeatureExtractor()

    features = extractor.extract(
        "trade_when(pcr_oi_270 < 1, implied_volatility_call_270 - implied_volatility_put_270, -1)",
        metrics={
            "settings": {
                "region": "USA",
                "universe": "TOP500",
                "decay": 4,
                "delay": 1,
                "truncation": 0.08,
                "neutralization": "MARKET",
                "maxTrade": "OFF",
            }
        },
    )

    assert features.values["has_options_data"] == 1.0
    assert features.values["has_alternative_data"] == 1.0
    assert features.values["setting_region_usa"] == 1.0
    assert features.values["setting_universe_top500"] == 1.0
    assert features.values["setting_neutralization_market"] == 1.0
    assert features.values["setting_options_profile"] == 1.0


def test_feature_extractor_detects_model_risk_dataset_fields():
    extractor = ExpressionFeatureExtractor()

    features = extractor.extract("rank(ts_rank(mdl77_momentum, 60) - ts_rank(mdl77_reversal, 60))")

    assert features.values["has_model_risk_data"] == 1.0


def test_public_alpha_research_parser_extracts_metrics_and_settings():
    sample = """USA, TOP3000, Decay 3, Delay 1, Truncation 0.05, Neutralization None\\
Sharpe 1.39, Turnover 17.15%, Fitness 1.11, Returns 10.87%, Drawdown 8.14%, Margin 12.67bps
```
rank(close)
```
"""

    seeds = parse_jglazar_submitted_alphas(sample)

    assert len(seeds) == 1
    assert seeds[0].expression == "rank(close)"
    assert seeds[0].region == "USA"
    assert seeds[0].universe == "TOP3000"
    assert seeds[0].neutralization == "NONE"
    assert round(seeds[0].turnover, 4) == 0.1715
    assert seeds[0].fitness == 1.11


def test_curated_perfect_alpha_seeds_upsert_training_rows():
    db = make_db()

    summary = upsert_curated_perfect_alpha_seeds(db, train=False)
    rows = db.query(Result).all()

    assert summary["seeded"] >= 50
    assert len(rows) == summary["seeded"]
    assert all(row.raw_metrics["source"] == "training_seed" for row in rows)
    assert all(row.raw_metrics["copy_policy"] == "training_signal_only_do_not_copy" for row in rows)
    assert all(row.all_checks_passed is True for row in rows)
    assert all("..." not in seed.expression for seed in CURATED_PERFECT_ALPHA_SEEDS)
    assert rows[0].raw_metrics["settings"]["language"] == "FASTEXPR"


def test_ranker_scores_structured_expression_above_simple_expression():
    ranker = AlphaRanker()

    simple = ranker.predict("rank(close)")
    structured = ranker.predict("group_neutralize(rank(ts_corr(close, volume, 20)), sector)")

    assert structured.pass_probability > simple.pass_probability
    assert structured.score > simple.score


def test_ranker_trains_with_labeled_examples():
    ranker = AlphaRanker()
    examples = [
        TrainingExample("group_neutralize(rank(ts_corr(close, volume, 20)), sector)", 1),
        TrainingExample("rank(ts_rank(returns, 60))", 1),
        TrainingExample("rank(ts_decay_linear(returns, 20))", 1),
        TrainingExample("rank(close)", 0),
        TrainingExample("rank(log(cap))", 0),
        TrainingExample("rank(0 - ts_std_dev(returns, 5))", 0),
    ]

    result = ranker.train(examples, epochs=50)

    assert result.trained is True
    assert result.example_count == 6
    assert result.accuracy is not None


def test_ml_service_trains_from_leaderboard_rows():
    db = make_db()
    rows = [
        LeaderboardAlpha(expression="group_neutralize(rank(ts_corr(close, volume, 20)), sector)", sharpe=1.6, fitness=1.4, turnover=0.4, self_correlation=0.2, passes_checks=True),
        LeaderboardAlpha(expression="rank(ts_rank(returns, 60))", sharpe=1.4, fitness=1.2, turnover=0.5, self_correlation=0.3, passes_checks=True),
        LeaderboardAlpha(expression="rank(ts_decay_linear(returns, 20))", sharpe=1.3, fitness=1.1, turnover=0.6, self_correlation=0.4, passes_checks=True),
        LeaderboardAlpha(expression="rank(close)", sharpe=0.1, fitness=0.0, turnover=1.4, self_correlation=0.8, passes_checks=False),
        LeaderboardAlpha(expression="rank(log(cap))", sharpe=0.2, fitness=0.1, turnover=1.1, self_correlation=0.9, passes_checks=False),
        LeaderboardAlpha(expression="rank(0 - ts_std_dev(returns, 5))", sharpe=-0.1, fitness=-0.2, turnover=1.0, self_correlation=0.7, passes_checks=False),
    ]
    db.add_all(rows)
    db.commit()

    service = MLRankingService(db)
    result = service.train_from_db()
    status = service.model_status()

    assert result.trained is True
    assert status["trained"] is True
    assert status["trained_on_count"] == 6


def test_ml_service_scores_stored_results():
    db = make_db()
    result = Result(
        expression="group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
        sharpe=1.5,
        fitness=1.2,
        turnover=0.45,
        self_correlation=0.2,
        all_checks_passed=True,
        final_score=1.1,
    )
    db.add(result)
    db.commit()

    scored = MLRankingService(db).score_results()

    assert len(scored) == 1
    assert scored[0].ml_pass_probability is not None
    assert scored[0].final_score is not None


def test_ml_routes_are_registered():
    route_paths = {route.path for route in app.routes}

    assert "/api/ml/status" in route_paths
    assert "/api/ml/score" in route_paths
    assert "/api/ml/train" in route_paths
    assert "/api/ml/score-results" in route_paths


def test_score_route_handler_returns_ranked_predictions():
    db = make_db()

    response = score_expressions(
        ScoreExpressionRequest(
            expressions=[
                "rank(close)",
                "group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
            ]
        ),
        db=db,
    )

    assert len(response.predictions) == 2
    assert response.predictions[0].score >= response.predictions[1].score
