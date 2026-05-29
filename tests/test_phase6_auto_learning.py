"""Tests for automatic learning feedback loop."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.ml.curated_alpha_seeds import upsert_curated_perfect_alpha_seeds
from backend.ml.auto_learner import AutoLearningService
from backend.models import Base, LeaderboardAlpha, Result, Simulation
from backend.routes.ml import approve_good_alpha, good_alphas


def make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory()


def test_auto_learning_trains_scores_and_recommends_patterns():
    db = make_db()
    db.add_all(
        [
            LeaderboardAlpha(
                expression="group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
                sharpe=1.6,
                fitness=1.3,
                turnover=0.4,
                self_correlation=0.2,
                passes_checks=True,
            ),
            LeaderboardAlpha(
                expression="rank(ts_rank(returns, 60))",
                sharpe=1.4,
                fitness=1.1,
                turnover=0.5,
                self_correlation=0.3,
                passes_checks=True,
            ),
            LeaderboardAlpha(
                expression="rank(close)",
                sharpe=0.2,
                fitness=0.1,
                turnover=1.1,
                self_correlation=0.8,
                passes_checks=False,
            ),
            LeaderboardAlpha(
                expression="rank(log(cap))",
                sharpe=0.1,
                fitness=0.0,
                turnover=1.3,
                self_correlation=0.9,
                passes_checks=False,
            ),
            LeaderboardAlpha(
                expression="rank(0 - ts_std_dev(returns, 5))",
                sharpe=-0.1,
                fitness=-0.2,
                turnover=1.0,
                self_correlation=0.7,
                passes_checks=False,
            ),
        ]
    )
    good_sim = Simulation(
        expression="group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
        settings={"region": "USA", "universe": "TOP3000", "decay": 8, "neutralization": "NONE"},
        status="completed",
    )
    bad_sim = Simulation(
        expression="rank(close)",
        settings={"region": "USA", "universe": "TOP3000", "decay": 10, "neutralization": "SUBINDUSTRY"},
        status="completed",
    )
    db.add_all([good_sim, bad_sim])
    db.commit()
    db.add_all(
        [
            Result(
                simulation_id=good_sim.id,
                expression=good_sim.expression,
                brain_alpha_id="live-good",
                sharpe=1.55,
                fitness=1.2,
                turnover=0.45,
                self_correlation=0.2,
                all_checks_passed=True,
                raw_metrics={"source": "live", "settings": good_sim.settings},
            ),
            Result(
                simulation_id=bad_sim.id,
                expression=bad_sim.expression,
                brain_alpha_id="live-bad",
                sharpe=0.2,
                fitness=0.1,
                turnover=1.1,
                self_correlation=0.8,
                all_checks_passed=False,
                raw_metrics={"source": "live", "settings": bad_sim.settings},
            ),
            Result(
                expression="rank(ts_rank(returns, 20))",
                brain_alpha_id="dry-run-1",
                sharpe=1.8,
                fitness=1.5,
                turnover=0.4,
                self_correlation=0.2,
                all_checks_passed=True,
                raw_metrics={"source": "dry_run", "dry_run": True},
            ),
        ]
    )
    db.commit()

    summary = AutoLearningService(db).run_once(limit=20)

    assert summary["trained"] is True
    assert summary["scored_count"] == 3
    assert summary["live_result_count"] == 2
    assert summary["dry_result_count"] == 1
    assert summary["positive_result_count"] == 1
    assert summary["best_focuses"][0]["name"] == "price_volume"
    assert summary["best_settings"][0]["settings"]["decay"] == 8


def test_auto_learning_reports_curated_seed_patterns_without_live_results():
    db = make_db()

    upsert_curated_perfect_alpha_seeds(db, train=False)
    summary = AutoLearningService(db).status(limit=5000)

    assert summary["live_result_count"] == 0
    assert summary["training_seed_count"] >= 50
    assert summary["best_training_seed_focuses"]
    assert summary["best_training_seed_settings"]


def test_auto_learning_routes_are_registered():
    route_paths = {route.path for route in app.routes}

    assert "/api/ml/auto-learn" in route_paths
    assert "/api/ml/learning-status" in route_paths
    assert "/api/ml/good-alphas" in route_paths


def test_good_alpha_vault_returns_only_live_quality_rows():
    db = make_db()
    db.add_all(
        [
            Result(
                expression="rank(ts_corr(close, volume, 20))",
                brain_alpha_id="live-good",
                sharpe=1.6,
                fitness=1.2,
                turnover=0.4,
                self_correlation=0.2,
                all_checks_passed=True,
                raw_metrics={
                    "source": "live",
                    "settings": {"region": "USA", "universe": "TOP3000", "decay": 8},
                },
                final_score=1.4,
            ),
            Result(
                expression="rank(close)",
                brain_alpha_id="live-bad",
                sharpe=0.6,
                fitness=0.3,
                turnover=0.4,
                all_checks_passed=True,
                raw_metrics={"source": "live"},
            ),
            Result(
                expression="group_neutralize(rank(0 - ts_corr(est_ptp, est_fcf, 60)), market)",
                brain_alpha_id="live-review",
                sharpe=1.49,
                fitness=1.15,
                turnover=0.058,
                all_checks_passed=False,
                raw_metrics={
                    "source": "live",
                    "is": {
                        "checks": [
                            {"name": "LOW_SHARPE", "result": "PASS"},
                            {"name": "LOW_FITNESS", "result": "PASS"},
                            {"name": "SELF_CORRELATION", "result": "PENDING"},
                        ]
                    },
                },
            ),
            Result(
                expression="rank(returns)",
                brain_alpha_id="dry-run-1",
                sharpe=2.0,
                fitness=1.8,
                turnover=0.2,
                all_checks_passed=True,
                raw_metrics={"source": "dry_run", "dry_run": True},
            ),
            Result(
                expression="rank(volume)",
                brain_alpha_id="training-seed-good",
                sharpe=3.0,
                fitness=2.0,
                turnover=0.2,
                all_checks_passed=True,
                raw_metrics={"source": "training_seed"},
            ),
        ]
    )
    db.commit()

    response = good_alphas(
        limit=100,
        min_sharpe=1.25,
        min_fitness=1.0,
        max_turnover=0.70,
        live_only=True,
        db=db,
    )

    assert response["summary"]["good_count"] == 1
    assert response["summary"]["training_seed_count"] == 1
    assert response["alphas"][0]["brain_alpha_id"] == "live-good"
    assert "Simulation Settings" in response["alphas"][0]["copy_text"]

    review = db.query(Result).filter(Result.brain_alpha_id == "live-review").first()
    approve_good_alpha(review.id, db=db)
    response = good_alphas(
        limit=100,
        min_sharpe=1.25,
        min_fitness=1.0,
        max_turnover=0.70,
        live_only=True,
        db=db,
    )

    alpha_ids = {row["brain_alpha_id"] for row in response["alphas"]}
    assert response["summary"]["good_count"] == 2
    assert "live-review" in alpha_ids
