"""Tests for the self-improving research loop (Phase 7).

Covers the four grafted patterns:
  A  objective evaluator + failure diagnosis (evaluator.py)
  B/H persistent attempt memory + auto-growing win library (memory.py)
  E  deterministic failure->fix refiner (refiner.py)
  C/D feedback-into-generation + diversification (feedback.py)
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import random as _random

from backend.models import AlphaLibrary, AttemptMemory, Base, Result, Simulation
from backend.selfimprove import bandit, feedback
from backend.selfimprove.evaluator import (
    CONCENTRATED_WEIGHT,
    COVERAGE_FAIL,
    FAILED_CHECKS,
    HIGH_PROD_CORRELATION,
    HIGH_SELF_CORRELATION,
    HIGH_TURNOVER,
    LOW_SHARPE,
    LOW_SUB_UNIVERSE_SHARPE,
    LOW_TURNOVER,
    UNITS,
    GateConfig,
    Verdict,
    composite_score,
    diagnose,
    evaluate,
)
from backend.selfimprove.memory import AttemptMemoryService, settings_signature
from backend.selfimprove.refiner import DeterministicRefiner


def make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def _make_result(db, expression, sharpe, fitness, turnover, self_corr, checks_passed, settings=None):
    sim = Simulation(expression=expression, settings=settings or {}, status="completed")
    db.add(sim)
    db.flush()
    res = Result(
        simulation_id=sim.id,
        expression=expression,
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover,
        self_correlation=self_corr,
        all_checks_passed=checks_passed,
        raw_metrics={"settings": settings or {}},
    )
    db.add(res)
    db.commit()
    db.refresh(res)
    return res


# --------------------------------------------------------------------------
# Pattern A: evaluator
# --------------------------------------------------------------------------
def test_evaluate_win():
    verdict = evaluate(
        {"sharpe": 1.5, "fitness": 1.2, "turnover": 0.4, "self_correlation": 0.2, "all_checks_passed": True}
    )
    assert verdict.is_ok
    assert verdict.outcome == "win"
    assert verdict.failures == []
    assert verdict.score > 0


def test_evaluate_near_high_turnover():
    verdict = evaluate(
        {"sharpe": 1.4, "fitness": 1.1, "turnover": 0.85, "self_correlation": 0.2, "all_checks_passed": False}
    )
    assert not verdict.is_ok
    assert HIGH_TURNOVER in verdict.failures
    assert verdict.outcome == "near"  # signal present, turnover within repairable slack


def test_evaluate_fail_when_no_signal():
    verdict = evaluate(
        {"sharpe": 0.3, "fitness": 0.2, "turnover": 0.9, "self_correlation": 0.9, "all_checks_passed": False}
    )
    assert verdict.outcome == "fail"
    assert LOW_SHARPE in verdict.failures
    assert HIGH_TURNOVER in verdict.failures
    assert HIGH_SELF_CORRELATION in verdict.failures


def test_diagnose_picks_up_named_brain_checks():
    metrics = {
        "sharpe": 1.4,
        "fitness": 1.1,
        "turnover": 0.5,
        "self_correlation": 0.5,
        "all_checks_passed": False,
        "checks": [
            {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"},
            {"name": "LOW_TURNOVER", "result": "PASS"},
        ],
    }
    tags = diagnose(metrics)
    assert LOW_SUB_UNIVERSE_SHARPE in tags


def test_composite_score_is_defined_on_pass_and_ranks():
    strong = composite_score({"sharpe": 2.0, "fitness": 1.5, "turnover": 0.3, "all_checks_passed": True})
    weak = composite_score({"sharpe": 1.3, "fitness": 1.0, "turnover": 0.3, "all_checks_passed": True})
    assert strong > weak


def test_relaxed_gate_passes_lower_bar():
    metrics = {"sharpe": 1.05, "fitness": 0.85, "turnover": 0.4, "self_correlation": 0.2, "all_checks_passed": True}
    assert evaluate(metrics).outcome != "win"  # strict bar
    assert evaluate(metrics, GateConfig.relaxed()).is_ok  # relaxed bar


# --------------------------------------------------------------------------
# Patterns B / H: memory + library
# --------------------------------------------------------------------------
def test_memory_records_attempt_and_promotes_win():
    db = make_db()
    mem = AttemptMemoryService(db)
    win = _make_result(
        db,
        "group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
        1.6, 1.3, 0.4, 0.2, True,
        {"neutralization": "SECTOR"},
    )
    verdict = mem.record_result(win, focus="price_volume", dataset_id="pv1")
    assert verdict.is_win
    assert db.query(AttemptMemory).count() == 1
    assert db.query(AlphaLibrary).count() == 1
    assert mem.library_expressions(focus="price_volume")


def test_memory_records_failure_without_promoting():
    db = make_db()
    mem = AttemptMemoryService(db)
    fail = _make_result(db, "rank(close)", 0.2, 0.1, 1.1, 0.8, False)
    verdict = mem.record_result(fail)
    assert verdict.outcome in {"fail", "near"}
    assert db.query(AlphaLibrary).count() == 0
    recent = mem.recent_failures(limit=3)
    assert recent and recent[0].expression


def test_memory_upsert_increments_attempts():
    db = make_db()
    mem = AttemptMemoryService(db)
    verdict = evaluate({"sharpe": 0.2, "fitness": 0.1, "turnover": 1.1})
    mem.record_attempt("rank(close)", verdict, settings={"decay": 10})
    mem.record_attempt("rank(close)", verdict, settings={"decay": 10})
    rows = db.query(AttemptMemory).all()
    assert len(rows) == 1
    assert rows[0].attempts == 2


def test_recent_near_misses_available_for_refiner():
    db = make_db()
    mem = AttemptMemoryService(db)
    near = _make_result(
        db, "rank(ts_rank(returns, 60))", 1.4, 1.1, 0.85, 0.2, False, {"neutralization": "SUBINDUSTRY"}
    )
    mem.record_result(near)
    near_misses = mem.recent_near_misses(limit=5)
    assert near_misses
    assert near_misses[0].outcome == "near"


def test_tried_signatures_and_stats():
    db = make_db()
    mem = AttemptMemoryService(db)
    mem.record_attempt("rank(close)", evaluate({"sharpe": 0.2, "fitness": 0.1}))
    sigs = mem.tried_signatures()
    assert len(sigs) == 1
    stats = mem.stats()
    assert stats["attempts"] == 1


def test_settings_signature_distinguishes_settings():
    assert settings_signature({"decay": 10}) != settings_signature({"decay": 20})
    assert settings_signature(None) is None


# --------------------------------------------------------------------------
# Pattern E: deterministic refiner
# --------------------------------------------------------------------------
def test_refiner_repairs_turnover_with_decay():
    verdict = evaluate(
        {"sharpe": 1.4, "fitness": 1.1, "turnover": 0.85, "self_correlation": 0.2, "all_checks_passed": False}
    )
    refiner = DeterministicRefiner()
    variants = refiner.repair(
        "rank(ts_rank(returns, 60))", verdict, settings={"decay": 8, "neutralization": "SUBINDUSTRY"}
    )
    assert variants
    assert any("ts_decay_linear" in v.expression for v in variants)
    for variant in variants:
        ok, _ = refiner.schema.validate_expression_basic(variant.expression)
        assert ok, variant.expression


def test_refiner_repairs_coverage_with_backfill():
    verdict = Verdict(False, [COVERAGE_FAIL], 1.0, "near", {})
    variants = DeterministicRefiner().repair("rank(operating_income)", verdict)
    assert any("ts_backfill" in v.expression for v in variants)


def test_refiner_swaps_neutralization_for_sub_universe():
    verdict = Verdict(False, [LOW_SUB_UNIVERSE_SHARPE], 1.0, "near", {})
    variants = DeterministicRefiner().repair(
        "rank(ts_corr(close, volume, 20))", verdict, settings={"neutralization": "SUBINDUSTRY"}
    )
    assert any("group_neutralize" in v.expression for v in variants)


def test_diagnose_low_turnover_distinct_from_high_turnover():
    # < 1% turnover is a distinct defect, must NOT be tagged HIGH_TURNOVER.
    tags = diagnose({"sharpe": 1.4, "fitness": 1.1, "turnover": 0.005, "all_checks_passed": True})
    assert LOW_TURNOVER in tags
    assert HIGH_TURNOVER not in tags


def test_diagnose_ignores_eligibility_checks():
    metrics = {
        "sharpe": 1.5, "fitness": 1.2, "turnover": 0.4, "self_correlation": 0.2,
        "all_checks_passed": False,
        "checks": [{"name": "MATCHES_COMPETITION", "result": "FAIL"}],
    }
    # A failing eligibility flag is not a quality failure.
    assert FAILED_CHECKS not in diagnose(metrics)


def test_refiner_low_turnover_sharpens_not_smooths():
    verdict = Verdict(False, [LOW_TURNOVER], 1.0, "near", {})
    variants = DeterministicRefiner().repair("rank(close)", verdict, settings={"decay": 10})
    assert variants
    joined = " ".join(v.expression for v in variants)
    # Must raise turnover (sharpen), never wrap in a turnover-reducing decay/smoother.
    assert "ts_zscore" in joined or "ts_delta" in joined
    assert "ts_decay_linear" not in joined


def test_refiner_self_correlation_uses_vector_neut():
    verdict = Verdict(False, [HIGH_SELF_CORRELATION], 1.0, "near", {})
    variants = DeterministicRefiner().repair("rank(ts_corr(close, volume, 20))", verdict)
    assert any("vector_neut" in v.expression for v in variants)
    for v in variants:
        ok, _ = DeterministicRefiner().schema.validate_expression_basic(v.expression)
        assert ok, v.expression


def test_refiner_concentrated_weight_clips_and_tightens_truncation():
    verdict = Verdict(False, [CONCENTRATED_WEIGHT], 1.0, "near", {})
    variants = DeterministicRefiner().repair("ts_rank(returns, 60)", verdict, settings={"truncation": 0.1})
    assert variants
    assert any("winsorize" in v.expression or "rank" in v.expression for v in variants)
    assert any((v.settings or {}).get("truncation") == 0.01 for v in variants)


def test_refiner_units_wraps_in_rank():
    verdict = Verdict(False, [UNITS], 1.0, "near", {})
    variants = DeterministicRefiner().repair("ebit / sales", verdict)
    assert any("rank" in v.expression or "zscore" in v.expression for v in variants)


def test_refiner_turnover_offers_hump_or_trade_when():
    verdict = Verdict(False, [HIGH_TURNOVER], 1.0, "near", {})
    variants = DeterministicRefiner().repair("rank(ts_delta(close, 5))", verdict, settings={"decay": 6})
    joined = " ".join(v.expression for v in variants)
    assert "hump" in joined or "trade_when" in joined


def test_refiner_variants_have_fresh_signatures():
    verdict = evaluate(
        {"sharpe": 1.3, "fitness": 1.0, "turnover": 0.9, "self_correlation": 0.2, "all_checks_passed": False}
    )
    refiner = DeterministicRefiner()
    base = "rank(ts_rank(returns, 60))"
    variants = refiner.repair(base, verdict, settings={"decay": 8})
    from backend.generation.dedup import expression_signature

    base_sig = expression_signature(base)
    assert variants
    assert all(expression_signature(v.expression) != base_sig for v in variants)


# --------------------------------------------------------------------------
# Patterns C / D: feedback + diversification
# --------------------------------------------------------------------------
def test_feedback_context_and_term_weights():
    db = make_db()
    mem = AttemptMemoryService(db)
    failing = _make_result(db, "rank(ts_zscore(close, 20))", 0.3, 0.2, 0.9, 0.85, False)
    mem.record_result(failing)
    failures = mem.recent_failures(limit=3)
    context = feedback.build_failure_context(failures)
    assert "avoid" in context.lower()
    assert "ts_zscore" in context
    weights = feedback.failure_term_weights(failures)
    assert weights
    penalty = feedback.shape_penalty("ts_zscore(close, 20)", weights)
    assert penalty > 0
    assert feedback.shape_penalty("rank(returns)", {}) == 0.0


# --------------------------------------------------------------------------
# The closed loop: prove the autopilot LEARNS across results, not just runs.
# --------------------------------------------------------------------------
from backend.automation.special_runner import SpecialAutopilot
from backend.core.field_intelligence import schema_with_persisted_fields
from backend.models import Account
from backend.orchestration.service import SimulationOrchestrator
from backend.security import encrypt_credential


class _NoopGateway:
    def submit_expression(self, *args, **kwargs):
        return "noop-1"

    def get_status(self, *args, **kwargs):
        return {"status": "completed", "progress": 100}

    def get_results(self, *args, **kwargs):
        return {}


def _add_account(db):
    account = Account(
        brain_email="loop@example.com",
        brain_password_encrypted=encrypt_credential("secret"),
        daily_quota=50,
        submissions_today=0,
        is_active=True,
        worker_enabled=True,
        max_running=6,
        max_pending=15,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def _completed_result(db, account, expression, sharpe, fitness, turnover, self_corr, checks_passed, settings):
    sim = Simulation(
        account_id=account.id,
        expression=expression,
        settings=settings,
        status="completed",
        brain_simulation_id=f"live-{expression[:6]}",
    )
    db.add(sim)
    db.flush()
    res = Result(
        account_id=account.id,
        simulation_id=sim.id,
        brain_alpha_id=f"alpha-live-{sim.id}",
        expression=expression,
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover,
        self_correlation=self_corr,
        all_checks_passed=checks_passed,
        raw_metrics={"source": "live", "settings": settings},
    )
    db.add(res)
    db.commit()
    return res


def test_loop_absorbs_results_then_repairs_a_near_miss():
    """A near-miss result -> recorded in memory -> deterministically repaired & queued."""
    db = make_db()
    account = _add_account(db)
    orchestrator = SimulationOrchestrator(gateway=_NoopGateway())
    autopilot = SpecialAutopilot(seed=11)

    _completed_result(
        db, account, "rank(ts_rank(returns, 60))",
        1.4, 1.1, 0.85, 0.2, False,
        {"region": "USA", "universe": "TOP3000", "decay": 8, "neutralization": "SUBINDUSTRY"},
    )

    absorbed = autopilot._absorb_results(db, [account.id])
    assert absorbed == 1
    assert db.query(AttemptMemory).filter(AttemptMemory.outcome == "near").count() == 1

    repaired = autopilot._queue_repairs(
        db, orchestrator, account_ids=[account.id], pending_room=10
    )
    assert repaired > 0
    pending = db.query(Simulation).filter(Simulation.status == "pending").all()
    assert pending
    repaired_expressions = " ".join(sim.expression for sim in pending)
    assert "ts_decay_linear" in repaired_expressions or "ts_backfill" in repaired_expressions


def test_loop_promotes_a_win_and_seeds_generation_from_it():
    """A winning result -> promoted to library -> mutated into fresh candidates."""
    db = make_db()
    account = _add_account(db)
    autopilot = SpecialAutopilot(seed=7)

    _completed_result(
        db, account, "group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
        1.7, 1.4, 0.35, 0.15, True,
        {"region": "USA", "universe": "TOP3000", "decay": 8, "neutralization": "SECTOR"},
    )

    absorbed = autopilot._absorb_results(db, [account.id])
    assert absorbed == 1
    assert db.query(AlphaLibrary).count() == 1

    schema = schema_with_persisted_fields(db)
    seeded = autopilot._library_seed_candidates(
        db, schema, focus="price_volume", seed=7, batch_size=5, existing=[], candidates=[]
    )
    assert seeded  # proven winner mutated into fresh candidates


def test_bandit_thompson_prefers_higher_winrate_arm():
    rng = _random.Random(0)
    stats = {"momentum": (40, 50), "sentiment": (1, 50)}
    picks = [bandit.thompson_select(["momentum", "sentiment"], stats, rng) for _ in range(200)]
    assert picks.count("momentum") > picks.count("sentiment") * 3


def test_bandit_has_signal_and_safe_fallback():
    assert bandit.has_signal(["a"], {"a": (0, 1)})
    assert not bandit.has_signal(["a", "b"], {})
    assert bandit.thompson_select(["a", "b"], {}, _random.Random(0)) in {"a", "b"}


def test_arm_stats_aggregates_pass_fail_by_focus():
    db = make_db()
    mem = AttemptMemoryService(db)
    win = evaluate({"sharpe": 1.6, "fitness": 1.3, "turnover": 0.4, "self_correlation": 0.2, "all_checks_passed": True})
    fail = evaluate({"sharpe": 0.2, "fitness": 0.1, "turnover": 1.1, "all_checks_passed": False})
    mem.record_attempt("rank(close)", win, focus="momentum")
    mem.record_attempt("rank(volume)", fail, focus="sentiment")
    stats = mem.arm_stats("focus")
    assert stats.get("momentum", (0, 0))[0] == 1  # one win
    assert stats.get("sentiment", (0, 0)) == (0, 1)  # one trial, no win


def test_proven_motifs_validate_and_use_decorrelation():
    from backend.core.data_fields import get_data_fields
    from backend.selfimprove import motifs as motifs_mod

    schema = get_data_fields()
    pv = motifs_mod.motif_candidates("price_volume", schema=schema, limit=10)
    assert pv
    for candidate in pv:
        ok, _ = schema.validate_expression_basic(candidate.expression)
        assert ok, candidate.expression
        assert candidate.rationale == "proven_motif"
    # The negated price-volume reversal motif is present (101-Alphas core structure).
    assert any(c.expression.startswith("-ts_corr") or "-rank(ts_corr" in c.expression for c in pv)
    # Decorrelation motifs use vector_neut (only possible now that it is whitelisted).
    dec = motifs_mod.motif_candidates("decorrelation", schema=schema, limit=10)
    assert any("vector_neut" in c.expression for c in dec)


def test_loop_is_a_noop_on_empty_memory():
    """No prior outcomes -> no repairs, no library seeds (existing behaviour preserved)."""
    db = make_db()
    account = _add_account(db)
    orchestrator = SimulationOrchestrator(gateway=_NoopGateway())
    autopilot = SpecialAutopilot(seed=11)

    assert autopilot._absorb_results(db, [account.id]) == 0
    assert autopilot._queue_repairs(db, orchestrator, account_ids=[account.id], pending_room=10) == 0
    weights, examples = autopilot._memory_feedback(db)
    assert weights == {}
    assert examples == []


def test_ml_training_pulls_library_positives_and_near_miss_negatives():
    from backend.ml.service import MLRankingService

    db = make_db()
    account = _add_account(db)
    _completed_result(
        db, account, "group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
        1.7, 1.4, 0.35, 0.15, True, {"neutralization": "SECTOR"},
    )
    _completed_result(
        db, account, "rank(ts_rank(returns, 60))",
        1.4, 1.1, 0.85, 0.2, False, {"neutralization": "SUBINDUSTRY"},
    )
    SpecialAutopilot(seed=1)._absorb_results(db, [account.id])

    examples = MLRankingService(db)._self_improving_examples()
    labels = {example.label for example in examples}
    assert 1 in labels  # confirmed library winner -> positive
    assert 0 in labels  # near-miss -> hard negative
    assert any("group_neutralize" in example.expression for example in examples)


def test_selfimprove_api_surfaces_memory_and_library():
    from backend.routes.selfimprove import (
        selfimprove_library,
        selfimprove_memory,
        selfimprove_near_misses,
        selfimprove_stats,
    )

    db = make_db()
    account = _add_account(db)
    autopilot = SpecialAutopilot(seed=1)
    _completed_result(
        db, account, "group_neutralize(rank(ts_corr(close, volume, 20)), sector)",
        1.7, 1.4, 0.35, 0.15, True, {"neutralization": "SECTOR"},
    )
    _completed_result(
        db, account, "rank(ts_rank(returns, 60))",
        1.4, 1.1, 0.85, 0.2, False, {"neutralization": "SUBINDUSTRY"},
    )
    autopilot._absorb_results(db, [account.id])

    stats = selfimprove_stats(db=db)
    assert stats["attempts"] == 2
    assert stats["library_size"] == 1

    assert selfimprove_memory(outcome=None, limit=50, db=db)["count"] == 2
    assert selfimprove_memory(outcome="win", limit=50, db=db)["count"] == 1
    assert selfimprove_near_misses(limit=20, db=db)["count"] >= 1
    assert selfimprove_library(focus=None, limit=50, db=db)["count"] == 1
