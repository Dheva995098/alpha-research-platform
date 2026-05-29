"""Tests for Phase 3 simulation orchestration."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import timedelta

from backend.automation import special_runner
from backend.automation.special_runner import SpecialAutopilot
from backend.core.brain_api import BRAINRateLimitError
from backend.main import app
from backend.models import Account, AlphaRegistry, Base, Result, Simulation
from backend.orchestration.service import SimulationOrchestrator
from backend.routes.orchestration import QueueRequest, enqueue_simulations
from backend.security import encrypt_credential
from backend.utils.time import utc_now
from backend.workers.orchestration_worker import OrchestrationWorker


class FakeBrainGateway:
    """Deterministic gateway for orchestration tests."""

    def __init__(self):
        self.submitted = []

    def submit_expression(self, account, password, expression, universe="default", settings=None):
        simulation_id = f"fake-{len(self.submitted) + 1}"
        self.submitted.append(
            {
                "account_id": account.id,
                "password": password,
                "expression": expression,
                "universe": universe,
                "settings": settings,
                "simulation_id": simulation_id,
            }
        )
        return simulation_id

    def get_status(self, account, password, brain_simulation_id):
        return {"status": "completed", "progress": 100}

    def get_results(self, account, password, brain_simulation_id):
        return {
            "alpha_id": f"alpha-{brain_simulation_id}",
            "sharpe": 1.3,
            "fitness": 1.1,
            "turnover": 0.45,
            "self_correlation": 0.12,
            "all_checks_passed": True,
        }


class UnknownVariableGateway(FakeBrainGateway):
    """Gateway that mirrors BRAIN rejecting a missing data field."""

    def submit_expression(self, account, password, expression, universe="default", settings=None):
        raise RuntimeError('Attempted to use unknown variable "scl12_volume".')


class RateLimitedGateway(FakeBrainGateway):
    """Gateway that simulates a transient BRAIN rate limit."""

    def __init__(self):
        super().__init__()
        self.status_calls = 0

    def get_status(self, account, password, brain_simulation_id):
        self.status_calls += 1
        raise BRAINRateLimitError("BRAIN API rate limit exceeded; retry polling later", retry_after=60)


class DuplicateAlphaGateway(FakeBrainGateway):
    """Gateway that returns the same BRAIN alpha id for multiple simulations."""

    def get_results(self, account, password, brain_simulation_id):
        return {
            "alpha_id": "same-live-alpha",
            "sharpe": 1.3,
            "fitness": 1.1,
            "turnover": 0.45,
            "self_correlation": 0.12,
            "all_checks_passed": True,
        }


class StuckGateway(FakeBrainGateway):
    """Gateway that leaves simulations running."""

    def get_status(self, account, password, brain_simulation_id):
        return {"status": "running", "progress": 0}


class CountingGateway(StuckGateway):
    """Gateway that counts poll requests."""

    def __init__(self):
        super().__init__()
        self.status_calls = 0

    def get_status(self, account, password, brain_simulation_id):
        self.status_calls += 1
        return super().get_status(account, password, brain_simulation_id)


def make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory()


def add_account(
    db,
    email="test@example.com",
    daily_quota=3,
    worker_enabled=True,
    max_running=6,
    max_pending=15,
):
    account = Account(
        brain_email=email,
        brain_password_encrypted=encrypt_credential("secret"),
        daily_quota=daily_quota,
        submissions_today=0,
        is_active=True,
        worker_enabled=worker_enabled,
        max_running=max_running,
        max_pending=max_pending,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def test_enqueue_expressions_dedupes_and_validates():
    db = make_db()
    account = add_account(db)
    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())

    result = orchestrator.enqueue_expressions(
        db,
        expressions=["rank(close)", " rank( close ) ", "rank(close); DROP"],
        account_ids=[account.id],
    )

    assert result.ok
    assert len(result.simulations) == 1
    assert result.metadata["duplicate_count"] == 1
    assert len(result.metadata["skipped"]) == 1
    assert result.simulations[0].status == "pending"


def test_submit_next_dry_run_updates_status_and_quota():
    db = make_db()
    account = add_account(db, daily_quota=1)
    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())
    orchestrator.enqueue_expressions(
        db,
        ["rank(close)"],
        account_ids=[account.id],
        settings={"decay": 6, "truncation": 0.01},
    )

    result = orchestrator.submit_next(db, dry_run=True)

    assert result.ok
    assert result.simulations[0].status == "completed"
    assert result.simulations[0].brain_simulation_id == "dry-run-1"
    db.refresh(account)
    assert account.submissions_today == 0
    stored_result = db.query(Result).filter(Result.simulation_id == result.simulations[0].id).first()
    assert stored_result is not None
    assert stored_result.raw_metrics["dry_run"] is True
    assert stored_result.raw_metrics["source"] == "dry_run"
    assert stored_result.raw_metrics["settings"]["delay"] == 1
    assert stored_result.raw_metrics["settings"]["decay"] == 6


def test_enqueue_applies_adaptive_settings_for_manual_group_neutralization():
    db = make_db()
    account = add_account(db, daily_quota=1)
    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())

    result = orchestrator.enqueue_expressions(
        db,
        ["group_neutralize(rank(close), industry)"],
        account_ids=[account.id],
        settings={"neutralization": "SUBINDUSTRY", "decay": 10},
    )

    assert result.simulations[0].settings["neutralization"] == "NONE"
    assert result.simulations[0].settings["decay"] == 10


def test_submit_next_uses_gateway_for_live_path():
    db = make_db()
    account = add_account(db)
    gateway = FakeBrainGateway()
    orchestrator = SimulationOrchestrator(gateway=gateway)
    orchestrator.enqueue_expressions(db, ["rank(close)"], account_ids=[account.id])

    result = orchestrator.submit_next(db, universe="usa", dry_run=False)

    assert result.ok
    assert result.simulations[0].status == "running"
    assert result.simulations[0].brain_simulation_id == "fake-1"
    assert gateway.submitted[0]["universe"] == "usa"


def test_orchestration_normalizes_legacy_operator_aliases_before_live_submit():
    db = make_db()
    account = add_account(db)
    gateway = FakeBrainGateway()
    orchestrator = SimulationOrchestrator(gateway=gateway)
    orchestrator.enqueue_expressions(db, ["rank(0 - ts_std(returns, 10))"], account_ids=[account.id])

    result = orchestrator.submit_next(db, universe="usa", dry_run=False)

    assert result.ok
    assert result.simulations[0].expression == "rank(0 - ts_std_dev(returns, 10))"
    assert gateway.submitted[0]["expression"] == "rank(0 - ts_std_dev(returns, 10))"


def test_schema_rejects_live_invalid_scl12_volume():
    from backend.core.data_fields import BRAINDataFields

    schema = BRAINDataFields(custom_fields={"scl12_volume"})

    assert not schema.validate_field("scl12_volume")
    valid, message = schema.validate_expression_basic("rank(scl12_volume)")
    assert valid is False
    assert "scl12_volume" in message


def test_unknown_variable_failure_disables_matching_pending_rows():
    db = make_db()
    account = add_account(db)
    db.add_all(
        [
            Simulation(account_id=account.id, expression="rank(scl12_volume)", status="pending"),
            Simulation(account_id=account.id, expression="zscore(scl12_volume)", status="pending"),
            Simulation(account_id=account.id, expression="rank(close)", status="pending"),
        ]
    )
    db.commit()

    result = SimulationOrchestrator(gateway=UnknownVariableGateway()).submit_next(db, dry_run=False)

    assert not result.ok
    assert result.metadata["invalid_field"] == "scl12_volume"
    assert result.metadata["disabled_pending_count"] == 1
    rows = db.query(Simulation).order_by(Simulation.id.asc()).all()
    assert [row.status for row in rows] == ["failed", "failed", "pending"]
    db.refresh(account)
    assert account.cooldown_until is None
    assert account.last_worker_error is None


def test_submit_result_live_creates_fresh_live_simulation_from_result():
    db = make_db()
    account = add_account(db)
    gateway = FakeBrainGateway()
    orchestrator = SimulationOrchestrator(gateway=gateway)
    orchestrator.enqueue_expressions(db, ["rank(close)"], account_ids=[account.id])
    dry_result = orchestrator.submit_next(db, dry_run=True)
    source_result = db.query(Result).filter(Result.simulation_id == dry_result.simulations[0].id).first()

    result = orchestrator.submit_result_live(db, source_result.id, universe="usa")

    assert result.ok
    assert result.simulations[0].status == "running"
    assert result.simulations[0].expression == source_result.expression
    assert result.simulations[0].brain_simulation_id == "fake-1"
    assert result.simulations[0].id != dry_result.simulations[0].id
    assert gateway.submitted[0]["expression"] == "rank(close)"
    assert gateway.submitted[0]["universe"] == "usa"
    assert gateway.submitted[0]["settings"]["neutralization"] == "SUBINDUSTRY"
    db.refresh(account)
    assert account.submissions_today == 1


def test_submit_next_skips_quota_exhausted_accounts():
    db = make_db()
    exhausted = add_account(db, email="exhausted@example.com", daily_quota=1)
    available = add_account(db, email="available@example.com", daily_quota=1)
    exhausted.submissions_today = 1
    db.add_all(
        [
            Simulation(account_id=exhausted.id, expression="rank(close)", status="pending"),
            Simulation(account_id=available.id, expression="rank(volume)", status="pending"),
        ]
    )
    db.commit()

    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())
    result = orchestrator.submit_next(db, dry_run=True)

    assert not result.ok
    assert not result.simulations
    assert result.metadata["skipped"]


def test_enqueue_registry_blocks_duplicate_across_accounts():
    db = make_db()
    first = add_account(db, email="first@example.com")
    second = add_account(db, email="second@example.com")
    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())

    first_result = orchestrator.enqueue_expressions(db, ["rank(close)"], account_ids=[first.id])
    second_result = orchestrator.enqueue_expressions(db, [" rank( close ) "], account_ids=[second.id])

    assert first_result.ok
    assert len(first_result.simulations) == 1
    assert len(second_result.simulations) == 0
    assert second_result.metadata["duplicate_count"] == 1
    assert db.query(Simulation).count() == 1
    assert db.query(AlphaRegistry).count() == 1


def test_submit_next_respects_account_running_cap():
    db = make_db()
    capped = add_account(db, email="capped@example.com", max_running=1)
    open_account = add_account(db, email="open@example.com", max_running=1)
    db.add_all(
        [
            Simulation(account_id=capped.id, expression="rank(close)", status="running", brain_simulation_id="fake-1"),
            Simulation(account_id=capped.id, expression="rank(volume)", status="pending"),
            Simulation(account_id=open_account.id, expression="rank(open)", status="pending"),
        ]
    )
    db.commit()

    result = SimulationOrchestrator(gateway=FakeBrainGateway()).submit_next(db, dry_run=True)

    assert not result.ok
    assert not result.simulations
    assert any("already has 1 running" in item for item in result.metadata["skipped"])


def test_special_autopilot_uses_only_worker_enabled_accounts():
    db = make_db()
    disabled = add_account(db, email="disabled@example.com", daily_quota=10, worker_enabled=False)
    enabled = add_account(db, email="enabled@example.com", daily_quota=10, worker_enabled=True)
    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())

    metadata = SpecialAutopilot(seed=11).tick(
        db,
        orchestrator,
        dry_run=False,
        universe="usa",
        batch_size=4,
        target_running=4,
        max_running=4,
    )

    assert metadata["account_ids"] == []
    assert metadata["batch"] is None
    account_ids = {row.account_id for row in db.query(Simulation).all()}
    assert account_ids == set()
    assert disabled.id not in account_ids


def test_rebalance_pending_moves_rows_from_cooling_account():
    db = make_db()
    cooling = add_account(db, email="cooling@example.com")
    healthy = add_account(db, email="healthy@example.com")
    cooling.cooldown_until = utc_now() + timedelta(minutes=10)
    simulation = Simulation(account_id=cooling.id, expression="rank(close)", status="pending")
    db.add(simulation)
    db.commit()

    result = SimulationOrchestrator(gateway=FakeBrainGateway()).rebalance_pending(
        db,
        account_ids=[cooling.id, healthy.id],
    )

    db.refresh(simulation)
    assert result.metadata["rebalanced_count"] == 0
    assert simulation.account_id == cooling.id


def test_special_autopilot_refills_empty_parallel_lane_when_global_target_is_met():
    db = make_db()
    full_lane = add_account(db, email="full@example.com", daily_quota=10, max_running=1)
    empty_lane = add_account(db, email="empty@example.com", daily_quota=10, max_running=3)
    db.add(
        Simulation(
            account_id=full_lane.id,
            expression="rank(close)",
            status="running",
            brain_simulation_id="already-running",
        )
    )
    db.commit()

    metadata = SpecialAutopilot(seed=13).tick(
        db,
        SimulationOrchestrator(gateway=StuckGateway()),
        dry_run=False,
        universe="usa",
        batch_size=2,
        target_running=1,
        max_running=4,
        refill_pending_below=10,
        max_pending=10,
    )

    empty_running = (
        db.query(Simulation)
        .filter(Simulation.account_id == empty_lane.id)
        .filter(Simulation.status == "running")
        .count()
    )
    assert metadata["batch"] is None
    assert empty_running == 0


def test_special_autopilot_can_skip_polling_on_submit_tick():
    db = make_db()
    account = add_account(db)
    db.add(
        Simulation(
            account_id=account.id,
            expression="rank(close)",
            status="running",
            brain_simulation_id="running-1",
        )
    )
    db.commit()
    gateway = CountingGateway()

    metadata = SpecialAutopilot(seed=11).tick(
        db,
        SimulationOrchestrator(gateway=gateway),
        dry_run=False,
        universe="usa",
        batch_size=1,
        target_running=1,
        max_running=1,
        refill_pending_below=0,
        max_pending=1,
        poll_first=False,
    )

    assert gateway.status_calls == 0
    assert metadata["poll_first"] is False
    assert "Skipped" in metadata["poll_message"]


def test_special_autopilot_limits_submit_burst_per_tick():
    db = make_db()
    add_account(db, daily_quota=10)
    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())

    metadata = SpecialAutopilot(seed=11).tick(
        db,
        orchestrator,
        dry_run=False,
        universe="usa",
        batch_size=5,
        target_running=5,
        max_running=5,
        poll_first=False,
        submit_batch_limit=2,
    )

    assert metadata["batch"]["queued_count"] == 5
    assert len(metadata["submitted_ids"]) == 2
    assert metadata["submit_batch_limit"] == 2
    assert db.query(Simulation).filter(Simulation.status == "running").count() == 2
    assert db.query(Simulation).filter(Simulation.status == "pending").count() == 3


def test_poll_running_persists_completed_result():
    db = make_db()
    account = add_account(db)
    simulation = Simulation(
        account_id=account.id,
        expression="rank(close)",
        status="running",
        progress=10.0,
        brain_simulation_id="fake-1",
    )
    db.add(simulation)
    db.commit()

    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())
    result = orchestrator.poll_running(db)

    assert result.ok
    assert result.simulations[0].status == "completed"
    stored_result = db.query(Result).filter(Result.simulation_id == simulation.id).first()
    assert stored_result is not None
    assert stored_result.brain_alpha_id == "alpha-fake-1"
    assert stored_result.all_checks_passed is True
    assert stored_result.final_score > 0


def test_poll_running_completes_dry_run_without_gateway_auth():
    db = make_db()
    account = add_account(db)
    simulation = Simulation(
        account_id=account.id,
        expression="rank(close)",
        status="running",
        progress=0.0,
        brain_simulation_id="dry-run-99",
    )
    db.add(simulation)
    db.commit()

    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())
    result = orchestrator.poll_running(db)

    assert result.ok
    assert result.simulations[0].status == "completed"
    stored_result = db.query(Result).filter(Result.simulation_id == simulation.id).first()
    assert stored_result.raw_metrics["dry_run"] is True


def test_poll_running_can_limit_one_row_per_account():
    db = make_db()
    first = add_account(db, email="first@example.com")
    second = add_account(db, email="second@example.com")
    db.add_all(
        [
            Simulation(account_id=first.id, expression="rank(close)", status="running", brain_simulation_id="fake-1"),
            Simulation(account_id=first.id, expression="rank(open)", status="running", brain_simulation_id="fake-2"),
            Simulation(account_id=second.id, expression="rank(volume)", status="running", brain_simulation_id="fake-3"),
            Simulation(account_id=second.id, expression="rank(vwap)", status="running", brain_simulation_id="fake-4"),
        ]
    )
    db.commit()

    result = SimulationOrchestrator(gateway=FakeBrainGateway()).poll_running(
        db,
        limit=10,
        per_account_limit=1,
    )

    assert result.ok
    assert len(result.simulations) == 1
    assert {simulation.account_id for simulation in result.simulations} == {first.id}
    assert db.query(Simulation).filter(Simulation.status == "completed").count() == 1
    assert db.query(Simulation).filter(Simulation.status == "running").count() == 3


def test_worker_staggers_polling_between_accounts(monkeypatch):
    db = make_db()
    first = add_account(db, email="first@example.com")
    second = add_account(db, email="second@example.com")
    db.add_all(
        [
            Simulation(account_id=first.id, expression="rank(close)", status="running", brain_simulation_id="fake-1"),
            Simulation(account_id=second.id, expression="rank(volume)", status="running", brain_simulation_id="fake-2"),
        ]
    )
    db.commit()

    monkeypatch.setattr("backend.workers.orchestration_worker.settings.brain_poll_account_gap_seconds", 30)
    worker = OrchestrationWorker(orchestrator=SimulationOrchestrator(gateway=FakeBrainGateway()))
    worker.state.poll_interval_seconds = 100

    assert worker._poll_account_due(db, 1000.0) == first.id
    worker._mark_polled_account(db, first.id, 1000.0)
    assert worker._poll_account_due(db, 1001.0) is None
    assert worker._poll_account_due(db, 1029.0) is None
    assert worker._poll_account_due(db, 1030.0) == second.id
    worker._mark_polled_account(db, second.id, 1030.0)
    assert worker._poll_account_due(db, 1099.0) is None
    assert worker._poll_account_due(db, 1100.0) == first.id


def test_poll_running_keeps_live_rows_running_on_rate_limit():
    db = make_db()
    account = add_account(db)
    simulation = Simulation(
        account_id=account.id,
        expression="rank(close)",
        status="running",
        progress=10.0,
        brain_simulation_id="fake-1",
    )
    db.add(simulation)
    db.commit()

    gateway = RateLimitedGateway()
    result = SimulationOrchestrator(gateway=gateway).poll_running(db)

    assert not result.ok
    assert result.metadata["rate_limited"] is True
    assert result.simulations[0].status == "running"
    assert "rate limit" in result.simulations[0].error_message.lower()
    assert result.simulations[0].progress == 10.0
    assert gateway.status_calls == 1


def test_poll_running_stops_batch_after_rate_limit():
    db = make_db()
    account = add_account(db)
    db.add_all(
        [
            Simulation(account_id=account.id, expression="rank(close)", status="running", brain_simulation_id="fake-1"),
            Simulation(account_id=account.id, expression="rank(volume)", status="running", brain_simulation_id="fake-2"),
        ]
    )
    db.commit()
    gateway = RateLimitedGateway()

    result = SimulationOrchestrator(gateway=gateway).poll_running(db, limit=2)

    assert not result.ok
    assert gateway.status_calls == 1
    assert len(result.simulations) == 1


def test_poll_running_saves_duplicate_alpha_ids_without_crashing():
    db = make_db()
    account = add_account(db)
    db.add_all(
        [
            Simulation(account_id=account.id, expression="rank(close)", status="running", brain_simulation_id="fake-1"),
            Simulation(account_id=account.id, expression="rank(volume)", status="running", brain_simulation_id="fake-2"),
        ]
    )
    db.commit()

    result = SimulationOrchestrator(gateway=DuplicateAlphaGateway()).poll_running(db, limit=2)

    stored = db.query(Result).order_by(Result.id.asc()).all()
    assert result.ok
    assert len(stored) == 2
    assert stored[0].brain_alpha_id == "same-live-alpha"
    assert stored[1].brain_alpha_id.startswith("same-live-alpha#sim-")
    assert stored[1].raw_metrics["brain_alpha_id_original"] == "same-live-alpha"


def test_special_autopilot_queues_and_submits_to_running_cap():
    db = make_db()
    add_account(db, daily_quota=10)
    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())
    autopilot = SpecialAutopilot(seed=11)

    metadata = autopilot.tick(
        db,
        orchestrator,
        dry_run=False,
        universe="usa",
        batch_size=5,
        target_running=5,
        max_running=6,
    )

    assert metadata["batch"]["queued_count"] == 5
    assert len(metadata["submitted_ids"]) == 5
    assert db.query(Simulation).filter(Simulation.status == "running").count() == 5
    assert db.query(Simulation).filter(Simulation.status == "pending").count() == 0


def test_special_autopilot_continues_when_openai_advice_fails(monkeypatch):
    class FailingAdvisor:
        def advise(self, *args, **kwargs):
            raise RuntimeError("advisor offline")

    monkeypatch.setattr(special_runner, "OpenAIAlphaAdvisor", FailingAdvisor)
    db = make_db()
    add_account(db, daily_quota=10)
    orchestrator = SimulationOrchestrator(gateway=FakeBrainGateway())
    autopilot = SpecialAutopilot(seed=11)

    metadata = autopilot.tick(
        db,
        orchestrator,
        dry_run=False,
        universe="usa",
        batch_size=5,
        target_running=5,
        max_running=6,
        openai_assist=True,
    )

    assert metadata["batch"]["queued_count"] == 5
    assert len(metadata["submitted_ids"]) == 5
    assert db.query(Simulation).filter(Simulation.status == "running").count() == 5


def test_special_autopilot_respects_pending_cap_and_reaps_stale_running():
    db = make_db()
    account = add_account(db, daily_quota=10)
    stale = Simulation(
        account_id=account.id,
        expression="rank(close)",
        status="running",
        brain_simulation_id="old-sim",
        submitted_at=utc_now() - timedelta(minutes=300),
    )
    pending = [
        Simulation(account_id=account.id, expression=f"rank(volume + {idx})", status="pending")
        for idx in range(20)
    ]
    db.add(stale)
    db.add_all(pending)
    db.commit()

    orchestrator = SimulationOrchestrator(gateway=StuckGateway())
    metadata = SpecialAutopilot(seed=11).tick(
        db,
        orchestrator,
        dry_run=False,
        universe="usa",
        batch_size=5,
        target_running=5,
        max_running=6,
        refill_pending_below=10,
        max_pending=15,
        stale_running_minutes=240,
    )

    db.refresh(stale)
    assert metadata["stale_reaped_count"] == 1
    assert metadata["batch"] is None
    assert metadata["refill_skipped"] is True
    assert stale.status == "failed"
    assert "Stale running timeout" in stale.error_message


def test_clear_pending_removes_only_pending_rows():
    db = make_db()
    account = add_account(db)
    db.add_all(
        [
            Simulation(account_id=account.id, expression="rank(close)", status="pending"),
            Simulation(account_id=account.id, expression="rank(volume)", status="running"),
            Simulation(account_id=account.id, expression="rank(open)", status="completed"),
        ]
    )
    db.commit()

    result = SimulationOrchestrator().clear_pending(db)

    statuses = [row.status for row in db.query(Simulation).order_by(Simulation.id.asc()).all()]
    assert result.metadata["cleared_count"] == 1
    assert statuses == ["running", "completed"]


def test_clear_terminal_removes_failed_rows():
    db = make_db()
    account = add_account(db)
    db.add(Simulation(account_id=account.id, expression="rank(close)", status="failed"))
    db.commit()

    result = SimulationOrchestrator().clear_terminal(db)

    assert result.metadata["cleared_count"] == 1
    assert db.query(Simulation).count() == 0


def test_orchestration_routes_are_registered():
    route_paths = {route.path for route in app.routes}

    assert "/api/orchestration/queue" in route_paths
    assert "/api/orchestration/submit-next" in route_paths
    assert "/api/orchestration/results/{result_id}/live-submit" in route_paths
    assert "/api/orchestration/poll" in route_paths
    assert "/api/orchestration/worker/status" in route_paths
    assert "/api/orchestration/queue/clear-pending" in route_paths


def test_enqueue_route_handler_accepts_db_dependency_directly():
    db = make_db()
    add_account(db)

    response = enqueue_simulations(
        QueueRequest(expressions=["rank(close)"]),
        db=db,
    )

    assert response.queued_count == 1
    assert response.simulations[0].status == "pending"
