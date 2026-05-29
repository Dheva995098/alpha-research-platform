"""Tests for Phase 2 alpha generation."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.data_fields import BRAINDataFields
from backend.core.expression_normalizer import normalize_brain_expression
from backend.core.field_intelligence import top_field_records, upsert_field_records
from backend.generation.dedup import ExpressionDeduplicator, expression_signature
from backend.generation.expression_generator import RuleBasedAlphaGenerator
from backend.generation.genetic import GeneticAlphaRefiner
from backend.main import app
from backend.models import Base
from backend.routes.generation import (
    DeduplicateRequest,
    GenerateAlphaRequest,
    deduplicate_alphas,
    generate_alphas,
    list_generation_datasets,
    list_field_intelligence,
    list_generation_fields,
)


def make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory()


def test_rule_based_generator_produces_valid_unique_candidates():
    schema = BRAINDataFields()
    generator = RuleBasedAlphaGenerator(schema=schema, seed=7)

    candidates = generator.generate(count=10)

    assert len(candidates) == 10
    signatures = {expression_signature(candidate.expression) for candidate in candidates}
    assert len(signatures) == len(candidates)

    for candidate in candidates:
        valid, message = schema.validate_expression_basic(candidate.expression)
        assert valid, message
        assert "ts_std(" not in candidate.expression
        assert candidate.score > 0
        assert candidate.strategy
        assert candidate.source_fields


def test_schema_rejects_unknown_function_operators():
    schema = BRAINDataFields()

    valid, message = schema.validate_expression_basic("rank(unknown_op(close, 20))")

    assert valid is False
    assert "Unknown operator" in message


def test_expression_normalizer_removes_winsorize_positional_std():
    expression = "group_neutralize(winsorize(rank(ts_zscore(ts_backfill(news_pct_1min, 120), 60)), 4), industry)"

    normalized = normalize_brain_expression(expression)

    assert normalized == "group_neutralize(winsorize(rank(ts_zscore(ts_backfill(news_pct_1min, 120), 60))), industry)"


def test_rule_based_generator_respects_focus_and_fields():
    schema = BRAINDataFields()
    generator = RuleBasedAlphaGenerator(schema=schema, seed=3)

    candidates = generator.generate(
        count=3,
        focus="liquidity",
        fields=["volume", "adv20"],
        neutralize=False,
    )

    assert len(candidates) == 3
    assert {candidate.strategy for candidate in candidates} == {"liquidity"}
    for candidate in candidates:
        assert set(candidate.source_fields).issubset({"volume", "adv20"})


def test_rule_based_generator_emits_fundamental_quality_templates():
    schema = BRAINDataFields()
    generator = RuleBasedAlphaGenerator(schema=schema, seed=5)

    candidates = generator.generate(count=4, focus="quality", neutralize=True)

    assert len(candidates) == 4
    assert any({"ebit", "cashflow_op", "capex"} & set(candidate.source_fields) for candidate in candidates)
    assert any("abs" in candidate.operators for candidate in candidates)


def test_rule_based_generator_emits_doc_learned_advanced_templates():
    schema = BRAINDataFields()
    generator = RuleBasedAlphaGenerator(schema=schema, seed=17)

    candidates = []
    for focus in ("analyst", "sentiment", "options", "decorrelation"):
        candidates.extend(generator.generate(count=2, focus=focus, neutralize=False))

    assert len(candidates) >= 8
    assert any("trade_when" in candidate.operators for candidate in candidates)
    assert any("ts_backfill" in candidate.operators for candidate in candidates)
    assert any(candidate.strategy == "decorrelation" for candidate in candidates)


def test_schema_exposes_dataset_families_and_field_metadata():
    schema = BRAINDataFields()

    assert "pcr_oi_720" in schema.fields
    assert "option8" in schema.field_info("pcr_oi_720")["datasets"]
    option_fields = schema.fields_for_dataset_ids(["option8"])

    assert "implied_volatility_call_720" in option_fields
    assert "pcr_oi_720" in option_fields


def test_rule_based_generator_can_target_dataset_family():
    schema = BRAINDataFields()
    generator = RuleBasedAlphaGenerator(schema=schema, seed=19)

    candidates = generator.generate(count=3, dataset_ids=["option8"], neutralize=False)

    assert len(candidates) == 3
    assert all("option8" in candidate.dataset_ids for candidate in candidates)
    assert any("pcr_oi_720" in candidate.source_fields for candidate in candidates)


def test_field_intelligence_scores_and_persists_live_fields():
    db = make_db()

    stats = upsert_field_records(
        db,
        [
            {
                "id": "option_live_skew",
                "dataset": {"id": "option8"},
                "type": "MATRIX",
                "coverage": 0.92,
                "alphaCount": 38,
                "userCount": 11,
                "valueScore": 7.5,
                "description": "Synthetic option skew field for testing",
            }
        ],
        dataset_id="option8",
        universe="TOPSP500",
    )
    records = top_field_records(db, dataset_ids=["option8"], prefix="option_live", limit=5)

    assert stats["imported"] == 1
    assert records[0].name == "option_live_skew"
    assert records[0].field_score > 0.7


def test_generation_route_uses_persisted_field_intelligence():
    db = make_db()
    upsert_field_records(
        db,
        [
            {
                "id": "mdl77_custom_momentum",
                "dataset": {"id": "model77"},
                "type": "MATRIX",
                "coverage": 0.88,
                "alphaCount": 22,
                "userCount": 7,
            }
        ],
        dataset_id="model77",
    )

    response = generate_alphas(
        GenerateAlphaRequest(
            count=2,
            focus="model_risk",
            dataset_ids=["model77"],
            fields=["mdl77_custom_momentum"],
            neutralize=False,
            seed=31,
        ),
        db=db,
    )

    assert response.generated_count == 2
    assert all("mdl77_custom_momentum" in candidate.source_fields for candidate in response.candidates)


def test_deduplicator_normalizes_whitespace():
    result = ExpressionDeduplicator().dedupe(
        [
            "rank(close)",
            " rank( close ) ",
            "rank(volume)",
        ]
    )

    assert result.unique == ["rank(close)", "rank(volume)"]
    assert len(result.duplicates) == 1
    assert result.duplicates[0].duplicate_of == "rank(close)"


def test_genetic_refiner_produces_valid_refinements():
    schema = BRAINDataFields()
    refiner = GeneticAlphaRefiner(schema=schema, seed=11)

    candidates = refiner.refine(
        [
            "rank(ts_rank(close, 20))",
            "rank(ts_corr(close, volume, 20))",
        ],
        count=5,
    )

    assert len(candidates) > 0
    for candidate in candidates:
        valid, message = schema.validate_expression_basic(candidate.expression)
        assert valid, message
        assert "ts_std(" not in candidate.expression
        assert candidate.strategy in {"mutation", "crossover"}


def test_generation_routes_are_registered():
    route_paths = {route.path for route in app.routes}

    assert "/api/generation/generate" in route_paths
    assert "/api/generation/refine" in route_paths
    assert "/api/generation/deduplicate" in route_paths
    assert "/api/generation/datasets" in route_paths
    assert "/api/generation/fields" in route_paths
    assert "/api/generation/field-intelligence" in route_paths


def test_generation_route_handler_returns_candidates():
    response = generate_alphas(
        GenerateAlphaRequest(count=5, focus="momentum", seed=13)
    )

    assert response.requested_count == 5
    assert response.generated_count == 5
    assert len(response.candidates) == 5
    assert all(candidate.expression for candidate in response.candidates)


def test_generation_route_handler_accepts_dataset_ids():
    response = generate_alphas(
        GenerateAlphaRequest(count=2, dataset_ids=["model77"], seed=23, neutralize=False)
    )

    assert response.generated_count == 2
    assert all(candidate.dataset_ids for candidate in response.candidates)
    assert response.settings_overrides["universe"] == "TOP3000"
    assert response.settings_overrides["neutralization"] == "SUBINDUSTRY"


def test_generation_route_handler_randomizes_dataset_focus_and_settings():
    response = generate_alphas(
        GenerateAlphaRequest(count=5, focus="random", seed=101, randomize=True)
    )

    assert response.generated_count == 5
    assert response.metadata["mode"] == "random"
    assert response.metadata["dataset_id"]
    assert response.metadata["focus"]
    assert response.metadata["seed"] != 101
    assert "decay" in response.settings_overrides
    assert all(candidate.expression for candidate in response.candidates)


def test_generation_datasets_route_returns_profiles():
    response = list_generation_datasets()

    dataset_ids = {dataset["id"] for dataset in response["datasets"]}
    assert "pv1" in dataset_ids
    assert "option8" in dataset_ids
    assert "model_risk" in response["categories"]


def test_generation_fields_route_filters_by_dataset():
    response = list_generation_fields(dataset_id="option8", prefix="pcr", limit=10)

    names = {field["name"] for field in response["fields"]}
    assert "pcr_oi_270" in names
    assert "pcr_oi_720" in names


def test_field_intelligence_route_returns_ranked_records():
    db = make_db()
    upsert_field_records(
        db,
        [
            {
                "id": "est_live_revision",
                "dataset": {"id": "analyst7"},
                "type": "MATRIX",
                "coverage": 0.76,
                "alphaCount": 18,
                "userCount": 5,
            }
        ],
        dataset_id="analyst7",
    )

    response = list_field_intelligence(dataset_id="analyst7", prefix="est_live", db=db)

    assert response["fields"][0]["name"] == "est_live_revision"
    assert response["fields"][0]["field_score"] > 0


def test_deduplicate_route_handler_reports_duplicates():
    response = deduplicate_alphas(
        DeduplicateRequest(expressions=["rank(close)", " rank( close ) "])
    )

    assert response.unique_count == 1
    assert response.duplicate_count == 1
