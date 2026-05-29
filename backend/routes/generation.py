"""API routes for alpha expression generation."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.data_fields import get_data_fields
from backend.core.dataset_catalog import category_names, dataset_settings_overrides, list_dataset_profiles
from backend.core.field_intelligence import (
    field_records_summary,
    record_to_dict,
    schema_with_persisted_fields,
    top_field_names,
    top_field_records,
)
from backend.generation.dedup import ExpressionDeduplicator
from backend.generation.expression_generator import (
    RuleBasedAlphaGenerator,
    available_strategies,
)
from backend.generation.genetic import GeneticAlphaRefiner
from backend.generation.openai_advisor import apply_openai_advice
from backend.models import get_db

router = APIRouter()


class AlphaCandidateResponse(BaseModel):
    """Generated alpha candidate response."""

    expression: str
    strategy: str
    source_fields: List[str]
    dataset_ids: List[str] = Field(default_factory=list)
    operators: List[str]
    rationale: str
    score: float


class GenerateAlphaRequest(BaseModel):
    """Request for rule-based alpha generation."""

    count: int = Field(default=20, ge=1, le=200)
    focus: Optional[str] = Field(default=None, description="Optional strategy focus")
    fields: Optional[List[str]] = Field(default=None, description="Optional field allow-list")
    dataset_ids: Optional[List[str]] = Field(default=None, description="Optional BRAIN dataset ids")
    neutralize: bool = True
    seed: Optional[int] = None
    randomize: bool = Field(default=False, description="Randomly pick dataset, focus, seed, and settings hints")
    use_openai: bool = Field(default=False, description="Use OpenAI to critique and rerank candidates")
    include_refinements: bool = False
    existing_expressions: List[str] = Field(default_factory=list)


class GenerateAlphaResponse(BaseModel):
    """Generated candidates plus summary counts."""

    requested_count: int
    generated_count: int
    candidates: List[AlphaCandidateResponse]
    settings_overrides: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class RefineAlphaRequest(BaseModel):
    """Request for genetic-style refinement."""

    expressions: List[str] = Field(min_length=1)
    count: int = Field(default=20, ge=1, le=200)
    seed: Optional[int] = None
    existing_expressions: List[str] = Field(default_factory=list)


class DuplicateExpressionResponse(BaseModel):
    """Duplicate expression report."""

    expression: str
    duplicate_of: str
    signature: str


class DeduplicateRequest(BaseModel):
    """Request to normalize and deduplicate expressions."""

    expressions: List[str] = Field(min_length=1)
    existing_expressions: List[str] = Field(default_factory=list)


class DeduplicateResponse(BaseModel):
    """Deduplication response."""

    total_count: int
    unique_count: int
    duplicate_count: int
    unique: List[str]
    duplicates: List[DuplicateExpressionResponse]


@router.get("/strategies", tags=["generation"])
def list_generation_strategies() -> dict:
    """List supported rule-based generation strategies."""
    return {"strategies": available_strategies()}


@router.get("/datasets", tags=["generation"])
def list_generation_datasets(category: Optional[str] = None, db: Session = Depends(get_db)) -> dict:
    """List local dataset-family profiles and known fields."""
    db_session = _db_or_none(db)
    schema = schema_with_persisted_fields(db_session) if db_session else get_data_fields()
    profiles = []
    for profile in list_dataset_profiles(category):
        known_fields = [field for field in profile.example_fields if schema.validate_field(field)]
        profiles.append(
            {
                **profile.as_dict(),
                "known_field_count": len(known_fields),
                "known_fields": known_fields[:50],
            }
        )
    return {
        "categories": list(category_names()),
        "datasets": profiles,
        "field_summary": field_records_summary(db_session) if db_session else {},
    }


@router.get("/fields", tags=["generation"])
def list_generation_fields(
    prefix: Optional[str] = None,
    dataset_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> dict:
    """List known fields with optional prefix, dataset, or category filters."""
    db_session = _db_or_none(db)
    if db_session:
        records = top_field_records(
            db_session,
            dataset_ids=[dataset_id] if dataset_id else None,
            category=category,
            prefix=prefix,
            limit=limit,
        )
        return {"count": len(records), "fields": [{"name": record.name, "metadata": record_to_dict(record)} for record in records]}

    schema = get_data_fields()
    if dataset_id:
        fields = schema.fields_for_dataset_ids([dataset_id])
    elif category:
        fields = schema.fields_for_category(category)
    else:
        fields = sorted(schema.fields)

    if prefix:
        normalized_prefix = prefix.strip().lower()
        fields = [field for field in fields if field.startswith(normalized_prefix)]

    limited = fields[: max(1, min(limit, 500))]
    return {
        "count": len(fields),
        "fields": [
            {
                "name": field,
                "metadata": schema.field_info(field) or {},
            }
            for field in limited
        ],
    }


@router.get("/field-intelligence", tags=["generation"])
def list_field_intelligence(
    dataset_id: Optional[str] = None,
    category: Optional[str] = None,
    field_type: Optional[str] = None,
    prefix: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> dict:
    """List highest-scoring persisted fields for research/generation."""
    db_session = _db_or_none(db)
    if db_session is None:
        return {"summary": {}, "fields": []}
    records = top_field_records(
        db_session,
        dataset_ids=[dataset_id] if dataset_id else None,
        category=category,
        field_type=field_type,
        prefix=prefix,
        limit=limit,
    )
    return {"summary": field_records_summary(db_session), "fields": [record_to_dict(record) for record in records]}


@router.post("/generate", response_model=GenerateAlphaResponse, tags=["generation"])
def generate_alphas(request: GenerateAlphaRequest, db: Session = Depends(get_db)) -> GenerateAlphaResponse:
    """Generate candidate alpha expressions."""
    db_session = _db_or_none(db)
    warnings: List[str] = []
    base_count = request.count
    if request.include_refinements and request.count > 1:
        base_count = max(1, request.count // 2)

    generation_seed = request.seed
    generation_focus = request.focus
    generation_dataset_ids = request.dataset_ids
    random_metadata: Dict[str, Any] = {}
    if request.randomize or _random_requested(request):
        random_metadata = _random_generation_metadata(request.seed)
        generation_seed = random_metadata["seed"]
        generation_focus = random_metadata["focus"]
        generation_dataset_ids = [random_metadata["dataset_id"]]

    schema = schema_with_persisted_fields(db_session) if db_session else None
    fields = request.fields
    if not fields and generation_dataset_ids and db_session:
        fields = top_field_names(db_session, dataset_ids=generation_dataset_ids, limit=100)

    generator = RuleBasedAlphaGenerator(schema=schema, seed=generation_seed)
    candidates = generator.generate(
        count=base_count,
        focus=generation_focus,
        fields=fields,
        dataset_ids=generation_dataset_ids,
        neutralize=request.neutralize,
        existing_expressions=request.existing_expressions,
    )
    if random_metadata and len(candidates) < base_count:
        fallback_generator = RuleBasedAlphaGenerator(schema=schema, seed=(generation_seed or 0) + 1)
        fallback_candidates = fallback_generator.generate(
            count=base_count - len(candidates),
            focus=None,
            dataset_ids=["pv1"],
            neutralize=request.neutralize,
            existing_expressions=request.existing_expressions
            + [candidate.expression for candidate in candidates],
        )
        candidates.extend(fallback_candidates)

    if request.include_refinements and candidates:
        refiner = GeneticAlphaRefiner(seed=request.seed)
        refinement_count = request.count - len(candidates)
        refined = refiner.refine(
            [candidate.expression for candidate in candidates[:5]],
            count=refinement_count,
            existing_expressions=request.existing_expressions
            + [candidate.expression for candidate in candidates],
        )
        candidates.extend(refined)

    generation_settings = {
        **dataset_settings_overrides(generation_dataset_ids),
        **random_metadata.get("settings_overrides", {}),
    }

    openai_metadata: Dict[str, Any] = {}
    if request.use_openai and candidates:
        candidates, openai_metadata = apply_openai_advice(
            candidates,
            settings=generation_settings,
            focus=generation_focus,
            dataset_id=(generation_dataset_ids or [None])[0],
        )

    if len(candidates) < request.count:
        warnings.append(
            "Generated fewer candidates than requested; try a broader focus or more source fields."
        )

    return GenerateAlphaResponse(
        requested_count=request.count,
        generated_count=len(candidates),
        candidates=[AlphaCandidateResponse(**candidate.as_dict()) for candidate in candidates],
        settings_overrides=generation_settings,
        metadata={**random_metadata, **({"openai": openai_metadata} if openai_metadata else {})},
        warnings=warnings,
    )


@router.post("/refine", response_model=GenerateAlphaResponse, tags=["generation"])
def refine_alphas(request: RefineAlphaRequest) -> GenerateAlphaResponse:
    """Generate mutations and crossovers from seed expressions."""
    refiner = GeneticAlphaRefiner(seed=request.seed)
    candidates = refiner.refine(
        request.expressions,
        count=request.count,
        existing_expressions=request.existing_expressions,
    )
    warnings: List[str] = []
    if len(candidates) < request.count:
        warnings.append(
            "Generated fewer refinements than requested; add more varied seed expressions."
        )

    return GenerateAlphaResponse(
        requested_count=request.count,
        generated_count=len(candidates),
        candidates=[AlphaCandidateResponse(**candidate.as_dict()) for candidate in candidates],
        settings_overrides={},
        metadata={},
        warnings=warnings,
    )


@router.post("/deduplicate", response_model=DeduplicateResponse, tags=["generation"])
def deduplicate_alphas(request: DeduplicateRequest) -> DeduplicateResponse:
    """Normalize and deduplicate alpha expressions."""
    result = ExpressionDeduplicator(request.existing_expressions).dedupe(request.expressions)
    return DeduplicateResponse(
        total_count=len(request.expressions),
        unique_count=len(result.unique),
        duplicate_count=len(result.duplicates),
        unique=result.unique,
        duplicates=[
            DuplicateExpressionResponse(
                expression=duplicate.expression,
                duplicate_of=duplicate.duplicate_of,
                signature=duplicate.signature,
            )
            for duplicate in result.duplicates
        ],
    )


def _db_or_none(db) -> Optional[Session]:
    return db if hasattr(db, "query") else None


def _random_requested(request: GenerateAlphaRequest) -> bool:
    dataset_ids = [str(item).strip().lower() for item in request.dataset_ids or []]
    return (request.focus or "").strip().lower() == "random" or "random" in dataset_ids


def _random_generation_metadata(seed: Optional[int] = None) -> Dict[str, Any]:
    import random

    rng = random.Random(seed)
    profiles = list_dataset_profiles()
    profile = rng.choice(profiles)
    focuses = list(profile.preferred_focuses or ())
    if not focuses:
        focuses = [item["name"] for item in available_strategies()]
    chosen_seed = rng.randrange(1, 2_147_483_647)
    settings = dict(profile.settings_overrides or {})
    settings["decay"] = rng.choice((2, 4, 6, 8, 10, 12, 16, 20, 40, 60))
    settings["truncation"] = rng.choice((0.01, 0.02, 0.04, 0.08))
    if rng.random() < 0.35:
        settings["neutralization"] = rng.choice(("SUBINDUSTRY", "INDUSTRY", "SECTOR"))
    return {
        "mode": "random",
        "seed": chosen_seed,
        "dataset_id": profile.id,
        "focus": rng.choice(focuses),
        "settings_overrides": settings,
    }
