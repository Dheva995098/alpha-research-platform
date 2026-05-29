"""Persistent field discovery and scoring utilities."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from backend.core.data_fields import BRAINDataFields, LIVE_INVALID_FIELDS, is_live_invalid_field
from backend.core.dataset_catalog import (
    DATASET_PROFILES,
    datasets_for_fields,
    field_metadata,
    get_dataset_profile,
    list_dataset_profiles,
    normalize_dataset_ids,
)
from backend.models import DataFieldRecord


def ensure_catalog_field_records(db: Session) -> int:
    """Seed local dataset-catalog fields into the persistent field table."""
    seeded = 0
    local_metadata = field_metadata()
    for name, metadata in local_metadata.items():
        if is_live_invalid_field(name):
            continue
        dataset_ids = metadata.get("datasets") or list(datasets_for_fields([name]))
        dataset_id = dataset_ids[0] if dataset_ids else None
        profile = get_dataset_profile(dataset_id) if dataset_id else None
        record = db.query(DataFieldRecord).filter(DataFieldRecord.name == name).first()
        if record is None:
            record = DataFieldRecord(name=name)
            db.add(record)
            seeded += 1
        record.dataset_id = record.dataset_id or dataset_id
        record.category = record.category or (profile.category if profile else None)
        record.field_type = record.field_type or "MATRIX"
        record.region = record.region or "USA"
        record.universe = record.universe or (profile.default_universe if profile else "TOP3000")
        record.delay = record.delay if record.delay is not None else 1
        record.raw_metadata = {**(record.raw_metadata or {}), "local_catalog": metadata}
        record.field_score = score_field_record(record)
    db.commit()
    return seeded


def upsert_field_records(
    db: Session,
    fields: Iterable[Dict[str, Any]],
    dataset_id: Optional[str] = None,
    region: str = "USA",
    universe: Optional[str] = None,
    delay: int = 1,
) -> Dict[str, int]:
    """Insert/update live BRAIN field metadata and return import stats."""
    imported = 0
    updated = 0
    skipped = 0
    profile = get_dataset_profile(dataset_id) if dataset_id else None

    for field in fields:
        if not isinstance(field, dict):
            skipped += 1
            continue
        name = _field_name(field)
        if not name or is_live_invalid_field(name):
            skipped += 1
            continue

        detected_dataset_id = _dataset_id(field) or dataset_id
        detected_profile = get_dataset_profile(detected_dataset_id) if detected_dataset_id else profile
        record = db.query(DataFieldRecord).filter(DataFieldRecord.name == name).first()
        if record is None:
            record = DataFieldRecord(name=name)
            db.add(record)
            imported += 1
        else:
            updated += 1

        record.dataset_id = detected_dataset_id
        record.category = _text(field.get("category")) or (detected_profile.category if detected_profile else None)
        record.field_type = _text(field.get("type")) or record.field_type
        record.region = region
        record.universe = universe or (detected_profile.default_universe if detected_profile else record.universe)
        record.delay = delay
        record.coverage = _number(field.get("coverage"))
        record.alpha_count = _int(field.get("alphaCount") or field.get("alpha_count"))
        record.user_count = _int(field.get("userCount") or field.get("user_count"))
        record.value_score = _number(field.get("valueScore") or field.get("value_score"))
        record.description = _text(field.get("description")) or _text(field.get("name"))
        record.raw_metadata = field
        record.field_score = score_field_record(record)

    db.commit()
    return {"imported": imported, "updated": updated, "skipped": skipped}


def schema_with_persisted_fields(db: Optional[Session]) -> BRAINDataFields:
    """Create a schema enriched with persisted field records."""
    if db is None:
        return BRAINDataFields()
    ensure_catalog_field_records(db)
    names = {
        row.name
        for row in db.query(DataFieldRecord.name).all()
        if row.name and not is_live_invalid_field(row.name)
    }
    schema = BRAINDataFields(custom_fields=names)
    for record in db.query(DataFieldRecord).all():
        if is_live_invalid_field(record.name):
            continue
        schema.field_metadata[record.name] = record_to_metadata(record)
    return schema


def top_field_records(
    db: Session,
    dataset_ids: Optional[Sequence[str]] = None,
    category: Optional[str] = None,
    field_type: Optional[str] = None,
    prefix: Optional[str] = None,
    limit: int = 50,
) -> List[DataFieldRecord]:
    """Return highest-scoring fields with optional filters."""
    ensure_catalog_field_records(db)
    query = db.query(DataFieldRecord)
    if LIVE_INVALID_FIELDS:
        query = query.filter(~DataFieldRecord.name.in_(LIVE_INVALID_FIELDS))
    normalized_datasets = normalize_dataset_ids(dataset_ids)
    if normalized_datasets:
        query = query.filter(DataFieldRecord.dataset_id.in_(normalized_datasets))
    if category:
        query = query.filter(DataFieldRecord.category == _norm(category))
    if field_type:
        query = query.filter(DataFieldRecord.field_type == str(field_type).strip().upper())
    if prefix:
        query = query.filter(DataFieldRecord.name.like(f"{_norm(prefix)}%"))
    return (
        query.order_by(DataFieldRecord.field_score.desc(), DataFieldRecord.alpha_count.desc().nullslast(), DataFieldRecord.name)
        .limit(max(1, min(limit, 500)))
        .all()
    )


def top_field_names(
    db: Session,
    dataset_ids: Optional[Sequence[str]] = None,
    category: Optional[str] = None,
    limit: int = 80,
) -> List[str]:
    """Return names of top fields for generator allow-lists."""
    return [record.name for record in top_field_records(db, dataset_ids=dataset_ids, category=category, limit=limit)]


def field_records_summary(db: Session) -> Dict[str, Any]:
    """Aggregate field-intelligence counts for dashboard/status views."""
    ensure_catalog_field_records(db)
    total = db.query(DataFieldRecord).count()
    by_dataset = []
    for profile in list_dataset_profiles():
        count = db.query(DataFieldRecord).filter(DataFieldRecord.dataset_id == profile.id).count()
        if count:
            best = (
                db.query(DataFieldRecord)
                .filter(DataFieldRecord.dataset_id == profile.id)
                .order_by(DataFieldRecord.field_score.desc())
                .first()
            )
            by_dataset.append(
                {
                    "dataset_id": profile.id,
                    "category": profile.category,
                    "count": count,
                    "best_field": best.name if best else None,
                    "best_score": round(best.field_score, 4) if best else None,
                }
            )
    return {"total": total, "by_dataset": sorted(by_dataset, key=lambda item: item["count"], reverse=True)}


def record_to_dict(record: DataFieldRecord) -> Dict[str, Any]:
    """Return JSON-safe field record data."""
    return {
        "name": record.name,
        "dataset_id": record.dataset_id,
        "category": record.category,
        "type": record.field_type,
        "region": record.region,
        "universe": record.universe,
        "delay": record.delay,
        "coverage": record.coverage,
        "alpha_count": record.alpha_count,
        "user_count": record.user_count,
        "value_score": record.value_score,
        "field_score": round(record.field_score or 0.0, 4),
        "description": record.description,
    }


def record_to_metadata(record: DataFieldRecord) -> Dict[str, Any]:
    """Return metadata shape compatible with BRAINDataFields.field_info."""
    return {
        "name": record.name,
        "type": record.field_type,
        "description": record.description,
        "datasets": [record.dataset_id] if record.dataset_id else [],
        "categories": [record.category] if record.category else [],
        "coverage": record.coverage,
        "alphaCount": record.alpha_count,
        "userCount": record.user_count,
        "valueScore": record.value_score,
        "fieldScore": record.field_score,
        "raw": record.raw_metadata or {},
    }


def score_field_record(record: DataFieldRecord) -> float:
    """Score field usefulness from coverage, type, and usage/crowding signals."""
    coverage = _coverage_score(record.coverage)
    alpha_count = max(record.alpha_count or 0, 0)
    user_count = max(record.user_count or 0, 0)
    usage = min(math.log1p(alpha_count) / math.log1p(250), 1.0)
    users = min(math.log1p(user_count) / math.log1p(150), 1.0)
    crowding_penalty = 0.0
    if alpha_count > 350:
        crowding_penalty += min((alpha_count - 350) / 1000.0, 0.18)
    if user_count > 250:
        crowding_penalty += min((user_count - 250) / 1000.0, 0.10)

    type_bonus = {
        "MATRIX": 0.13,
        "VECTOR": 0.06,
        "GROUP": -0.02,
        "UNIVERSE": -0.05,
    }.get(str(record.field_type or "MATRIX").upper(), 0.08)

    category_bonus = {
        "options": 0.08,
        "analyst": 0.07,
        "news_sentiment": 0.06,
        "fundamental": 0.05,
        "model_risk": 0.05,
        "price_volume": 0.03,
    }.get(record.category or "", 0.02)

    value_score = min(max((record.value_score or 0.0) / 10.0, 0.0), 1.0)
    score = 0.44 * coverage + 0.20 * usage + 0.10 * users + 0.08 * value_score + type_bonus + category_bonus
    return round(max(min(score - crowding_penalty, 1.0), 0.0), 4)


def _coverage_score(value: Optional[float]) -> float:
    if value is None:
        return 0.55
    value = float(value)
    if value > 1.0:
        value = value / 100.0
    return max(min(value, 1.0), 0.0)


def _field_name(field: Dict[str, Any]) -> str:
    return _norm(field.get("id") or field.get("name"))


def _dataset_id(field: Dict[str, Any]) -> Optional[str]:
    dataset = field.get("dataset")
    if isinstance(dataset, dict):
        return _norm(dataset.get("id") or dataset.get("name"))
    return _norm(field.get("dataset_id") or field.get("datasetId")) or None


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None else None
