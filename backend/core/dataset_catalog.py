"""Dataset-family catalog for WorldQuant BRAIN style data fields.

The live BRAIN API remains the source of truth for exact field availability.
This catalog gives the local generator a richer map of public/common dataset
families so it can target fundamentals, analyst data, options, news/social, and
model/risk fields instead of relying only on OHLCV examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class DatasetProfile:
    """Local metadata for one BRAIN dataset family."""

    id: str
    name: str
    category: str
    description: str
    default_universe: str = "TOP3000"
    field_prefix: Optional[str] = None
    example_fields: Tuple[str, ...] = ()
    preferred_focuses: Tuple[str, ...] = ()
    settings_overrides: Dict[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "default_universe": self.default_universe,
            "field_prefix": self.field_prefix,
            "example_fields": list(self.example_fields),
            "preferred_focuses": list(self.preferred_focuses),
            "settings_overrides": self.settings_overrides or {},
        }


DATASET_PROFILES: Dict[str, DatasetProfile] = {
    "pv1": DatasetProfile(
        id="pv1",
        name="Price Volume",
        category="price_volume",
        description="Core OHLCV, returns, liquidity, size, and realized-risk fields.",
        field_prefix=None,
        example_fields=(
            "open",
            "close",
            "high",
            "low",
            "vwap",
            "volume",
            "adv20",
            "adv5",
            "returns",
            "ret_0_1",
            "ret_1_5",
            "ret_5_20",
            "ret_20_60",
            "ret_60_252",
            "cap",
            "mcap",
            "beta",
            "volatility",
            "rel_ret_all",
            "parkinson_volatility_120",
        ),
        preferred_focuses=("momentum", "mean_reversion", "price_volume", "liquidity", "intraday", "volatility"),
        settings_overrides={"universe": "TOP3000", "neutralization": "SUBINDUSTRY"},
    ),
    "fundamental2": DatasetProfile(
        id="fundamental2",
        name="Report Footnotes",
        category="fundamental",
        description="Report-footnote and filing-derived fundamental indicators.",
        field_prefix="fn",
        example_fields=("fn_liab_fair_val_l1_a", "fn_report_count", "fn_footnote_length"),
        preferred_focuses=("fundamental", "quality", "decorrelation"),
        settings_overrides={"universe": "TOP3000", "neutralization": "SUBINDUSTRY"},
    ),
    "fundamental6": DatasetProfile(
        id="fundamental6",
        name="Financial Statements",
        category="fundamental",
        description="Statement, balance-sheet, profitability, investment, and cash-flow fields.",
        example_fields=(
            "ebit",
            "ebitda",
            "operating_income",
            "net_income",
            "gross_profit",
            "sales",
            "revenue",
            "assets",
            "liabilities",
            "equity",
            "debt",
            "cash",
            "cashflow",
            "cashflow_op",
            "free_cash_flow",
            "operating_cash_flow",
            "capex",
            "eps",
            "bookvalue",
            "dividend",
            "enterprise_value",
            "inventory",
            "average_inventory",
            "sales_growth",
            "fnd6_newqv1300_ivltq",
        ),
        preferred_focuses=("fundamental", "quality", "hybrid"),
        settings_overrides={"universe": "TOP3000", "neutralization": "SUBINDUSTRY"},
    ),
    "fundamental7": DatasetProfile(
        id="fundamental7",
        name="Comprehensive Fundamentals",
        category="fundamental",
        description="Broad fundamental data variants useful for long-horizon value and quality ideas.",
        field_prefix="fnd7",
        example_fields=(
            "fnd7_net_income",
            "fnd7_total_assets",
            "fnd7_operating_cash_flow",
            "fnd7_sales",
            "operating_income",
            "assets",
            "cashflow_op",
        ),
        preferred_focuses=("fundamental", "quality", "decorrelation"),
        settings_overrides={"universe": "TOP3000", "neutralization": "SUBINDUSTRY"},
    ),
    "analyst7": DatasetProfile(
        id="analyst7",
        name="Broker Estimates",
        category="analyst",
        description="Analyst forecast, target-price, earnings, and cash-flow estimate fields.",
        example_fields=("est_eps", "est_ptp", "est_fcf", "est_cashflow_op", "est_capex", "etz_eps"),
        preferred_focuses=("analyst", "hybrid", "decorrelation"),
        settings_overrides={"universe": "TOP3000", "neutralization": "INDUSTRY"},
    ),
    "analyst4": DatasetProfile(
        id="analyst4",
        name="Analyst Estimates SP500",
        category="analyst",
        description="Analyst estimate variants commonly used on more liquid US universes.",
        default_universe="TOPSP500",
        field_prefix="anl4",
        example_fields=("est_eps", "est_ptp", "est_fcf", "anl4_eps_mean", "anl4_ptp_mean"),
        preferred_focuses=("analyst", "hybrid"),
        settings_overrides={"universe": "TOPSP500", "neutralization": "INDUSTRY"},
    ),
    "option8": DatasetProfile(
        id="option8",
        name="Options Implied Volatility",
        category="options",
        description="Options implied-volatility, put/call, and open-interest signals.",
        default_universe="TOPSP500",
        example_fields=(
            "implied_volatility_call_120",
            "implied_volatility_put_120",
            "implied_volatility_call_180",
            "implied_volatility_put_180",
            "implied_volatility_mean_180",
            "implied_volatility_call_270",
            "implied_volatility_put_270",
            "implied_volatility_call_720",
            "implied_volatility_put_720",
            "pcr_oi_270",
            "pcr_oi_720",
        ),
        preferred_focuses=("options", "volatility", "decorrelation"),
        settings_overrides={"universe": "TOPSP500", "neutralization": "SECTOR", "truncation": 0.08, "maxTrade": "OFF"},
    ),
    "option9": DatasetProfile(
        id="option9",
        name="Options And Event Data",
        category="options",
        description="Event-aware options and volatility fields useful for gated trading ideas.",
        default_universe="TOPSP500",
        example_fields=("pcr_oi_270", "pcr_oi_720", "implied_volatility_call_720", "implied_volatility_put_720"),
        preferred_focuses=("options", "sentiment", "hybrid"),
        settings_overrides={"universe": "TOPSP500", "neutralization": "SECTOR", "maxTrade": "OFF"},
    ),
    "news12": DatasetProfile(
        id="news12",
        name="US News",
        category="news_sentiment",
        description="News event, attention, and reaction fields.",
        default_universe="TOP200",
        field_prefix="news",
        example_fields=("news_pct_1min", "news_max_up_ret"),
        preferred_focuses=("sentiment", "hybrid", "decorrelation"),
        settings_overrides={"universe": "TOP200", "neutralization": "SECTOR", "truncation": 0.08},
    ),
    "news18": DatasetProfile(
        id="news18",
        name="Ravenpack News",
        category="news_sentiment",
        description="Ravenpack-style news analytics and event sentiment fields.",
        default_universe="TOP200",
        example_fields=("rpna_sentiment", "rpna_relevance", "news_pct_1min", "news_max_up_ret"),
        preferred_focuses=("sentiment", "hybrid"),
        settings_overrides={"universe": "TOP200", "neutralization": "SECTOR", "truncation": 0.08},
    ),
    "socialmedia12": DatasetProfile(
        id="socialmedia12",
        name="Social Media Buzz",
        category="news_sentiment",
        description="Social attention, buzz, and crowd-sentiment fields.",
        default_universe="TOP500",
        field_prefix="scl12",
        example_fields=("scl12_buzz", "scl12_alltype_buzzvec", "scl12_sentiment"),
        preferred_focuses=("sentiment", "hybrid", "decorrelation"),
        settings_overrides={"universe": "TOP500", "neutralization": "SECTOR", "truncation": 0.08},
    ),
    "sentiment1": DatasetProfile(
        id="sentiment1",
        name="Sentiment Scores",
        category="news_sentiment",
        description="Composite sentiment score fields for attention and mood changes.",
        default_universe="TOP500",
        field_prefix="snt1",
        example_fields=("snt1_score", "snt1_sentiment", "snt1_buzz", "scl12_buzz"),
        preferred_focuses=("sentiment", "hybrid"),
        settings_overrides={"universe": "TOP500", "neutralization": "SECTOR", "truncation": 0.08},
    ),
    "model51": DatasetProfile(
        id="model51",
        name="Systematic Risk",
        category="model_risk",
        description="Model-derived systematic risk, volatility, and residual-return indicators.",
        example_fields=("beta", "volatility", "rel_ret_all", "momentum", "parkinson_volatility_120"),
        preferred_focuses=("model_risk", "volatility", "decorrelation"),
        settings_overrides={"universe": "TOP3000", "neutralization": "SUBINDUSTRY"},
    ),
    "model16": DatasetProfile(
        id="model16",
        name="Fundamental Scores",
        category="model_risk",
        description="Model-derived fundamental score features for quality/value overlays.",
        field_prefix="mdl16",
        example_fields=("mdl16_quality_score", "mdl16_value_score", "mdl16_growth_score", "operating_income", "assets"),
        preferred_focuses=("model_risk", "quality", "hybrid"),
        settings_overrides={"universe": "TOP3000", "neutralization": "SUBINDUSTRY"},
    ),
    "model77": DatasetProfile(
        id="model77",
        name="Technical Models",
        category="model_risk",
        description="Model and technical-score features for trend, reversal, and volatility regimes.",
        field_prefix="mdl77",
        example_fields=("mdl77_momentum", "mdl77_reversal", "mdl77_volatility", "momentum", "rsi", "macd"),
        preferred_focuses=("model_risk", "momentum", "mean_reversion", "volatility"),
        settings_overrides={"universe": "TOP3000", "neutralization": "SUBINDUSTRY"},
    ),
    "model53": DatasetProfile(
        id="model53",
        name="Credit Risk Model",
        category="model_risk",
        description="Creditworthiness and risk-model fields for defensive quality screens.",
        field_prefix="mdl53",
        example_fields=("mdl53_credit_score", "mdl53_default_risk", "debt", "assets", "liabilities"),
        preferred_focuses=("model_risk", "quality", "fundamental"),
        settings_overrides={"universe": "TOP3000", "neutralization": "SUBINDUSTRY"},
    ),
}


def list_dataset_profiles(category: Optional[str] = None) -> List[DatasetProfile]:
    """Return dataset profiles, optionally filtered by category."""
    normalized = _normalize(category)
    profiles = sorted(DATASET_PROFILES.values(), key=lambda item: (item.category, item.id))
    if not normalized:
        return profiles
    return [profile for profile in profiles if _normalize(profile.category) == normalized]


def get_dataset_profile(dataset_id: str) -> Optional[DatasetProfile]:
    """Return one dataset profile by id."""
    return DATASET_PROFILES.get(_normalize(dataset_id))


def normalize_dataset_ids(dataset_ids: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Normalize and keep only known dataset ids."""
    if not dataset_ids:
        return ()
    normalized = []
    for dataset_id in dataset_ids:
        key = _normalize(dataset_id)
        if key in DATASET_PROFILES and key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def fields_for_datasets(dataset_ids: Optional[Iterable[str]]) -> Set[str]:
    """Return example fields associated with selected datasets."""
    fields: Set[str] = set()
    for dataset_id in normalize_dataset_ids(dataset_ids):
        fields.update(DATASET_PROFILES[dataset_id].example_fields)
    return fields


def focuses_for_datasets(dataset_ids: Optional[Iterable[str]]) -> Set[str]:
    """Return preferred generator focuses for selected datasets."""
    focuses: Set[str] = set()
    for dataset_id in normalize_dataset_ids(dataset_ids):
        focuses.update(DATASET_PROFILES[dataset_id].preferred_focuses)
    return focuses


def datasets_for_fields(fields: Iterable[str]) -> Tuple[str, ...]:
    """Return dataset ids containing any of the supplied field names."""
    field_set = {_normalize(field) for field in fields}
    matches = [
        profile.id
        for profile in DATASET_PROFILES.values()
        if field_set.intersection({_normalize(field) for field in profile.example_fields})
    ]
    return tuple(dict.fromkeys(matches))


def field_metadata() -> Dict[str, Dict[str, Any]]:
    """Return local field metadata derived from dataset profiles."""
    metadata: Dict[str, Dict[str, Any]] = {}
    for profile in DATASET_PROFILES.values():
        for field in profile.example_fields:
            key = _normalize(field)
            item = metadata.setdefault(
                key,
                {
                    "name": key,
                    "datasets": [],
                    "categories": [],
                    "field_prefix": profile.field_prefix,
                },
            )
            if profile.id not in item["datasets"]:
                item["datasets"].append(profile.id)
            if profile.category not in item["categories"]:
                item["categories"].append(profile.category)
    return metadata


def category_names() -> Tuple[str, ...]:
    """Return known dataset category names."""
    return tuple(sorted({profile.category for profile in DATASET_PROFILES.values()}))


def dataset_settings_overrides(dataset_ids: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Merge local settings hints for selected datasets."""
    settings: Dict[str, Any] = {}
    for dataset_id in normalize_dataset_ids(dataset_ids):
        settings.update(DATASET_PROFILES[dataset_id].settings_overrides or {})
    return settings


def _normalize(value: Optional[str]) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
