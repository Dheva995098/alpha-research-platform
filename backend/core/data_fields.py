"""
Data fields schema and validators for BRAIN expressions.
Caches 125,000+ available data fields and FASTEXPR operators.
"""
import logging
from typing import Set, Dict, List, Optional
import json
import re

from backend.core.dataset_catalog import (
    fields_for_datasets,
    field_metadata,
    list_dataset_profiles,
    normalize_dataset_ids,
)

logger = logging.getLogger(__name__)

LIVE_INVALID_FIELDS = {
    "news_sentiment",
    "news_volume",
    "scl12_volume",
}


def is_live_invalid_field(field: str) -> bool:
    """Return true for catalog fields rejected by the live BRAIN API."""
    return str(field or "").strip().lower() in LIVE_INVALID_FIELDS


class BRAINDataFields:
    """
    Schema of available BRAIN data fields and operators.
    Provides validation for alpha expressions.
    """
    
    # Common time-series data fields (subset of 125,000+)
    CORE_FIELDS = {
        "open", "close", "high", "low", "vwap", "volume",
        "adv20", "adv5", "cap", "mcap", "returns", "ret_0_1",
        "ret_1_5", "ret_5_20", "ret_20_60", "ret_60_252",
        "volatility", "beta", "momentum", "rsi", "sma_20",
        "sma_50", "sma_200", "ema_12", "ema_26", "macd",
        "ebit", "ebitda", "capex", "sales", "revenue", "assets",
        "liabilities", "equity", "debt", "cash", "cashflow",
        "operating_income", "net_income", "gross_profit", "eps",
        "bookvalue", "dividend", "free_cash_flow", "operating_cash_flow",
        "cashflow_op",
        "enterprise_value", "cashflow", "est_eps", "est_ptp", "est_fcf",
        "etz_eps",
        "est_cashflow_op", "est_capex", "fn_liab_fair_val_l1_a",
        "scl12_buzz", "implied_volatility_call_120",
        "implied_volatility_put_120", "implied_volatility_call_180",
        "implied_volatility_put_180", "implied_volatility_mean_180",
        "implied_volatility_call_270", "implied_volatility_put_270",
        "pcr_oi_270", "parkinson_volatility_120", "rel_ret_all",
        "fnd6_newqv1300_ivltq", "news_pct_1min", "news_max_up_ret",
        "inventory", "average_inventory", "sales_growth",
        "pcr_oi_720", "implied_volatility_call_720", "implied_volatility_put_720",
        "scl12_alltype_buzzvec", "scl12_sentiment",
        "snt1_score", "snt1_sentiment", "snt1_buzz",
        "rpna_sentiment", "rpna_relevance",
        "mdl16_quality_score", "mdl16_value_score", "mdl16_growth_score",
        "mdl77_momentum", "mdl77_reversal", "mdl77_volatility",
        "mdl53_credit_score", "mdl53_default_risk",
        "fnd7_net_income", "fnd7_total_assets", "fnd7_operating_cash_flow", "fnd7_sales",
        "anl4_eps_mean", "anl4_ptp_mean",
    }
    
    # FASTEXPR operators (aligned with the live BRAIN operator reference)
    TIME_SERIES_OPS = {
        "ts_rank", "ts_corr", "ts_covariance", "ts_mean",
        "ts_sum", "ts_decay_linear", "ts_decay_exp", "ts_decay_exp_window",
        "ts_min", "ts_max", "ts_zscore", "ts_skewness",
        "ts_kurtosis", "ts_count", "ts_count_nans", "ts_delta",
        "ts_arg_min", "ts_arg_max", "ts_product", "ts_scale",
        "ts_av_diff", "ts_backfill", "ts_regression",
        "ts_delay", "ts_std_dev", "ts_step", "ts_median",
        "ts_quantile", "ts_moment", "ts_ir", "ts_entropy",
        "ts_returns", "ts_weighted_delay", "days_from_last_change",
        "last_diff_value", "hump",
    }

    GROUPING_OPS = {
        "group_neutralize",  # group_neutralize(expr, sector|industry|subindustry|market)
        "group_rank",
        "group_scale",
        "group_mean",
        "group_std",
        "group_zscore",
        "group_backfill",
        "group_vector_neut",  # group_vector_neut(x, y, group) -> in-group orthogonalization
        "group_count",
        "group_normalize",
        "group_sum",
        "group_max",
        "group_min",
    }

    MATH_OPS = {
        "rank", "sigmoid", "scale", "zscore", "sign", "power",
        "abs", "sqrt", "log", "exp", "sin", "cos", "tan",
        "ceil", "floor", "round", "min", "max", "clip",
        "winsorize", "normalize", "quantile", "reverse",
        "signed_power", "signedpower", "vector_neut",
        "tail", "densify", "fraction", "log_diff", "purify",
        "bucket", "pasteurize",
    }

    LOGIC_OPS = {
        "if_else", "and", "or", "not", "mask", "trade_when", "is_nan"
    }

    VECTOR_OPS = {
        "vec_avg", "vec_sum", "vec_count", "vec_max", "vec_min",
        "vec_stddev", "vec_norm", "vec_range", "vec_ir",
        "vec_skewness", "vec_kurtosis", "vec_choose",
    }
    
    ALL_OPS = TIME_SERIES_OPS | GROUPING_OPS | MATH_OPS | LOGIC_OPS | VECTOR_OPS
    
    def __init__(self, custom_fields: Optional[Set[str]] = None):
        """
        Initialize with core fields + custom fields (from API).
        custom_fields: additional fields fetched from BRAIN API.
        """
        self.field_metadata = field_metadata()
        self.field_metadata = {
            name: metadata
            for name, metadata in self.field_metadata.items()
            if not is_live_invalid_field(name)
        }
        self.fields = (self.CORE_FIELDS | set(self.field_metadata)) - LIVE_INVALID_FIELDS
        if custom_fields:
            self.fields.update(
                field.strip().lower()
                for field in custom_fields
                if not is_live_invalid_field(field)
            )
        logger.info(f"Initialized BRAINDataFields with {len(self.fields)} fields")
    
    def add_fields_from_api(self, api_fields: List[Dict]) -> None:
        """
        Add fields from BRAIN API response.
        api_fields: list of {"name": "...", "type": "...", ...}
        """
        for field_def in api_fields:
            field_name = field_def.get("name") or field_def.get("id")
            if field_name:
                normalized = str(field_name).strip().lower()
                if is_live_invalid_field(normalized):
                    continue
                self.fields.add(normalized)
                self.field_metadata[normalized] = self._metadata_from_api_field(field_def, normalized)
        logger.info(f"Added {len(api_fields)} fields from API. Total: {len(self.fields)}")
    
    def validate_field(self, field: str) -> bool:
        """Check if field exists in schema."""
        return field in self.fields
    
    def validate_operator(self, op: str) -> bool:
        """Check if operator is valid FASTEXPR operator."""
        return op in self.ALL_OPS
    
    def validate_expression_basic(self, expression: str) -> tuple[bool, str]:
        """
        Basic validation of FASTEXPR expression.
        Returns (is_valid, error_message).
        """
        if not expression or not isinstance(expression, str):
            return False, "Expression must be non-empty string"
        
        if len(expression) > 10000:
            return False, "Expression too long (max 10000 chars)"

        for field in LIVE_INVALID_FIELDS:
            if re.search(rf"\b{re.escape(field)}\b", expression, flags=re.IGNORECASE):
                return False, f"Field rejected by live BRAIN API: {field}"
        
        # Simple parenthesis balance check
        if expression.count("(") != expression.count(")"):
            return False, "Unmatched parentheses"
        
        # Extract function-like tokens and reject operators BRAIN will not know.
        for op in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*(?=\()", expression):
            op = op.strip()
            if not self.validate_operator(op):
                return False, f"Unknown operator: {op}"
        
        # Check for suspicious patterns (injection attempts)
        dangerous = [";", "--", "/*", "*/", "import", "exec", "__"]
        for pattern in dangerous:
            if pattern in expression.lower():
                return False, f"Suspicious pattern detected: {pattern}"
        
        return True, "Valid"
    
    def suggest_fields(self, prefix: str, limit: int = 10) -> List[str]:
        """Suggest fields matching prefix (autocomplete)."""
        matching = [f for f in self.fields if f.startswith(prefix.lower())]
        return sorted(matching)[:limit]

    def field_info(self, field: str) -> Optional[Dict]:
        """Return local/API metadata for a data field."""
        return self.field_metadata.get(str(field or "").strip().lower())

    def fields_for_dataset_ids(self, dataset_ids: Optional[List[str]]) -> List[str]:
        """Return valid known fields for selected datasets."""
        selected = fields_for_datasets(normalize_dataset_ids(dataset_ids))
        return sorted(field for field in selected if self.validate_field(field))

    def fields_for_category(self, category: str) -> List[str]:
        """Return valid known fields for one dataset category."""
        dataset_ids = [profile.id for profile in list_dataset_profiles(category)]
        return self.fields_for_dataset_ids(dataset_ids)
    
    def get_operator_info(self, op: str) -> Optional[Dict]:
        """Get information about an operator."""
        op_info = {
            "ts_rank": "Time series rank of values over window",
            "ts_corr": "Time series correlation over window",
            "ts_mean": "Time series mean over window",
            "ts_std_dev": "Time series standard deviation over window",
            "ts_decay_linear": "Time series linear decay weighting",
            "rank": "Cross-sectional rank (0-1)",
            "sigmoid": "Sigmoid normalization",
            "zscore": "Z-score normalization",
            "group_neutralize": "Neutralize by group (sector, industry, etc)",
        }
        return op_info.get(op)
    
    def export_schema(self) -> Dict:
        """Export schema for frontend/validation."""
        return {
            "fields": sorted(list(self.fields)),
            "operators": {
                "time_series": sorted(list(self.TIME_SERIES_OPS)),
                "grouping": sorted(list(self.GROUPING_OPS)),
                "math": sorted(list(self.MATH_OPS)),
                "logic": sorted(list(self.LOGIC_OPS)),
                "vector": sorted(list(self.VECTOR_OPS)),
            },
            "datasets": [
                {
                    **profile.as_dict(),
                    "known_field_count": len([field for field in profile.example_fields if field in self.fields]),
                }
                for profile in list_dataset_profiles()
            ],
            "total_fields": len(self.fields),
            "total_operators": len(self.ALL_OPS),
        }

    @staticmethod
    def _metadata_from_api_field(field_def: Dict, name: str) -> Dict:
        dataset = field_def.get("dataset")
        dataset_id = None
        if isinstance(dataset, dict):
            dataset_id = dataset.get("id") or dataset.get("name")
        return {
            "name": name,
            "type": field_def.get("type"),
            "description": field_def.get("description"),
            "datasets": [dataset_id] if dataset_id else [],
            "categories": [],
            "alpha_count": field_def.get("alphaCount"),
            "user_count": field_def.get("userCount"),
            "coverage": field_def.get("coverage"),
            "raw": field_def,
        }


# Global schema instance (lazy-loaded)
_schema_instance = None


def get_data_fields() -> BRAINDataFields:
    """Get or create global data fields schema."""
    global _schema_instance
    if _schema_instance is None:
        _schema_instance = BRAINDataFields()
    return _schema_instance


def set_data_fields(schema: BRAINDataFields) -> None:
    """Set global data fields schema (e.g., after API fetch)."""
    global _schema_instance
    _schema_instance = schema
