"""Feature extraction for alpha expressions."""
from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import mean
from typing import Dict, List, Sequence

from backend.core.data_fields import BRAINDataFields, get_data_fields


FEATURE_NAMES = [
    "expression_length",
    "token_count",
    "field_count",
    "unique_field_count",
    "operator_count",
    "time_series_operator_count",
    "group_operator_count",
    "math_operator_count",
    "window_count",
    "avg_window",
    "max_window",
    "nesting_depth",
    "has_group_neutralize",
    "has_price_volume_pair",
    "has_fundamental_pair",
    "has_reversion_shape",
    "has_outlier_control",
    "has_trade_when",
    "has_backfill",
    "has_group_rank",
    "has_options_data",
    "has_analyst_data",
    "has_sentiment_data",
    "has_model_risk_data",
    "has_alternative_data",
    "constant_count",
    "arithmetic_operator_count",
    "result_sharpe",
    "result_fitness",
    "result_turnover",
    "result_self_correlation",
    "result_checks_passed",
    "result_check_pass_rate",
    "result_check_fail_count",
    "result_check_pending_count",
    "failed_low_sharpe",
    "failed_low_fitness",
    "failed_low_sub_universe_sharpe",
    "failed_turnover",
    "pending_self_correlation",
    "result_grade_score",
    "setting_decay",
    "setting_truncation",
    "setting_delay",
    "setting_region_usa",
    "setting_region_chn",
    "setting_universe_top3000",
    "setting_universe_top1000",
    "setting_universe_top500",
    "setting_universe_top200",
    "setting_neutralization_subindustry",
    "setting_neutralization_sector",
    "setting_neutralization_industry",
    "setting_neutralization_market",
    "setting_neutralization_none",
    "setting_max_trade_off",
    "setting_options_profile",
]


@dataclass(frozen=True)
class ExpressionFeatures:
    """Structured features extracted from one expression."""

    expression: str
    values: Dict[str, float]
    fields: List[str]
    operators: List[str]
    windows: List[int]

    def vector(self, feature_names: Sequence[str] = FEATURE_NAMES) -> List[float]:
        """Return values in stable feature order."""
        return [float(self.values.get(name, 0.0)) for name in feature_names]


class ExpressionFeatureExtractor:
    """Extract numeric features from FASTEXPR-like expressions."""

    def __init__(self, schema: BRAINDataFields | None = None):
        self.schema = schema or get_data_fields()

    def extract(self, expression: str, metrics: Dict | None = None) -> ExpressionFeatures:
        """Extract expression and optional result-metric features."""
        expression = expression or ""
        metrics = metrics or {}
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression)
        operators = self._extract_operators(expression)
        fields = [token for token in tokens if self.schema.validate_field(token)]
        windows = self._extract_windows(expression)
        constants = re.findall(r"(?<![a-zA-Z_])-?\d+(?:\.\d+)?", expression)
        check_stats = self._check_stats(metrics)
        settings = self._settings(metrics)

        values = {
            "expression_length": min(len(expression) / 500.0, 5.0),
            "token_count": min(len(tokens) / 50.0, 5.0),
            "field_count": min(len(fields) / 10.0, 5.0),
            "unique_field_count": min(len(set(fields)) / 10.0, 5.0),
            "operator_count": min(len(operators) / 10.0, 5.0),
            "time_series_operator_count": min(
                len([operator for operator in operators if operator in self.schema.TIME_SERIES_OPS]) / 8.0,
                5.0,
            ),
            "group_operator_count": min(
                len([operator for operator in operators if operator in self.schema.GROUPING_OPS]) / 4.0,
                5.0,
            ),
            "math_operator_count": min(
                len([operator for operator in operators if operator in self.schema.MATH_OPS]) / 8.0,
                5.0,
            ),
            "window_count": min(len(windows) / 6.0, 5.0),
            "avg_window": min((mean(windows) if windows else 0.0) / 252.0, 5.0),
            "max_window": min((max(windows) if windows else 0.0) / 252.0, 5.0),
            "nesting_depth": min(self._nesting_depth(expression) / 10.0, 5.0),
            "has_group_neutralize": 1.0 if "group_neutralize" in operators else 0.0,
            "has_price_volume_pair": 1.0 if self._has_price_volume_pair(fields) else 0.0,
            "has_fundamental_pair": 1.0 if self._has_fundamental_pair(fields) else 0.0,
            "has_reversion_shape": 1.0 if self._has_reversion_shape(expression, operators) else 0.0,
            "has_outlier_control": 1.0 if {"winsorize", "normalize", "zscore"} & set(operators) else 0.0,
            "has_trade_when": 1.0 if "trade_when" in operators else 0.0,
            "has_backfill": 1.0 if {"ts_backfill", "group_backfill"} & set(operators) else 0.0,
            "has_group_rank": 1.0 if "group_rank" in operators else 0.0,
            "has_options_data": 1.0 if self._has_prefix(fields, ("implied_volatility", "pcr_")) else 0.0,
            "has_analyst_data": 1.0 if self._has_prefix(fields, ("est_",)) else 0.0,
            "has_sentiment_data": 1.0 if self._has_prefix(fields, ("news_", "scl")) else 0.0,
            "has_model_risk_data": 1.0 if self._has_model_risk_data(fields) else 0.0,
            "has_alternative_data": 1.0
            if self._has_prefix(fields, ("implied_volatility", "pcr_", "est_", "news_", "scl", "snt1", "rpna"))
            else 0.0,
            "constant_count": min(len(constants) / 10.0, 5.0),
            "arithmetic_operator_count": min(len(re.findall(r"[+\-*/]", expression)) / 10.0, 5.0),
            "result_sharpe": self._metric(metrics, "sharpe", scale=3.0),
            "result_fitness": self._metric(metrics, "fitness", scale=3.0),
            "result_turnover": self._metric(metrics, "turnover", scale=1.0),
            "result_self_correlation": self._metric(
                metrics,
                "self_correlation",
                aliases=("selfCorrelation",),
                scale=1.0,
            ),
            "result_checks_passed": self._bool_metric(
                metrics,
                "all_checks_passed",
                aliases=("checksPassed", "passes_checks"),
            ),
            "result_check_pass_rate": check_stats["pass_rate"],
            "result_check_fail_count": min(check_stats["fail_count"] / 8.0, 5.0),
            "result_check_pending_count": min(check_stats["pending_count"] / 8.0, 5.0),
            "failed_low_sharpe": 1.0 if "low_sharpe" in check_stats["failed_names"] else 0.0,
            "failed_low_fitness": 1.0 if "low_fitness" in check_stats["failed_names"] else 0.0,
            "failed_low_sub_universe_sharpe": 1.0
            if "low_sub_universe_sharpe" in check_stats["failed_names"]
            else 0.0,
            "failed_turnover": 1.0
            if {"high_turnover", "low_turnover"} & check_stats["failed_names"]
            else 0.0,
            "pending_self_correlation": 1.0 if "self_correlation" in check_stats["pending_names"] else 0.0,
            "result_grade_score": self._grade_score(metrics),
            "setting_decay": self._setting_number(settings, "decay", scale=64.0),
            "setting_truncation": self._setting_number(settings, "truncation", scale=0.10),
            "setting_delay": self._setting_number(settings, "delay", scale=2.0),
            "setting_region_usa": self._setting_equals(settings, "region", "USA"),
            "setting_region_chn": self._setting_equals(settings, "region", "CHN"),
            "setting_universe_top3000": self._setting_equals(settings, "universe", "TOP3000"),
            "setting_universe_top1000": self._setting_equals(settings, "universe", "TOP1000"),
            "setting_universe_top500": self._setting_equals(settings, "universe", "TOP500"),
            "setting_universe_top200": self._setting_equals(settings, "universe", "TOP200"),
            "setting_neutralization_subindustry": self._setting_equals(settings, "neutralization", "SUBINDUSTRY"),
            "setting_neutralization_sector": self._setting_equals(settings, "neutralization", "SECTOR"),
            "setting_neutralization_industry": self._setting_equals(settings, "neutralization", "INDUSTRY"),
            "setting_neutralization_market": self._setting_equals(settings, "neutralization", "MARKET"),
            "setting_neutralization_none": self._setting_equals(settings, "neutralization", "NONE"),
            "setting_max_trade_off": self._setting_equals(settings, "maxTrade", "OFF"),
            "setting_options_profile": 1.0
            if self._has_prefix(fields, ("implied_volatility", "pcr_"))
            and self._setting_equals(settings, "maxTrade", "OFF")
            and self._setting_number(settings, "truncation", scale=0.10) >= 0.75
            else 0.0,
        }

        return ExpressionFeatures(
            expression=expression,
            values=values,
            fields=sorted(set(fields)),
            operators=sorted(set(operators)),
            windows=windows,
        )

    def feature_names(self) -> List[str]:
        """Return stable feature names."""
        return list(FEATURE_NAMES)

    def _extract_operators(self, expression: str) -> List[str]:
        raw_operators = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(", expression)
        operators = [operator.strip().rstrip("(") for operator in raw_operators]
        return [operator for operator in operators if self.schema.validate_operator(operator)]

    @staticmethod
    def _extract_windows(expression: str) -> List[int]:
        windows = []
        for match in re.findall(r"(?<=,\s)\d+(?=\))", expression):
            try:
                windows.append(int(match))
            except ValueError:
                continue
        return windows

    @staticmethod
    def _nesting_depth(expression: str) -> int:
        depth = 0
        max_depth = 0
        for char in expression:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth = max(depth - 1, 0)
        return max_depth

    @staticmethod
    def _has_price_volume_pair(fields: Sequence[str]) -> bool:
        price_fields = {"open", "close", "high", "low", "vwap", "returns"}
        volume_fields = {"volume", "adv20", "adv5"}
        field_set = set(fields)
        return bool(field_set & price_fields) and bool(field_set & volume_fields)

    @staticmethod
    def _has_fundamental_pair(fields: Sequence[str]) -> bool:
        profitability = {
            "ebit",
            "ebitda",
            "operating_income",
            "net_income",
            "gross_profit",
            "cashflow",
            "cashflow_op",
            "free_cash_flow",
            "operating_cash_flow",
        }
        scale = {"assets", "bookvalue", "equity", "sales", "revenue", "capex", "debt", "liabilities"}
        field_set = set(fields)
        return bool(field_set & profitability) and bool(field_set & scale)

    @staticmethod
    def _has_reversion_shape(expression: str, operators: Sequence[str]) -> bool:
        lowered = expression.lower()
        return "ts_mean" in operators and "ts_std_dev" in operators and any(
            token in lowered for token in ("- close", "-close", "0 -", "0-")
        )

    @staticmethod
    def _has_prefix(fields: Sequence[str], prefixes: Sequence[str]) -> bool:
        return any(field.startswith(prefix) for field in fields for prefix in prefixes)

    @staticmethod
    def _has_model_risk_data(fields: Sequence[str]) -> bool:
        model_fields = {"beta", "volatility", "momentum", "rsi", "macd", "rel_ret_all", "parkinson_volatility_120"}
        return bool(set(fields) & model_fields) or any(field.startswith("mdl") for field in fields)

    @staticmethod
    def _metric(metrics: Dict, name: str, aliases: Sequence[str] = (), scale: float = 1.0) -> float:
        value = metrics.get(name)
        for alias in aliases:
            if value is None:
                value = metrics.get(alias)
        if value is None:
            return 0.0
        try:
            return max(min(float(value) / scale, 5.0), -5.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _bool_metric(metrics: Dict, name: str, aliases: Sequence[str] = ()) -> float:
        value = metrics.get(name)
        for alias in aliases:
            if value is None:
                value = metrics.get(alias)
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, str):
            return 1.0 if value.strip().lower() in {"true", "1", "yes", "pass", "passed"} else 0.0
        return 1.0 if value else 0.0

    @classmethod
    def _check_stats(cls, metrics: Dict) -> Dict:
        raw_metrics = metrics.get("raw_metrics") if isinstance(metrics.get("raw_metrics"), dict) else metrics
        checks = metrics.get("checks")
        if checks is None and isinstance(raw_metrics, dict):
            checks = raw_metrics.get("checks")
            is_block = raw_metrics.get("is")
            if checks is None and isinstance(is_block, dict):
                checks = is_block.get("checks")

        if not isinstance(checks, list):
            passed = cls._bool_metric(metrics, "all_checks_passed", aliases=("checksPassed", "passes_checks"))
            return {
                "pass_count": int(passed),
                "fail_count": 0 if passed else 1,
                "pending_count": 0,
                "pass_rate": passed,
                "failed_names": set(),
                "pending_names": set(),
            }

        pass_count = 0
        fail_count = 0
        pending_count = 0
        failed_names = set()
        pending_names = set()
        for check in checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name") or "").strip().lower()
            result = str(check.get("result") or "").strip().lower()
            if result == "pass":
                pass_count += 1
            elif result == "fail":
                fail_count += 1
                if name:
                    failed_names.add(name)
            elif result == "pending":
                pending_count += 1
                if name:
                    pending_names.add(name)

        total = pass_count + fail_count + pending_count
        return {
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pending_count": pending_count,
            "pass_rate": round(pass_count / total, 4) if total else 0.0,
            "failed_names": failed_names,
            "pending_names": pending_names,
        }

    @staticmethod
    def _grade_score(metrics: Dict) -> float:
        raw_metrics = metrics.get("raw_metrics") if isinstance(metrics.get("raw_metrics"), dict) else metrics
        grade = None
        if isinstance(raw_metrics, dict):
            grade = raw_metrics.get("grade")
        grade = grade or metrics.get("grade")
        grade_map = {
            "SUPERIOR": 1.0,
            "EXCELLENT": 0.9,
            "GOOD": 0.75,
            "ABOVE_AVERAGE": 0.55,
            "AVERAGE": 0.25,
            "BELOW_AVERAGE": -0.10,
            "INFERIOR": -0.45,
        }
        return grade_map.get(str(grade or "").strip().upper(), 0.0)

    @staticmethod
    def _settings(metrics: Dict) -> Dict:
        settings = metrics.get("settings")
        if isinstance(settings, dict):
            return settings
        raw_metrics = metrics.get("raw_metrics") if isinstance(metrics.get("raw_metrics"), dict) else metrics
        if isinstance(raw_metrics, dict) and isinstance(raw_metrics.get("settings"), dict):
            return raw_metrics["settings"]
        return {}

    @staticmethod
    def _setting_number(settings: Dict, name: str, scale: float) -> float:
        value = settings.get(name)
        if value is None:
            return 0.0
        try:
            return max(min(float(value) / scale, 5.0), -5.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _setting_equals(settings: Dict, name: str, expected: str) -> float:
        return 1.0 if str(settings.get(name) or "").strip().upper() == expected else 0.0
