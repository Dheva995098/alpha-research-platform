"""User-curated alpha seeds for positive training signals.

These rows are not queued or treated as live BRAIN results. They teach the
ranker what strong structures and simulation settings look like.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from backend.core.expression_normalizer import normalize_brain_expression
from backend.core.simulation_settings import merge_simulation_settings
from backend.ml.auto_learner import AutoLearningService
from backend.models import Result


@dataclass(frozen=True)
class CuratedAlphaSeed:
    """One user-provided perfect-alpha training example."""

    seed_id: str
    expression: str
    settings: Dict[str, Any]
    note: str = "user_curated_perfect_alpha"


def _setting_token(value: str) -> str:
    normalized = str(value or "").strip().upper()
    return "NONE" if normalized in {"", "NONE", "NON"} else normalized


def settings(
    *,
    region: str = "USA",
    universe: str = "TOP3000",
    decay: int | float = 4,
    delay: int | float = 1,
    truncation: float = 0.08,
    neutralization: str = "SUBINDUSTRY",
    nan_handling: str = "OFF",
    unit_handling: str = "VERIFY",
    max_trade: Optional[str] = "OFF",
    max_position: Optional[str] = None,
    test_period: Optional[str] = "P5Y",
) -> Dict[str, Any]:
    """Build sanitized BRAIN settings for a curated seed."""
    overrides = {
        "instrumentType": "EQUITY",
        "region": region.upper(),
        "universe": universe.upper(),
        "language": "FASTEXPR",
        "decay": decay,
        "delay": delay,
        "truncation": truncation,
        "neutralization": _setting_token(neutralization),
        "pasteurization": "ON",
        "nanHandling": nan_handling.upper(),
        "unitHandling": unit_handling.upper(),
        "maxTrade": max_trade.upper() if isinstance(max_trade, str) else max_trade,
        "maxPosition": max_position.upper() if isinstance(max_position, str) else max_position,
        "testPeriod": test_period,
    }
    return merge_simulation_settings(overrides)


CURATED_PERFECT_ALPHA_SEEDS: List[CuratedAlphaSeed] = [
    CuratedAlphaSeed("ucpa001", "ts_rank(operating_income/close,120)", settings(universe="TOP1000", decay=0, truncation=0.08, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa002", "group_mean(-ts_delta(ebit*debt_lt,612),3,sector)", settings(decay=65, truncation=0.9, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa003", "ts_rank(operating_income/cap,252)+rank(fnd6_fopo/cash_st)", settings(universe="TOP200", decay=4, truncation=0.07, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa004", "ts_corr(close,open,67)", settings(decay=3, truncation=0.09)),
    CuratedAlphaSeed("ucpa005", "multiply(rank(-returns),rank(volume/adv20),filter=true)+(rank(liabilities_curr/assets))", settings(decay=5, truncation=0.01, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa006", "-ts_rank(fn_liab_fair_val_l1_a,126)", settings(decay=0, truncation=0.08)),
    CuratedAlphaSeed("ucpa007", "winsorize(ts_backfill(liabilities/assets,60))*(ts_arg_min(debt/equity,250))", settings(universe="TOP500", decay=0, truncation=0.08, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa008", "a=group_neutralize(-ts_corr(fnd6_ci, fn_assets_fair_val_l1_a, 231), industry);", settings(decay=0, truncation=0.08, neutralization="SECTOR")),
    CuratedAlphaSeed("ucpa009", "rank(fnd6_fopo/cash_st)", settings(decay=1, truncation=0.08, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa010", "group_mean(-ts_delta(ebit*debt_lt,252),1,industry)", settings(decay=50, truncation=0.8, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa011", "ts_backfill(liabilities/assets,340)", settings(universe="TOPSP500", decay=1, truncation=0.02, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa012", "rank(ts_mean(anl4_adjusted_netincome_ft,232)+liabilities_curr/assets)", settings(decay=4, truncation=0.08, neutralization="SECTOR")),
    CuratedAlphaSeed("ucpa013", "-ts_delta(close, 4) / ts_delay(close, 4)", settings(decay=5, truncation=0.09, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa014", "group_neutralize(ts_rank(operating_income/cap,150),subindustry)", settings(universe="TOPSP500", decay=0, truncation=0.01, neutralization="NONE")),
    CuratedAlphaSeed("ucpa015", "alpha=(rank(ts_sum(vec_avg(nws12_prez_1_minute),252))>0.3)?1:rank(-ts_delta(close,1))*1.367;\ntrade_when(volume>0.9*adv20,alpha,-2);", settings(universe="TOP1000", decay=4, truncation=0.08, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa016", "-open / ts_delay(open, 4) - 1", settings(decay=25, truncation=0.17, nan_handling="ON")),
    CuratedAlphaSeed("ucpa017", "ts_backfill(fnd6_drc, 252)/assets", settings(decay=0, truncation=0.08, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa018", "rank(ts_rank(operating_income/cap,252)) + group_rank(ts_rank(cashflow_op/cap,60), industry)", settings(universe="TOP1000", decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa019", "sqrt(rank(ts_mean(volume,5)/ts_mean(volume,240)))", settings(decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa020", "ts_backfill(implied_volatility_call_120/parkinson_volatility_90,2)", settings(universe="TOPSP500", decay=0, truncation=0.08)),
    CuratedAlphaSeed("ucpa021", "ts_corr(vwap, close, 20)", settings(decay=80, truncation=0.01, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa022", "ts_backfill((-scl12_buzz),200)", settings(decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa023", "group_rank(ts_rank(cashflow_op/cap,60),industry)", settings(decay=0, truncation=0.08)),
    CuratedAlphaSeed("ucpa024", "group_neutralize(-ts_corr(fnd6_ci,fnd6_cicurr,240),industry)", settings(decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa025", "anl4_adjusted_netincome_ft", settings(decay=0, truncation=0.08)),
    CuratedAlphaSeed("ucpa026", "multiply(rank(-returns),rank(volume/adv20),filter=true)", settings(decay=5, truncation=0.01, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa027", "-ts_quantile(debt,35)", settings(decay=0, truncation=0.01, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa028", "sqrt(rank(ts_mean(volume,5)/ts_mean(volume,120)))", settings(universe="ILLIQUID_MINVOL1M", decay=4, truncation=0.08, neutralization="FAST FACTORS")),
    CuratedAlphaSeed("ucpa029", "-ts_delta(close,5)", settings(region="GLB", universe="MINVOL1M", decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa030", "group_neutralize(rank(snt_social_value + snt_social_volume),sector)", settings(universe="ILLIQUID_MINVOL1M", decay=0, delay=0, truncation=0.01, neutralization="FAST FACTORS", unit_handling="VERIFY", max_trade=None)),
    CuratedAlphaSeed("ucpa031", "-rank(ts_arg_max(oth432_earnings_yield1,100))", settings(universe="ILLIQUID_MINVOL1M", decay=0, delay=0, truncation=0.01, neutralization="STATISTICAL")),
    CuratedAlphaSeed("ucpa032", "rank(ts_delta(close,5)) * -1", settings(decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa033", "rank(ts_delta(close,5)) * -1", settings(universe="ILLIQUID_MINVOL1M", decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa034", "-ts_delta(close,10)", settings(decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa035", "-ts_delta(close,5)", settings(decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa036", "-rank(ts_arg_max(oth432_earnings_yield1,50))", settings(universe="ILLIQUID_MINVOL1M", decay=0, delay=0, truncation=0.01, neutralization="STATISTICAL")),
    CuratedAlphaSeed("ucpa037", "ts_backfill(ts_rank(fnd6_fopo/debt_lt,440),400)", settings(decay=0, truncation=0.08, neutralization="SECTOR", max_position="OFF")),
    CuratedAlphaSeed("ucpa038", "-(close - ts_mean(close,5))", settings(decay=4, truncation=0.08, test_period="P1Y")),
    CuratedAlphaSeed("ucpa039", "-rank(days_from_last_change(pv13_com_page_rank)) * ts_rank(ts_delta(close,11),90)", settings(decay=4, truncation=0.08)),
    CuratedAlphaSeed("ucpa040", "-ts_delta(close,3)", settings(decay=14, truncation=0.01, max_position="OFF")),
    CuratedAlphaSeed("ucpa041", "-rank(days_from_last_change(pv13_com_page_rank))*ts_rank(returns,10)", settings(decay=6, truncation=0.08)),
    CuratedAlphaSeed("ucpa042", "-rank(ts_mean(ts_delta(close,7),5))", settings(decay=117, truncation=0.01, max_position="OFF")),
    CuratedAlphaSeed("ucpa043", "ts_corr(scale(ts_rank(fn_oth_income_loss_tb_translation_and_tax_translation_adj,8)), scale(ts_rank(income_tax,8)), 5)", settings(decay=50, truncation=0.09, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa044", "trade_when(pcr_oi_180 < 0.4, (implied_volatility_call_180 - implied_volatility_put_180), -1)", settings(decay=5, truncation=0.009, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa045", "ts_delta((vwap-close)/vwap,10)", settings(decay=25, truncation=0.07, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa046", "group_neutralize(rank(sqrt(rank(ts_mean(volume,5)/ts_mean(volume,250)))) * winsorize(ts_backfill(liabilities/assets,120)), industry)", settings(decay=4, truncation=0.01, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa047", "alpha = group_rank(fnd2_ebitdm, industry) - group_rank(fnd2_ebitfr, industry);\ngroup_rank(fn_assets_fair_val_a, industry) > 0.2 ? alpha : -alpha", settings(universe="TOP200", decay=30, truncation=0.08)),
    CuratedAlphaSeed("ucpa048", "ts_backfill((-scl12_buzz),10)", settings(decay=11, truncation=0.08, neutralization="MARKET", unit_handling="VERIFY", max_trade=None)),
    CuratedAlphaSeed("ucpa049", "ts_rank(operating_income/cap,252)", settings(universe="TOP1000", decay=0, truncation=0.08, max_trade=None)),
    CuratedAlphaSeed("ucpa050", "ts_corr(scale(ts_rank(fnd2_ebitdm,80)), scale(ts_rank(bookvalue_ps/income,80)), 82)", settings(decay=300, truncation=0.4, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa051", "rank(ts_rank(operating_income/cap,252)) + rank(liabilities/assets)", settings(decay=0, truncation=0.01)),
    CuratedAlphaSeed("ucpa052", "ts_corr(scale(ts_rank(fn_comp_options_out_intrinsic_value_a,20)), scale(ts_rank(income,20)), 5)", settings(decay=400, truncation=0.7, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa053", "ts_corr(rank(ts_delta(ebit/interest_expense,120)), rank(ts_delta(inventory/liabilities_curr,120)), 6)", settings(decay=60, truncation=0.02)),
    CuratedAlphaSeed("ucpa054", "ts_decay_linear(ts_delta(implied_volatility_put_60,25)^0.26)", settings(decay=12, truncation=0.005, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa055", "rank(ts_sum(vec_avg(nws12_afterh15z_s1),400)) > 0.4 ? power(rank(ts_mean(volume,5) / ts_mean(volume,252)),0.5) * -1 : power(rank(ts_mean(volume,5) / ts_mean(volume,252)),0.5)", settings(decay=40, truncation=0.08, neutralization="INDUSTRY")),
    CuratedAlphaSeed("ucpa056", "ts_corr(scale(ts_delta(est_bookvalue_ps/income,65)),scale(ts_delta(cashflow_op/income,65)),5)", settings(decay=20, truncation=0.0009, neutralization="SECTOR")),
    CuratedAlphaSeed("ucpa057", "trade_when(volume > adv20, -1 * rank(ts_delta(close, 4)), -1)", settings(decay=2, truncation=0.0001, neutralization="MARKET", unit_handling="VERIFY", max_trade=None)),
    CuratedAlphaSeed("ucpa058", "ts_corr(scale(ts_rank(cashflow,15)),scale(ts_rank(debt/equity,15)),3)", settings(decay=170, truncation=0.7, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa059", "ts_corr(rank(depr_amort),rank(cashflow_fin),5)", settings(universe="TOP500", decay=100, truncation=0.04)),
    CuratedAlphaSeed("ucpa060", "rank(ts_delta(income/debt,122))", settings(decay=5, truncation=0.007, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa061", "trade_when(pcr_oi_270 < 1, implied_volatility_call_270 - implied_volatility_put_270, -1)", settings(decay=4, truncation=0.08, neutralization="INDUSTRY", unit_handling="VERIFY", max_trade=None)),
    CuratedAlphaSeed("ucpa062", "ts_mean(trade_when(md153_fc6_4year3, ts_arg_max(ts_arg_min(md153_fc5_7year_3),181), -1),301)", settings(decay=0, truncation=0.13, neutralization="MARKET")),
    CuratedAlphaSeed("ucpa063", "ts_arg_max(winsorize(ts_backfill(vec_avg(anl4_fsguidancebasicqf4_item),120),std=4),120)", settings(decay=5, truncation=0.57)),
    CuratedAlphaSeed("ucpa064", "event = volume > adv20;\nalpha_1 = -(ts_delta(close,5) / ts_delay(close,5));\nalpha_2 = -(ts_delta(close,5) / ts_delay(close,5));\nif_else(event,alpha_1,alpha_2)", settings(decay=15, truncation=0.2)),
]


def upsert_curated_perfect_alpha_seeds(
    db: Session,
    seeds: Iterable[CuratedAlphaSeed] = CURATED_PERFECT_ALPHA_SEEDS,
    *,
    train: bool = True,
) -> Dict[str, Any]:
    """Upsert curated positive examples and optionally retrain the ranker."""
    seeded = 0
    updated = 0
    skipped = 0
    now = datetime.now(timezone.utc).isoformat()
    seen_keys: set[str] = set()
    for seed in seeds:
        expression = normalize_brain_expression(seed.expression)
        if _is_ambiguous_expression(expression):
            skipped += 1
            continue

        seed_key = _seed_key(expression, seed.settings)
        if seed_key in seen_keys:
            skipped += 1
            continue
        seen_keys.add(seed_key)

        alpha_id = f"user-perfect-seed-{seed.seed_id}"
        result = db.query(Result).filter(Result.brain_alpha_id == alpha_id).first()
        if result is None:
            result = Result(brain_alpha_id=alpha_id)
            db.add(result)
            seeded += 1
        else:
            updated += 1

        metrics = _proxy_positive_metrics(seed)
        result.expression = expression
        result.sharpe = metrics["sharpe"]
        result.fitness = metrics["fitness"]
        result.turnover = metrics["turnover"]
        result.self_correlation = metrics["self_correlation"]
        result.all_checks_passed = True
        result.final_score = metrics["final_score"]
        result.human_approved = True
        result.submitted_to_brain = False
        result.raw_metrics = {
            "source": "training_seed",
            "seed_id": seed.seed_id,
            "seed_kind": "user_curated_perfect_alpha",
            "source_name": "user_pasted_perfect_alphas_2026_05_28",
            "observed_at": now,
            "settings": seed.settings,
            "grade": "USER_PERFECT",
            "all_checks_passed": True,
            "label_policy": "positive_user_curated_training_signal",
            "metrics_policy": "proxy_positive_label_no_live_metrics",
            "copy_policy": "training_signal_only_do_not_copy",
            "checks": [
                {"name": "USER_CURATED_PERFECT_ALPHA", "result": "PASS"},
                {"name": "COPY_POLICY_TRAINING_ONLY", "result": "PASS"},
                {"name": "SETTINGS_PROVIDED", "result": "PASS"},
            ],
            "proxy_metrics": metrics,
        }

    db.commit()
    summary = {
        "seeded": seeded,
        "updated": updated,
        "skipped": skipped,
        "total": seeded + updated,
        "trained": False,
        "example_count": None,
        "positive_count": None,
        "negative_count": None,
        "training_seed_count": None,
    }
    if train:
        learning = AutoLearningService(db).run_once(limit=5000, min_examples=5)
        training = learning.get("training") or {}
        summary.update(
            {
                "trained": bool(learning.get("trained")),
                "example_count": training.get("example_count"),
                "positive_count": training.get("positive_count"),
                "negative_count": training.get("negative_count"),
                "training_seed_count": learning.get("training_seed_count"),
            }
        )
    return summary


def _is_ambiguous_expression(expression: str) -> bool:
    text = expression.strip()
    if not text:
        return True
    if "..." in text or "blur" in text.lower():
        return True
    return "?" in text and ":" not in text


def _seed_key(expression: str, seed_settings: Dict[str, Any]) -> str:
    rendered = repr((expression, sorted(seed_settings.items())))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _proxy_positive_metrics(seed: CuratedAlphaSeed) -> Dict[str, float]:
    expression = seed.expression.lower()
    windows = [int(item) for item in re.findall(r"(?<=,)\s*(\d+)(?=\))", expression)]
    complexity = min(len(expression) / 650.0, 0.45)
    multi_data_bonus = 0.10 if len(set(re.findall(r"\b[a-z][a-z0-9_]*\b", expression))) >= 6 else 0.0
    alt_bonus = 0.12 if any(token in expression for token in ("implied_volatility", "pcr_", "nws", "scl", "snt_", "pv13", "anl4")) else 0.0
    long_window_bonus = 0.08 if windows and max(windows) >= 120 else 0.0
    sharpe = round(min(2.55, 1.55 + complexity + multi_data_bonus + alt_bonus + long_window_bonus), 4)
    fitness = round(min(2.10, 1.12 + complexity * 0.80 + multi_data_bonus + alt_bonus), 4)
    truncation = float(seed.settings.get("truncation") or 0.08)
    turnover = round(max(0.08, min(0.64, 0.24 + truncation * 0.22)), 4)
    self_correlation = round(max(0.08, min(0.38, 0.16 + complexity * 0.20)), 4)
    final_score = round(sharpe * 0.45 + fitness * 0.45 + 0.10, 4)
    return {
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
        "self_correlation": self_correlation,
        "final_score": final_score,
    }
