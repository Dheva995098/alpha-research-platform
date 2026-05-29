"""Research-oriented WorldQuant BRAIN simulation settings presets."""
from __future__ import annotations

from typing import Any, Dict, Optional


DEFAULT_SIMULATION_SETTINGS: Dict[str, Any] = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 10,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.01,
    "maxTrade": "ON",
    "pasteurization": "ON",
    "testPeriod": "P5Y",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}


SETTING_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {},
    "reversion": {
        "decay": 6,
        "truncation": 0.01,
        "neutralization": "SUBINDUSTRY",
    },
    "momentum": {
        "decay": 12,
        "truncation": 0.02,
        "neutralization": "SUBINDUSTRY",
    },
    "price_volume": {
        "decay": 8,
        "truncation": 0.01,
        "neutralization": "INDUSTRY",
    },
    "low_turnover": {
        "decay": 20,
        "truncation": 0.01,
        "neutralization": "SUBINDUSTRY",
    },
    "intraday": {
        "decay": 4,
        "truncation": 0.01,
        "neutralization": "SUBINDUSTRY",
    },
}


def merge_simulation_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return default settings with optional sanitized overrides."""
    settings = DEFAULT_SIMULATION_SETTINGS.copy()
    if overrides:
        settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings


def settings_preset(name: Optional[str]) -> Dict[str, Any]:
    """Return a named settings preset merged with the default settings."""
    normalized = (name or "balanced").strip().lower()
    return merge_simulation_settings(SETTING_PRESETS.get(normalized, {}))
