"""BRAIN expression compatibility helpers."""
from __future__ import annotations

import re
from typing import Optional


OPERATOR_ALIASES = {
    "ts_std": "ts_std_dev",
}


def normalize_brain_expression(expression: Optional[str]) -> str:
    """Return a BRAIN-compatible expression string with known aliases fixed."""
    text = (expression or "").strip()
    for source, target in OPERATOR_ALIASES.items():
        text = re.sub(rf"\b{re.escape(source)}\s*\(", f"{target}(", text)
    return _strip_winsorize_positional_std(text)


def _strip_winsorize_positional_std(expression: str) -> str:
    """Convert winsorize(x, 4) to winsorize(x) for BRAIN variants with one input."""
    text = expression
    search_from = 0
    while True:
        match = re.search(r"\bwinsorize\s*\(", text[search_from:], flags=re.IGNORECASE)
        if not match:
            return text
        start = search_from + match.start()
        open_index = search_from + match.end() - 1
        close_index = _matching_paren(text, open_index)
        if close_index is None:
            return text
        args = text[open_index + 1 : close_index]
        comma_index = _top_level_comma(args)
        if comma_index is None:
            search_from = close_index + 1
            continue
        first_arg = args[:comma_index].strip()
        text = f"{text[:open_index + 1]}{first_arg}{text[close_index:]}"
        search_from = open_index + len(first_arg) + 2


def _matching_paren(text: str, open_index: int) -> Optional[int]:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _top_level_comma(text: str) -> Optional[int]:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            return index
    return None


def clean_brain_error_message(message: Optional[str]) -> Optional[str]:
    """Remove noisy BRAIN markup from error messages shown in the UI."""
    if not message:
        return None

    text = str(message)
    text = re.sub(
        r"<linkToCommonErrorMessages>.*?</linkToCommonErrorMessages>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text or None
