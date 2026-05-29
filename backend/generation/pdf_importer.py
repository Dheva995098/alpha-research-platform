"""Import submitted alpha examples from text-based PDF exports."""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import List, Optional


ALPHA_OPERATOR_HINTS = (
    "rank(",
    "group_rank(",
    "group_neutralize(",
    "ts_rank(",
    "ts_corr(",
    "ts_mean(",
    "ts_zscore(",
    "ts_decay_linear(",
    "trade_when(",
    "winsorize(",
)


@dataclass(frozen=True)
class ImportedAlpha:
    expression: str
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    self_correlation: Optional[float] = None
    passes_checks: bool = True


@dataclass(frozen=True)
class PDFImportResult:
    examples: List[ImportedAlpha]
    page_count: int
    extracted_text_chars: int
    warnings: List[str]


def extract_alpha_examples_from_pdf(path: str | Path) -> PDFImportResult:
    """Extract candidate expressions and nearby metrics from a text PDF."""
    path = Path(path)
    warnings: List[str] = []
    try:
        from pypdf import PdfReader
    except ImportError:
        return PDFImportResult(
            examples=[],
            page_count=0,
            extracted_text_chars=0,
            warnings=["pypdf is not installed; install requirements.txt and retry"],
        )

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages)
    if not text.strip():
        warnings.append("PDF has no extractable text; it likely needs OCR before import")
        return PDFImportResult([], len(reader.pages), 0, warnings)

    examples = _extract_examples(text)
    if not examples:
        warnings.append("No alpha expressions found in extracted PDF text")
    return PDFImportResult(examples, len(reader.pages), len(text), warnings)


def _extract_examples(text: str) -> List[ImportedAlpha]:
    expressions = []
    seen = set()
    for line in text.splitlines():
        cleaned = _clean_line(line)
        if not cleaned or not any(hint in cleaned for hint in ALPHA_OPERATOR_HINTS):
            continue
        expression = _expression_span(cleaned)
        if not expression or expression in seen:
            continue
        seen.add(expression)
        expressions.append(
            ImportedAlpha(
                expression=expression,
                sharpe=_nearby_metric(cleaned, "sharpe"),
                fitness=_nearby_metric(cleaned, "fitness"),
                turnover=_nearby_metric(cleaned, "turnover"),
                self_correlation=_nearby_metric(cleaned, "self"),
                passes_checks="fail" not in cleaned.lower(),
            )
        )
    return expressions


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _expression_span(line: str) -> Optional[str]:
    starts = [line.find(hint) for hint in ALPHA_OPERATOR_HINTS if line.find(hint) >= 0]
    if not starts:
        return None
    start = min(starts)
    depth = 0
    end = None
    for index, char in enumerate(line[start:], start=start):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        return line[start:].strip()
    return line[start:end].strip()


def _nearby_metric(text: str, name: str) -> Optional[float]:
    pattern = rf"{name}[^\-\d]{{0,12}}(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
