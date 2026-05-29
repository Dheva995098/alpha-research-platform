"""Expression normalization and deduplication utilities."""
from dataclasses import dataclass, field
import hashlib
import re
from typing import Dict, Iterable, List, Optional, Tuple


def normalize_expression(expression: str) -> str:
    """Normalize an expression enough to catch exact structural duplicates."""
    normalized = expression.strip().lower()
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r",+", ",", normalized)
    return normalized


def expression_signature(expression: str) -> str:
    """Return a stable signature for the normalized expression."""
    normalized = normalize_expression(expression)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DuplicateExpression:
    """A duplicate expression and the earlier expression it matched."""

    expression: str
    duplicate_of: str
    signature: str


@dataclass
class DeduplicationResult:
    """Result of a deduplication pass."""

    unique: List[str] = field(default_factory=list)
    duplicates: List[DuplicateExpression] = field(default_factory=list)


class ExpressionDeduplicator:
    """Track generated alpha expressions and keep the first occurrence."""

    def __init__(self, existing: Optional[Iterable[str]] = None):
        self._seen: Dict[str, str] = {}
        if existing:
            for expression in existing:
                self.add(expression)

    def add(self, expression: str) -> Tuple[bool, Optional[str], str]:
        """Add expression and return (is_unique, duplicate_of, signature)."""
        signature = expression_signature(expression)
        duplicate_of = self._seen.get(signature)
        if duplicate_of is not None:
            return False, duplicate_of, signature

        self._seen[signature] = expression
        return True, None, signature

    def dedupe(self, expressions: Iterable[str]) -> DeduplicationResult:
        """Deduplicate a sequence while preserving the first unique order."""
        result = DeduplicationResult()
        for expression in expressions:
            is_unique, duplicate_of, signature = self.add(expression)
            if is_unique:
                result.unique.append(expression)
            else:
                result.duplicates.append(
                    DuplicateExpression(
                        expression=expression,
                        duplicate_of=duplicate_of or expression,
                        signature=signature,
                    )
                )
        return result
