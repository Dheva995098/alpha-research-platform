"""Shared candidate types for generated alpha expressions."""
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class AlphaCandidate:
    """A generated FASTEXPR alpha candidate plus lightweight metadata."""

    expression: str
    strategy: str
    source_fields: Tuple[str, ...] = field(default_factory=tuple)
    dataset_ids: Tuple[str, ...] = field(default_factory=tuple)
    operators: Tuple[str, ...] = field(default_factory=tuple)
    rationale: str = ""
    score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "expression": self.expression,
            "strategy": self.strategy,
            "source_fields": list(self.source_fields),
            "dataset_ids": list(self.dataset_ids),
            "operators": list(self.operators),
            "rationale": self.rationale,
            "score": self.score,
        }
