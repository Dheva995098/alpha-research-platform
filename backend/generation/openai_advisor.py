"""Optional OpenAI-assisted candidate critique and reranking."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.generation.candidates import AlphaCandidate
from backend.generation.model_adapter import OpenAIAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlphaAdvice:
    """LLM critique for one alpha candidate."""

    expression: str
    score: float
    rationale: str
    risk_flags: Tuple[str, ...] = ()


class OpenAIAlphaAdvisor:
    """Ask an OpenAI model to score candidate robustness and novelty."""

    def __init__(self, adapter: Optional[OpenAIAdapter] = None):
        self.adapter = adapter or OpenAIAdapter()

    def advise(
        self,
        candidates: Sequence[AlphaCandidate],
        *,
        settings: Optional[Dict[str, Any]] = None,
        focus: Optional[str] = None,
        dataset_id: Optional[str] = None,
        limit: int = 30,
        recent_failures: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[AlphaAdvice]:
        """Return bounded advice for candidates.

        Only expressions, candidate metadata, and simulation-setting hints are
        sent. Account credentials and BRAIN session data never belong here.

        ``recent_failures`` (optional) are prior attempts that fell short; they are
        injected as negative examples so the model steers candidates away from the
        same failure modes (self-improving loop Pattern C).
        """
        scoped = list(candidates[: max(1, min(limit, 50))])
        if not scoped:
            return []

        prompt = self._prompt(
            scoped,
            settings=settings or {},
            focus=focus,
            dataset_id=dataset_id,
            recent_failures=recent_failures,
        )
        text = self.adapter.generate(prompt, temperature=0.1, max_tokens=1800)
        return self._parse_advice(text)

    @staticmethod
    def _prompt(
        candidates: Sequence[AlphaCandidate],
        *,
        settings: Dict[str, Any],
        focus: Optional[str],
        dataset_id: Optional[str],
        recent_failures: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        rows = [
            {
                "expression": candidate.expression,
                "strategy": candidate.strategy,
                "source_fields": list(candidate.source_fields),
                "dataset_ids": list(candidate.dataset_ids),
                "local_score": candidate.score,
            }
            for candidate in candidates
        ]
        failure_rows = []
        for item in (recent_failures or [])[:5]:
            if not isinstance(item, dict):
                continue
            failure_rows.append(
                {
                    "expression": str(item.get("expression") or ""),
                    "sharpe": item.get("sharpe"),
                    "fitness": item.get("fitness"),
                    "turnover": item.get("turnover"),
                    "self_correlation": item.get("self_correlation"),
                    "issues": list(item.get("failures") or []),
                }
            )
        payload = {
            "task": "Score WorldQuant BRAIN FASTEXPR alpha candidates for research triage.",
            "focus": focus,
            "dataset_id": dataset_id,
            "settings": settings,
            "candidates": rows,
            "prior_failures_to_avoid": failure_rows,
        }
        failure_hint = (
            "Some prior attempts already fell short; they are in `prior_failures_to_avoid` "
            "with their metrics and failed checks. Down-score candidates that repeat those "
            "shapes or failure modes, and up-score genuinely different angles.\n"
            if failure_rows
            else ""
        )
        return (
            "You are a cautious quantitative research reviewer for WorldQuant BRAIN FASTEXPR.\n"
            "Score each candidate from 0.0 to 1.0 for likely robustness, data-fit, and originality.\n"
            "Penalize obvious overfitting, fragile constants, excessive complexity, weak data/settings fit, "
            "and formulas that look like direct public-copy patterns.\n"
            f"{failure_hint}"
            "Do not invent BRAIN results. Do not claim a candidate will pass. Return JSON only.\n"
            "JSON schema: {\"advice\":[{\"expression\":\"...\",\"score\":0.0,"
            "\"rationale\":\"short reason\",\"risk_flags\":[\"...\"]}]}.\n\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=True)}"
        )

    @staticmethod
    def _parse_advice(text: str) -> List[AlphaAdvice]:
        payload = OpenAIAlphaAdvisor._json_payload(text)
        rows = payload.get("advice") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return []

        advice: List[AlphaAdvice] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            expression = str(row.get("expression") or "").strip()
            if not expression:
                continue
            try:
                score = float(row.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            risk_flags = row.get("risk_flags") or []
            if not isinstance(risk_flags, list):
                risk_flags = [str(risk_flags)]
            advice.append(
                AlphaAdvice(
                    expression=expression,
                    score=max(0.0, min(score, 1.0)),
                    rationale=str(row.get("rationale") or "OpenAI research critique").strip()[:240],
                    risk_flags=tuple(str(item).strip()[:80] for item in risk_flags if str(item).strip()),
                )
            )
        return advice

    @staticmethod
    def _json_payload(text: str) -> Any:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.debug("OpenAI advice response was not parseable JSON: %s", cleaned[:500])
            return {}


def apply_openai_advice(
    candidates: Sequence[AlphaCandidate],
    *,
    settings: Optional[Dict[str, Any]] = None,
    focus: Optional[str] = None,
    dataset_id: Optional[str] = None,
    weight: float = 0.25,
) -> Tuple[List[AlphaCandidate], Dict[str, Any]]:
    """Return candidates reranked with optional OpenAI advice.

    Any adapter/key/network failure falls back to the original order.
    """
    if not candidates:
        return [], {"openai_assist": False, "reason": "no_candidates"}

    try:
        advice = OpenAIAlphaAdvisor().advise(
            list(candidates),
            settings=settings,
            focus=focus,
            dataset_id=dataset_id,
        )
    except Exception as exc:
        logger.info("OpenAI alpha advice unavailable: %s", exc)
        return list(candidates), {"openai_assist": False, "reason": str(exc)}

    by_expression = {item.expression: item for item in advice}
    if not by_expression:
        return list(candidates), {"openai_assist": False, "reason": "empty_advice"}

    blended: List[AlphaCandidate] = []
    blend_weight = max(0.0, min(weight, 0.75))
    for candidate in candidates:
        item = by_expression.get(candidate.expression)
        if item is None:
            blended.append(candidate)
            continue
        score = round(candidate.score * (1.0 - blend_weight) + item.score * blend_weight, 4)
        rationale = candidate.rationale
        if item.rationale:
            rationale = f"{rationale} OpenAI: {item.rationale}"
        blended.append(
            AlphaCandidate(
                expression=candidate.expression,
                strategy=candidate.strategy,
                source_fields=candidate.source_fields,
                dataset_ids=candidate.dataset_ids,
                operators=candidate.operators,
                rationale=rationale,
                score=score,
            )
        )

    blended.sort(key=lambda candidate: candidate.score, reverse=True)
    return blended, {
        "openai_assist": True,
        "advised_count": len(by_expression),
        "model_role": "candidate_research_reranker",
    }
