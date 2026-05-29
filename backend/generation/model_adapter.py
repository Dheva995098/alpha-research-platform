"""
Model adapter abstraction and simple implementations.
Provides a Local rule-based adapter (fallback) and an OpenAI adapter (optional).
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from backend.config import settings

# Try to import openai lazily
try:
    import openai
except Exception:
    openai = None


class ModelAdapter(ABC):
    """Abstract adapter interface for text generation models."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text for the given prompt."""
        raise NotImplementedError()


class LocalRuleAdapter(ModelAdapter):
    """Fallback lightweight adapter that generates deterministic plans.

    Used when no LLM credentials are available or for fast offline testing.
    """

    def generate(self, prompt: str, **kwargs) -> str:
        # Heuristic-based plan generation — deterministic and safe.
        out_lines = []
        if "Alpha Research" in prompt or "BRAIN" in prompt or "Alpha" in prompt:
            out_lines = [
                "1. Implement `ModelAdapter` abstraction and register adapters (backend/generation/model_adapter.py)",
                "2. Implement `AgentController` to analyze repo and produce prioritized steps (backend/generation/agent_controller.py)",
                "3. Add CLI `scripts/codex_agent.py` to run the agent locally and write a plan file",
                "4. Add prompt templates at backend/generation/prompts/agent_prompt.md",
                "5. Add unit tests for agent (tests/test_agent.py)",
                "6. Human review the generated plan before any destructive actions"
            ]
        else:
            # Generic repository plan
            out_lines = [
                "1. Scan repository structure and list missing pieces",
                "2. Produce a short prioritized todo list (3-8 items)",
                "3. Write the plan to backend/generation/agent_plan.json",
            ]
        return "\n".join(out_lines)


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI (or compatible) APIs.

    This is a thin wrapper. The system will default to `LocalRuleAdapter`
    when no API key is configured.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if api_key is None:
            api_key = settings.openai_api_key
        if openai is None:
            raise RuntimeError("openai package not available")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.client = openai.OpenAI(api_key=api_key) if hasattr(openai, "OpenAI") else openai
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512, **kwargs) -> str:
        if hasattr(self.client, "responses"):
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs,
            )
            return self._response_text(resp)

        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return self._response_text(resp)

        raise RuntimeError("Installed openai package does not expose a supported text generation client")

    @staticmethod
    def _response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                first = choices[0]
                message = first.get("message") if isinstance(first, dict) else None
                if isinstance(message, dict):
                    return str(message.get("content") or "")
                return str(first.get("text") or "")

        choices = getattr(response, "choices", None) or []
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            content = getattr(message, "content", None)
            if content is not None:
                return str(content)
            text = getattr(first, "text", None)
            if text is not None:
                return str(text)

        return str(response)
