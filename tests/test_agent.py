import os
from types import SimpleNamespace

from backend.generation.agent_controller import AgentController
from backend.generation import model_adapter
from backend.generation import openai_advisor
from backend.generation.candidates import AlphaCandidate
from backend.generation.model_adapter import LocalRuleAdapter
from backend.generation.openai_advisor import AlphaAdvice, OpenAIAlphaAdvisor


def test_agent_produces_plan():
    repo_root = os.path.abspath(".")
    agent = AgentController(adapter=LocalRuleAdapter(), repo_path=repo_root)
    result = agent.run(user_goal="Make the project Codex-ready and model-agnostic")
    assert "plan_file" in result
    assert isinstance(result["steps"], list)
    assert len(result["steps"]) >= 1


def test_openai_adapter_uses_modern_client(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="1. Inspect data\n2. Train model")
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(model_adapter, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    adapter = model_adapter.OpenAIAdapter(api_key="test-key", model="test-model")

    assert adapter.generate("make a plan").startswith("1. Inspect")


def test_openai_advisor_parses_and_reranks_with_fake_adapter(monkeypatch):
    class FakeAdapter:
        def generate(self, prompt, **kwargs):
            assert "FASTEXPR" in prompt
            return {
                "advice": [
                    {
                        "expression": "rank(volume)",
                        "score": 0.95,
                        "rationale": "Better data-fit",
                        "risk_flags": [],
                    },
                    {
                        "expression": "rank(close)",
                        "score": 0.10,
                        "rationale": "Too simple",
                        "risk_flags": ["fragile"],
                    },
                ]
            }.__repr__().replace("'", '"')

    advisor = OpenAIAlphaAdvisor(adapter=FakeAdapter())
    candidates = [
        AlphaCandidate(expression="rank(close)", strategy="price_volume", score=0.80),
        AlphaCandidate(expression="rank(volume)", strategy="price_volume", score=0.50),
    ]

    advice = advisor.advise(candidates)

    class FakeAdvisor:
        def advise(self, candidates, **kwargs):
            return [
                AlphaAdvice(expression="rank(volume)", score=0.95, rationale="Better data-fit"),
                AlphaAdvice(expression="rank(close)", score=0.10, rationale="Too simple"),
            ]

    monkeypatch.setattr(openai_advisor, "OpenAIAlphaAdvisor", FakeAdvisor)
    reranked, metadata = openai_advisor.apply_openai_advice(
        candidates,
        settings={"region": "USA"},
        weight=0.50,
    )

    assert len(advice) == 2
    assert advice[0].expression == "rank(volume)"
    assert metadata["openai_assist"] is True
    assert [candidate.expression for candidate in reranked] == ["rank(volume)", "rank(close)"]


if __name__ == "__main__":
    print("Running agent test")
    res = test_agent_produces_plan()
    print("OK")
