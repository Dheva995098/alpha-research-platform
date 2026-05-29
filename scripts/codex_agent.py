"""
CLI entry to run the Codex agent locally.
Usage:
  python scripts/codex_agent.py --model local --goal "Make project model-agnostic for Codex"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from backend.generation.model_adapter import LocalRuleAdapter, OpenAIAdapter
from backend.generation.agent_controller import AgentController


def main():
    parser = argparse.ArgumentParser(description="Run the Codex agent to plan next steps")
    parser.add_argument("--model", choices=["local", "openai"], default="local")
    parser.add_argument("--goal", default="", help="Short user goal to guide the agent")
    args = parser.parse_args()

    if args.model == "openai":
        # Attempt to create an OpenAI adapter; may raise if API not available
        try:
            adapter = OpenAIAdapter()
        except Exception as e:
            print("OpenAI adapter unavailable, falling back to local adapter:", e)
            adapter = LocalRuleAdapter()
    else:
        adapter = LocalRuleAdapter()

    agent = AgentController(adapter=adapter, repo_path=os.getcwd())
    result = agent.run(user_goal=args.goal)

    print("Plan written to:", result["plan_file"])
    print("Steps:")
    for i, s in enumerate(result["steps"], start=1):
        print(f"{i}. {s['text']}")


if __name__ == "__main__":
    main()
