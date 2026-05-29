"""
AgentController: analyze repository, build prompt, ask model, and write plan.
"""
import os
import json
import re
from typing import List, Dict

from .model_adapter import ModelAdapter, LocalRuleAdapter


class AgentController:
    """High-level agent controller used by Codex/agents to continue the project.

    - `analyze_repo()` collects a lightweight summary of repository files and TODOs
    - `build_prompt()` composes a prompt for the configured model adapter
    - `plan_next_steps()` asks the adapter to return a short prioritized plan
    - `write_plan()` writes `agent_plan.json` for human review
    """

    def __init__(self, adapter: ModelAdapter = None, repo_path: str = "."):
        self.repo_path = os.path.abspath(repo_path)
        self.adapter = adapter or LocalRuleAdapter()

    def analyze_repo(self) -> Dict:
        summary = {
            "repo_path": self.repo_path,
            "files": [],
            "python_files": 0,
            "has_backend": False,
            "has_frontend": False,
            "todo_lines": [],
        }

        for root, dirs, files in os.walk(self.repo_path):
            # skip large or irrelevant folders
            if ".git" in root or "node_modules" in root:
                continue
            for f in files:
                path = os.path.join(root, f)
                rel = os.path.relpath(path, self.repo_path)
                summary["files"].append(rel)
                if f.endswith(".py"):
                    summary["python_files"] += 1
                if rel.startswith(os.path.join("backend") + os.sep) or rel.startswith("backend/"):
                    summary["has_backend"] = True
                if rel.startswith(os.path.join("frontend") + os.sep) or rel.startswith("frontend/"):
                    summary["has_frontend"] = True

                if f.endswith((".py", ".md", ".js", ".ts", ".tsx", ".json", ".txt")):
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            for i, line in enumerate(fh, start=1):
                                if "TODO" in line or "FIXME" in line:
                                    summary["todo_lines"].append({"file": rel, "line": i, "text": line.strip()})
                    except Exception:
                        # skip files we cannot read
                        pass

        return summary

    def build_prompt(self, repo_summary: Dict, user_goal: str = "") -> str:
        # Try to load a template if present
        tpl_path = os.path.join(self.repo_path, "backend", "generation", "prompts", "agent_prompt.md")
        template = ""
        if os.path.exists(tpl_path):
            try:
                with open(tpl_path, "r", encoding="utf-8") as fh:
                    template = fh.read()
            except Exception:
                template = ""

        prompt_lines = []
        prompt_lines.append("Repository summary:")
        prompt_lines.append(f"- path: {repo_summary.get('repo_path')}")
        prompt_lines.append(f"- files: {len(repo_summary.get('files', []))}")
        prompt_lines.append(f"- python_files: {repo_summary.get('python_files')}")
        prompt_lines.append(f"- has_backend: {repo_summary.get('has_backend')}")
        prompt_lines.append(f"- has_frontend: {repo_summary.get('has_frontend')}")
        prompt_lines.append(f"- todos_count: {len(repo_summary.get('todo_lines', []))}")
        prompt_lines.append("")
        if user_goal:
            prompt_lines.append(f"User goal: {user_goal}")
            prompt_lines.append("")

        if template:
            prompt_lines.append("Template:")
            prompt_lines.append(template)
            prompt_lines.append("")

        prompt_lines.append("TASK: Provide a concise, prioritized list of next implementation steps (max 8). For each step provide: action, files to modify/create (paths), and estimated time. Keep steps actionable and non-destructive. Output as a numbered list.")

        return "\n".join(prompt_lines)

    def plan_next_steps(self, user_goal: str = "") -> List[Dict]:
        summary = self.analyze_repo()
        prompt = self.build_prompt(summary, user_goal)
        response = self.adapter.generate(prompt)

        steps: List[Dict] = []
        for line in response.splitlines():
            m = re.match(r"^\s*\d+\.\s*(.+)$", line)
            if m:
                text = m.group(1).strip()
                steps.append({"text": text})

        if not steps:
            # fallback: split into non-empty lines
            for line in response.splitlines():
                if line.strip():
                    steps.append({"text": line.strip()})

        return steps

    def write_plan(self, plan: List[Dict], filename: str = None) -> str:
        if filename is None:
            filename = os.path.join(self.repo_path, "backend", "generation", "agent_plan.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump({"plan": plan}, fh, indent=2)
        return filename

    def run(self, user_goal: str = "") -> Dict:
        plan = self.plan_next_steps(user_goal)
        path = self.write_plan(plan)
        return {"plan_file": path, "steps": plan}
