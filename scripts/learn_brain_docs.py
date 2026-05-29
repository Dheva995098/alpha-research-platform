"""Extract readable WorldQuant BRAIN docs into reusable local research notes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.generation.pdf_importer import ALPHA_OPERATOR_HINTS
from backend.models import LeaderboardAlpha, SessionLocal, init_db


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default=r"C:\Users\shree\Downloads\wd_*.pdf")
    parser.add_argument("--out", default="backend/generation/learned_brain_docs.json")
    parser.add_argument("--import-examples", action="store_true")
    parser.add_argument("--clean-imported", action="store_true")
    args = parser.parse_args()

    pdfs = sorted(Path().glob(args.glob) if not Path(args.glob).drive else Path(args.glob).parent.glob(Path(args.glob).name))
    docs = []
    expressions = []
    for pdf in pdfs:
        text = _pdf_text(pdf)
        docs.append({"path": str(pdf), "pages": _page_count(pdf), "chars": len(text)})
        expressions.extend(_extract_expression_lines(text))

    unique_expressions = sorted(set(expressions))
    payload = {
        "doc_count": len(docs),
        "text_chars": sum(item["chars"] for item in docs),
        "docs": docs,
        "expression_count": len(unique_expressions),
        "expressions": unique_expressions,
        "learned_principles": [
            "Use Delay 1 for conservative simulations and Challenge scoring preference.",
            "Use decay to reduce turnover, but avoid over-smoothing signals.",
            "Use truncation to reduce concentration in single stocks.",
            "Use neutralization to reduce market/industry/subindustry exposure.",
            "Use rank/group_rank/ts_rank to control extreme values and compare within cross-section/history.",
            "Use ts_backfill for sparse analyst/options/news fields.",
            "Use trade_when to gate noisy signals and reduce unnecessary turnover.",
            "Reduce correlation by varying equivalent fields, operators, grouping, and time horizons.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"docs={payload['doc_count']} chars={payload['text_chars']} expressions={payload['expression_count']} out={out}")

    if args.import_examples and unique_expressions:
        init_db()
        db = SessionLocal()
        try:
            if args.clean_imported:
                for row in db.query(LeaderboardAlpha).filter(
                    LeaderboardAlpha.sharpe.is_(None),
                    LeaderboardAlpha.fitness.is_(None),
                    LeaderboardAlpha.turnover.is_(None),
                    LeaderboardAlpha.self_correlation.is_(None),
                    LeaderboardAlpha.passes_checks == True,
                ).all():
                    if row.expression not in unique_expressions:
                        db.delete(row)

            imported = 0
            for expression in unique_expressions:
                existing = db.query(LeaderboardAlpha).filter(LeaderboardAlpha.expression == expression).first()
                if existing:
                    continue
                db.add(
                    LeaderboardAlpha(
                        expression=expression,
                        sharpe=None,
                        fitness=None,
                        turnover=None,
                        self_correlation=None,
                        passes_checks=True,
                    )
                )
                imported += 1
            db.commit()
            print(f"imported_examples={imported}")
        finally:
            db.close()

    return 0


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _page_count(path: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def _extract_expression_lines(text: str) -> list[str]:
    expressions = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not any(hint in line for hint in ALPHA_OPERATOR_HINTS):
            continue
        line = re.sub(r"^\d+\s+", "", line)
        line = re.sub(r"\s+Simulate Alphas.*$", "", line)
        if _is_clean_expression(line):
            expressions.append(line.strip())
    return expressions


def _is_clean_expression(line: str) -> bool:
    lowered = line.lower()
    if any(
        token in lowered
        for token in (
            "alpha expression:",
            "suppose ",
            "description:",
            "operator ",
            "example:",
            "however,",
            "note:",
            "when to use",
            "the expression",
            "you get",
        )
    ):
        return False
    if not 8 <= len(line) <= 220:
        return False
    if line.count("(") != line.count(")"):
        return False
    if not any(hint in line for hint in ALPHA_OPERATOR_HINTS):
        return False
    if not re.match(r"^-?(rank|ts_|group_|trade_when|winsorize|is_nan|bucket|pasteurize)", line):
        return False
    if not re.fullmatch(r"[A-Za-z0-9_(),.+\-*/ <>=?\"']+", line):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
