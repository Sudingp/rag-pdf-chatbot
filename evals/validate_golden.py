"""
Checks the golden set itself, because a wrong expected answer makes every
score meaningless.

    python -m evals.validate_golden

Rules:
  * ids are unique, categories are known, questions are non-empty
  * answerable cases have at least one expected answer
  * every evidence sentence appears verbatim in the PDF text
  * at least one expected answer appears inside the evidence
  * unanswerable and adversarial cases have no answers and no evidence
"""

import json
import sys
from typing import List

from evals.metrics import squash

CATEGORIES = {"factual", "numeric", "paraphrase", "distractor", "reasoning", "unanswerable", "adversarial"}


def load_golden(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(cases: List[dict], document_text: str) -> List[str]:
    errors, seen = [], set()
    doc = squash(document_text)
    for c in cases:
        cid = c.get("id", "<no id>")
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)
        if c.get("category") not in CATEGORIES:
            errors.append(f"{cid}: unknown category {c.get('category')!r}")
        if not c.get("question", "").strip():
            errors.append(f"{cid}: empty question")

        answers, evidence = c.get("answers", []), c.get("evidence", "")
        if c.get("category") in ("unanswerable", "adversarial"):
            if answers or evidence:
                errors.append(f"{cid}: {c.get('category')} case must have no answers/evidence")
            continue
        if not answers:
            errors.append(f"{cid}: answerable case has no expected answers")
        if evidence:
            if squash(evidence) not in doc:
                errors.append(f"{cid}: evidence not found verbatim in document")
            elif not any(squash(a) in squash(evidence) for a in answers):
                errors.append(f"{cid}: no expected answer appears in its evidence")
        elif c.get("category") != "reasoning":
            errors.append(f"{cid}: answerable case has no evidence")
    return errors


def main(argv=None) -> int:
    import argparse
    from ragbot.pipeline import read_pdf

    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default="evals/data/kestrel_savings_terms.pdf")
    p.add_argument("--golden", default="evals/data/golden.jsonl")
    a = p.parse_args(argv)
    cases = load_golden(a.golden)
    errors = validate(cases, read_pdf(a.pdf))
    for e in errors:
        print("ERROR", e)
    print(f"{len(cases)} cases checked, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
