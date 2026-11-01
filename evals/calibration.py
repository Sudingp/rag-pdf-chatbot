"""
Is the chatbot's confidence worth anything?

Two separate questions, because a model can pass one and fail the other:

* Calibration: when it says 0.9, is it right about 90% of the time?
  Measured with a reliability table, expected calibration error (ECE) and Brier score.
* Discrimination: are its right answers more confident than its wrong ones?
  Measured with AUROC (0.5 = confidence tells you nothing, 1.0 = perfectly separates
  right from wrong) and a risk-coverage table: if we only answer when confidence is at
  least t, how many questions do we answer and how many of those are right?

A reader that says "high" for everything can have a small ECE and still be useless
for deciding when to abstain. That is exactly what the first live run showed (README).

    python -m evals.calibration evals/results/<run>.json
    python -m evals.calibration evals/results/<run>.json --signal retrieval_score
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from evals.metrics import PASSING

SIGNALS = {
    "score": "reader confidence",
    "retrieval_score": "cosine similarity of the best retrieved chunk",
}


def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def signal_value(row: dict, signal: str) -> Optional[float]:
    if signal in row and row[signal] is not None:
        return float(row[signal])
    v = (row.get("trace") or {}).get(signal)
    return None if v is None else float(v)


INPUT_GUARDRAIL = ("possible prompt injection", "empty question", "question longer than")


def pairs(rows: List[dict], signal: str = "score", answered_only: bool = False) -> List[Tuple[float, int]]:
    """(confidence, correct) per case. Questions refused by the input guardrail are skipped:
    the model never saw them, and their score is a fixed 1.0 it never produced."""
    out = []
    for r in rows:
        guard = str((r.get("trace") or {}).get("guardrail") or "")
        if guard.startswith(INPUT_GUARDRAIL):
            continue
        if answered_only and r.get("abstained"):
            continue
        v = signal_value(r, signal)
        if v is None:
            continue
        out.append((min(max(v, 0.0), 1.0), int(r["failure_mode"] in PASSING)))
    return out


def reliability(data: Sequence[Tuple[float, int]], bins: int = 10) -> List[Dict]:
    """Equal-width bins; a reader with few distinct values (high/medium/low) gets one bin per value."""
    distinct = sorted({c for c, _ in data})
    if len(distinct) <= 5:
        groups = {v: [(c, y) for c, y in data if c == v] for v in distinct}
        labels = {v: f"{v:.2f}" for v in distinct}
    else:
        groups, labels = {}, {}
        for c, y in data:
            b = min(int(c * bins), bins - 1)
            groups.setdefault(b, []).append((c, y))
            labels[b] = f"{b / bins:.1f}-{(b + 1) / bins:.1f}"
    table = []
    for key in sorted(groups):
        g = groups[key]
        conf = sum(c for c, _ in g) / len(g)
        acc = sum(y for _, y in g) / len(g)
        table.append({"bin": labels[key], "n": len(g), "confidence": round(conf, 4),
                      "accuracy": round(acc, 4), "gap": round(abs(acc - conf), 4)})
    return table


def ece(data, bins: int = 10) -> Optional[float]:
    if not data:
        return None
    return round(sum(b["n"] * b["gap"] for b in reliability(data, bins)) / len(data), 4)


def brier(data) -> Optional[float]:
    if not data:
        return None
    return round(sum((c - y) ** 2 for c, y in data) / len(data), 4)


def auroc(data) -> Optional[float]:
    """Probability that a random right answer is more confident than a random wrong one
    (ties count half). None when every answer is right, or every answer is wrong."""
    pos = [c for c, y in data if y == 1]
    neg = [c for c, y in data if y == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def risk_coverage(data) -> List[Dict]:
    """For each confidence threshold: share of questions answered, and accuracy on those."""
    out = []
    for t in sorted({c for c, _ in data}):
        kept = [y for c, y in data if c >= t]
        out.append({"threshold": round(t, 4), "coverage": round(len(kept) / len(data), 4),
                    "accuracy": round(sum(kept) / len(kept), 4)})
    return out


def analyse(rows: List[dict], signal: str = "score") -> Dict:
    data = pairs(rows, signal)
    if not data:
        return {"signal": signal, "cases": 0}
    return {
        "signal": signal, "cases": len(data),
        "accuracy": round(sum(y for _, y in data) / len(data), 4),
        "mean_confidence": round(sum(c for c, _ in data) / len(data), 4),
        "distinct_values": len({c for c, _ in data}),
        "ece": ece(data), "brier": brier(data), "auroc": auroc(data),
        "reliability": reliability(data), "risk_coverage": risk_coverage(data),
        "wrong_but_confident": [r["id"] for r in rows
                                if signal_value(r, signal) is not None and signal_value(r, signal) >= 0.8
                                and r["failure_mode"] not in PASSING],
    }


def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def render(run_id: str, results: List[Dict]) -> str:
    lines = [f"# Calibration report: {run_id}", ""]
    lines += ["| Signal | Cases | Accuracy | Mean confidence | Distinct values | ECE | Brier | AUROC |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        if not r["cases"]:
            lines.append(f"| {r['signal']} | 0 | not recorded in this run | | | | | |")
            continue
        lines.append(f"| {r['signal']} ({SIGNALS.get(r['signal'], '')}) | {r['cases']} | {_fmt(r['accuracy'], True)} | "
                     f"{_fmt(r['mean_confidence'])} | {r['distinct_values']} | {_fmt(r['ece'])} | "
                     f"{_fmt(r['brier'])} | {_fmt(r['auroc'])} |")
    for r in results:
        if not r["cases"]:
            continue
        lines += ["", f"## {r['signal']}: reliability", "", "| Bin | Cases | Mean confidence | Accuracy | Gap |",
                  "| --- | --- | --- | --- | --- |"]
        lines += [f"| {b['bin']} | {b['n']} | {_fmt(b['confidence'])} | {_fmt(b['accuracy'], True)} | {_fmt(b['gap'])} |"
                  for b in r["reliability"]]
        lines += ["", f"## {r['signal']}: answer only above a threshold", "",
                  "| Threshold | Coverage | Accuracy |", "| --- | --- | --- |"]
        lines += [f"| {x['threshold']} | {_fmt(x['coverage'], True)} | {_fmt(x['accuracy'], True)} |"
                  for x in r["risk_coverage"]]
        if r["wrong_but_confident"]:
            lines += ["", "Wrong but confident (>= 0.8): " + ", ".join(r["wrong_but_confident"])]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Calibration and discrimination of the chatbot's confidence")
    p.add_argument("run", help="evals/results/<run>.json written by evals.harness")
    p.add_argument("--signal", action="append", choices=sorted(SIGNALS),
                   help="confidence signal(s) to analyse (default: all recorded)")
    a = p.parse_args(argv)
    run = load_run(a.run)
    rows = run["cases"]
    signals = a.signal or list(SIGNALS)
    results = [analyse(rows, s) for s in signals]
    run_id = run.get("config", {}).get("run_id", os.path.basename(a.run))
    out = os.path.splitext(a.run)[0] + "_calibration.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(run_id, results))
    for r in results:
        if r["cases"]:
            print(f"{r['signal']}: n={r['cases']} accuracy={_fmt(r['accuracy'], True)} ECE={_fmt(r['ece'])} "
                  f"Brier={_fmt(r['brier'])} AUROC={_fmt(r['auroc'])} distinct={r['distinct_values']}")
        else:
            print(f"{r['signal']}: not recorded in this run")
    print("report:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
