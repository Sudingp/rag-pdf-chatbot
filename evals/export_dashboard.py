"""
Exports every saved evaluation result as tidy CSV files for a BI dashboard (Power BI, Tableau,
Excel). Nothing is re-run and no model is called: it only reshapes saved results.

    python -m evals.export_dashboard --rag-run evals/results/<run>.json --kg-run evals/results/<run>_kg.json

Writes to dashboard/data/:
  rag_cases.csv      one row per golden-set question: category, outcome, pass, F1, latency, confidence
  metrics.csv        long format, one row per (evaluation, system, metric): the headline numbers
  kg_questions.csv   graph vs RAG precision/recall for each list-style question
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

from evals import calibration, kg_eval, ner_eval

HERE = os.path.dirname(__file__)
OUT = os.path.join(os.path.dirname(HERE), "dashboard", "data")
PASSING = {"CORRECT", "CORRECT_ABSTAIN"}


def rag_case_rows(run: dict) -> List[Dict]:
    rid = run["config"]["run_id"]
    rows = []
    for c in run["cases"]:
        t = c.get("trace") or {}
        rows.append({
            "run_id": rid, "case_id": c["id"], "category": c["category"],
            "outcome": c["failure_mode"], "passed": int(c["failure_mode"] in PASSING),
            "answerable": int(c["category"] not in ("unanswerable", "adversarial")),
            "f1": c.get("f1"), "exact_match": c.get("em"),
            "evidence_retrieved": _int(c.get("evidence_retrieved")),
            "grounded": _int(c.get("grounded")), "abstained": _int(c.get("abstained")),
            "stated_confidence": t.get("confidence") or "", "tool_calls": len(t.get("tool_calls") or []),
            "latency_ms": c.get("latency_ms"), "model": t.get("model", ""),
        })
    return rows


def _int(v):
    return None if v is None else int(bool(v))


def metric(evaluation, system, name, value, unit="ratio", n=None):
    return {"evaluation": evaluation, "system": system, "metric": name,
            "value": None if value is None else round(float(value), 4), "unit": unit, "n": n}


def rag_metrics(run: dict) -> List[Dict]:
    s, model = run["summary"], run["cases"][0].get("trace", {}).get("model", "LLM")
    out = [metric("RAG golden set", model, k, s[k], n=s["cases"]) for k in
           ("pass_rate", "answerable_em", "answerable_f1", "retrieval_hit_rate", "hallucination_rate",
            "guardrail_breach_rate", "wrong_abstain_rate", "grounded_rate")]
    out.append(metric("RAG golden set", model, "median_latency_ms", s["median_latency_ms"], "ms", s["cases"]))
    for cat, v in s["by_category"].items():
        out.append(metric("RAG golden set by category", cat, "pass_rate", v["pass_rate"], n=v["cases"]))
    return out


def calibration_metrics(run: dict) -> List[Dict]:
    rep = calibration.analyse(run["cases"], "score")
    if not rep.get("cases"):
        return []
    return [metric("Confidence calibration", "reader confidence", k, rep[k], n=rep["cases"])
            for k in ("accuracy", "mean_confidence", "ece", "brier", "auroc") if rep.get(k) is not None]


def kg_rows(kg_path: str):
    res = kg_eval.rescore(kg_path, kg_eval.load_golden())
    stats, qa = res["stats"], res["qa"]
    rows = []
    for kind in ("charges", "exemptions"):
        st = stats[kind]
        for k in ("precision", "recall", "f1"):
            rows.append(metric(f"Knowledge-graph extraction: {kind}", "LLM extraction", k, st[k], n=st["expected"]))
    q_rows = []
    for q in qa:
        for system in ("graph", "rag"):
            sc = q[system]
            q_rows.append({"question_id": q["id"], "question": q["question"],
                           "system": "Knowledge graph + tools" if system == "graph" else "RAG",
                           "precision": sc["precision"], "recall": sc["recall"], "f1": sc["f1"],
                           "expected_items": sc["expected"]})
    # Mean over questions (macro average), the figure the README reports: each question counts
    # once, however many items its answer lists.
    for system, label in (("graph", "Knowledge graph + tools"), ("rag", "RAG")):
        for k in ("precision", "recall", "f1"):
            rows.append(metric("List questions: graph vs RAG (mean per question)", label, k,
                               sum(q[system][k] for q in qa) / len(qa), n=len(qa)))
    return rows, q_rows


def ner_rows(kg_path: str) -> List[Dict]:
    try:
        results = ner_eval.run(kg_path)
    except (ImportError, OSError):          # spaCy or its model not installed
        return []
    out = []
    for r in results:
        out += [metric("NER: rupee charge amounts", r.name, "precision", r.precision, n=r.expected),
                metric("NER: rupee charge amounts", r.name, "recall", r.recall, n=r.expected),
                metric("NER: rupee charge amounts", r.name, "f1", r.f1, n=r.expected)]
    return out


def write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def latest(pattern: str):
    found = sorted(glob.glob(os.path.join(HERE, "results", pattern)))
    return found[-1] if found else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-run", default=None)
    ap.add_argument("--kg-run", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    rag_path = args.rag_run or latest("*_llm_*.json")
    kg_path = args.kg_run or latest("*_kg.json")
    if not rag_path:
        raise SystemExit("no saved RAG run found; pass --rag-run")
    with open(rag_path, encoding="utf-8") as f:
        run = json.load(f)

    cases = rag_case_rows(run)
    metrics = rag_metrics(run) + calibration_metrics(run)
    q_rows = []
    if kg_path:
        kg_m, q_rows = kg_rows(kg_path)
        metrics += kg_m + ner_rows(kg_path)

    write_csv(os.path.join(args.out, "rag_cases.csv"), cases)
    write_csv(os.path.join(args.out, "metrics.csv"), metrics)
    if q_rows:
        write_csv(os.path.join(args.out, "kg_questions.csv"), q_rows)
    print(f"wrote {len(cases)} cases, {len(metrics)} metrics, {len(q_rows)} question rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
