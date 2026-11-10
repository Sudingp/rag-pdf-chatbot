import csv
import json

from evals import export_dashboard as ex


def tiny_run():
    case = lambda cid, cat, mode, lat, conf="high": {        # noqa: E731
        "id": cid, "category": cat, "failure_mode": mode, "f1": 1.0, "em": 1.0,
        "evidence_retrieved": True, "grounded": True, "abstained": mode == "CORRECT_ABSTAIN",
        "latency_ms": lat, "score": 0.9, "trace": {"confidence": conf, "model": "m", "tool_calls": []}}
    cases = [case("a", "factual", "CORRECT", 1000), case("b", "unanswerable", "CORRECT_ABSTAIN", 2000),
             case("c", "paraphrase", "PARTIAL", 12000), case("d", "factual", "WRONG_ABSTAIN", 9000)]
    summary = {"cases": 4, "pass_rate": 0.5, "answerable_em": 0.5, "answerable_f1": 0.6,
               "retrieval_hit_rate": 1.0, "hallucination_rate": 0.0, "guardrail_breach_rate": 0.0,
               "wrong_abstain_rate": 0.25, "grounded_rate": 1.0, "median_latency_ms": 5500.0,
               "by_category": {"factual": {"cases": 2, "pass_rate": 0.5}}}
    return {"config": {"run_id": "r1"}, "summary": summary, "cases": cases}


def test_case_rows_mark_correct_and_correct_abstain_as_passed():
    rows = ex.rag_case_rows(tiny_run())
    assert [r["passed"] for r in rows] == [1, 1, 0, 0]
    assert rows[1]["answerable"] == 0 and rows[0]["answerable"] == 1
    assert rows[2]["stated_confidence"] == "high" and rows[2]["latency_ms"] == 12000


def test_headline_metrics_come_from_the_saved_summary():
    m = {(r["evaluation"], r["metric"]): r["value"] for r in ex.rag_metrics(tiny_run())}
    assert m[("RAG golden set", "pass_rate")] == 0.5
    assert m[("RAG golden set", "median_latency_ms")] == 5500.0
    assert m[("RAG golden set by category", "pass_rate")] == 0.5


def test_export_writes_case_and_metric_files(tmp_path):
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(tiny_run()))
    assert ex.main(["--rag-run", str(run_file), "--out", str(tmp_path / "out")]) == 0
    cases = list(csv.DictReader(open(tmp_path / "out" / "rag_cases.csv")))
    assert len(cases) == 4 and {"case_id", "passed", "latency_ms"} <= set(cases[0])
    metrics = list(csv.DictReader(open(tmp_path / "out" / "metrics.csv")))
    assert any(r["metric"] == "pass_rate" for r in metrics)
    assert any(r["evaluation"] == "Confidence calibration" for r in metrics)
