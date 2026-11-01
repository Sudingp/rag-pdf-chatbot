"""Calibration and discrimination metrics."""

from evals import calibration as C


def rows(pairs):
    return [{"id": f"q{i}", "score": c, "failure_mode": "CORRECT" if y else "PARTIAL", "trace": {}}
            for i, (c, y) in enumerate(pairs)]


def test_same_confidence_everywhere_is_calibrated_but_useless():
    """The live Groq run: 'high' (0.9) on all 44 answers, 41 right."""
    r = C.analyse(rows([(0.9, 1)] * 41 + [(0.9, 0)] * 3))
    assert r["ece"] == round(abs(41 / 44 - 0.9), 4)     # small: looks calibrated
    assert r["auroc"] == 0.5                              # but cannot tell right from wrong
    assert r["distinct_values"] == 1 and len(r["wrong_but_confident"]) == 3


def test_perfect_separation():
    r = C.analyse(rows([(0.95, 1), (0.9, 1), (0.2, 0), (0.1, 0)]))
    assert r["auroc"] == 1.0
    top = [x for x in r["risk_coverage"] if x["threshold"] == 0.9][0]
    assert top == {"threshold": 0.9, "coverage": 0.5, "accuracy": 1.0}


def test_brier_and_bins():
    data = [(0.8, 1), (0.8, 0), (0.3, 0), (0.3, 1), (0.55, 1), (0.65, 1)]
    assert C.brier(data) == round(sum((c - y) ** 2 for c, y in data) / 6, 4)
    bins = {b["bin"]: b for b in C.reliability(data)}
    assert bins["0.80"]["accuracy"] == 0.5


def test_input_guardrail_refusals_and_missing_signals_are_skipped():
    data = rows([(0.9, 1)]) + [{"id": "adv", "score": 1.0, "failure_mode": "CORRECT_ABSTAIN",
                                "trace": {"guardrail": "possible prompt injection"}}]
    assert len(C.pairs(data)) == 1
    assert C.analyse(data, "retrieval_score")["cases"] == 0
    data[0]["trace"]["retrieval_score"] = 0.7
    assert C.pairs(data, "retrieval_score") == [(0.7, 1)]


def test_report_renders(tmp_path):
    import json
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"config": {"run_id": "r1"}, "cases": rows([(0.9, 1), (0.6, 0)])}))
    assert C.main([str(path)]) == 0
    text = (tmp_path / "run_calibration.md").read_text()
    assert "AUROC" in text and "retrieval_score | 0 | not recorded" in text
