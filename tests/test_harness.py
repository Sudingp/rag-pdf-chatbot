"""End-to-end tests of the evaluation harness with fake models (no downloads)."""

import json

import pytest

from evals import harness
from evals import metrics as M
from evals.validate_golden import load_golden
from ragbot.pipeline import RagPipeline
from tests.conftest import ScriptedQA, fake_embed


def builder_for(qa):
    return lambda reader="extractive", **kw: RagPipeline(fake_embed, qa, **kw)


@pytest.fixture
def golden():
    return load_golden(harness.DEFAULT_GOLDEN)


@pytest.fixture
def perfect_qa(golden):
    """Answers every answerable case correctly with high confidence, low confidence otherwise."""
    script = {c["question"]: (c["answers"][0], 0.9) for c in golden if c["answers"]}
    return ScriptedQA(script, default=("Rs 999", 0.05))


def test_perfect_model_scores_full_marks_when_abstaining(tmp_path, perfect_qa):
    code = harness.main(["--out", str(tmp_path), "--min-score", "0.3"], build_pipeline=builder_for(perfect_qa))
    assert code == 0
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    s = data["summary"]
    assert s["answerable_em"] == 1.0
    assert s["hallucination_rate"] == 0.0
    assert s["pass_rate"] == 1.0
    assert (tmp_path / (data["config"]["run_id"] + ".md")).exists()


def test_always_answering_model_hallucinates_on_unanswerable(tmp_path, perfect_qa):
    harness.main(["--out", str(tmp_path)], build_pipeline=builder_for(perfect_qa))
    s = json.loads(next(tmp_path.glob("*.json")).read_text())["summary"]
    assert s["hallucination_rate"] == 1.0
    assert s["failure_modes"][M.HALLUCINATION] == 6


def test_wrong_answers_are_classified(tmp_path, golden):
    qa = ScriptedQA({}, default=("Rs 999", 0.9))
    harness.main(["--out", str(tmp_path)], build_pipeline=builder_for(qa))
    rows = json.loads(next(tmp_path.glob("*.json")).read_text())["cases"]
    modes = {r["failure_mode"] for r in rows}
    assert M.CORRECT not in modes
    # "Rs 999" appears nowhere in the PDF, so wrong answers are flagged as made up
    assert M.UNGROUNDED in modes
    assert modes <= {M.UNGROUNDED, M.REASONING_GAP, M.HALLUCINATION, M.PARTIAL, M.GUARDRAIL_BREACH}
    report = next(tmp_path.glob("*.md")).read_text()
    assert "## Failing cases" in report and "Rs 999" in report


def test_baseline_gate_detects_regression(tmp_path, perfect_qa, monkeypatch):
    monkeypatch.setattr(harness, "BASELINE", str(tmp_path / "baseline.json"))
    out = str(tmp_path / "r")
    assert harness.main(["--out", out, "--min-score", "0.3", "--save-baseline"],
                        build_pipeline=builder_for(perfect_qa)) == 0
    assert harness.main(["--out", out, "--min-score", "0.3", "--check-baseline"],
                        build_pipeline=builder_for(perfect_qa)) == 0
    worse = ScriptedQA({}, default=("Rs 999", 0.9))
    assert harness.main(["--out", out, "--min-score", "0.3", "--check-baseline"],
                        build_pipeline=builder_for(worse)) == 1


def test_invalid_golden_set_stops_the_run(tmp_path, golden, perfect_qa):
    bad = tmp_path / "golden.jsonl"
    golden[0]["evidence"] = "This sentence is not in the document."
    bad.write_text("\n".join(json.dumps(c) for c in golden))
    assert harness.main(["--golden", str(bad), "--out", str(tmp_path)],
                        build_pipeline=builder_for(perfect_qa)) == 2


def test_sweep_writes_comparison_table(tmp_path, perfect_qa):
    assert harness.main(["--out", str(tmp_path), "--sweep"], build_pipeline=builder_for(perfect_qa)) == 0
    table = (tmp_path / "sweep_extractive.md").read_text()
    assert table.count("\n| ") == len(harness.SWEEP_GRID) + 2  # header + separator + rows


class FakeGenerativeReader:
    """Mimics load_generative_reader(): writes answers, refuses with 'unanswerable'."""

    def __init__(self, golden):
        self.script = {c["question"]: c["answers"][0] for c in golden if c["answers"]}

    def __call__(self, question, context):
        if question in self.script:
            return {"answer": self.script[question], "score": 0.8, "refused": False}
        return {"answer": "unanswerable", "score": 0.9, "refused": True}


def test_generative_reader_refusals_count_as_abstentions(tmp_path, golden):
    harness.main(["--out", str(tmp_path), "--reader", "generative", "--chunk-size", "500", "--k", "3"],
                 build_pipeline=builder_for(FakeGenerativeReader(golden)))
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["config"]["reader"] == "generative"
    s = data["summary"]
    assert s["hallucination_rate"] == 0.0
    # reas-01 (7.30%) is never written in the PDF: a correct generated answer is
    # still right, even though it is not copied from the context
    reas = next(r for r in data["cases"] if r["id"] == "reas-01")
    assert reas["failure_mode"] == M.CORRECT and reas["grounded"] is False


def test_compare_readers_writes_side_by_side_table(tmp_path, golden, perfect_qa, monkeypatch):
    # Only the two offline readers: a developer's .env (with LLM_* keys) must not
    # pull hosted readers into this test
    monkeypatch.setattr(harness, "COMPARE", {k: harness.COMPARE[k] for k in ("extractive", "generative")})
    readers = {"extractive": perfect_qa, "generative": FakeGenerativeReader(golden)}
    build = lambda reader="extractive", **kw: RagPipeline(fake_embed, readers[reader], **kw)  # noqa: E731
    assert harness.main(["--out", str(tmp_path), "--compare-readers"], build_pipeline=build) == 0
    table = (tmp_path / "readers.md").read_text()
    assert "| extractive |" in table and "| generative |" in table


def test_prompt_injection_obeyed_counts_as_guardrail_breach(tmp_path, perfect_qa):
    harness.main(["--out", str(tmp_path)], build_pipeline=builder_for(perfect_qa))  # always answers
    s = json.loads(next(tmp_path.glob("*.json")).read_text())["summary"]
    assert s["guardrail_breach_rate"] == 1.0
    assert s["failure_modes"][M.GUARDRAIL_BREACH] == 4
    assert s["by_category"]["adversarial"]["f1"] is None


def test_refusing_model_has_no_guardrail_breaches(tmp_path, perfect_qa):
    harness.main(["--out", str(tmp_path), "--min-score", "0.3"], build_pipeline=builder_for(perfect_qa))
    s = json.loads(next(tmp_path.glob("*.json")).read_text())["summary"]
    assert s["guardrail_breach_rate"] == 0.0 and s["pass_rate"] == 1.0
