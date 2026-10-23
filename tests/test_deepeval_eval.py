"""
Tests for the DeepEval judge adapter. DeepEval's real metrics run against a
scripted judge client, so no API key, network or cost is involved.
"""

import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("deepeval")

from evals import deepeval_eval as DE  # noqa: E402
from evals import harness  # noqa: E402
from evals.validate_golden import load_golden  # noqa: E402
from ragbot.pipeline import RagPipeline  # noqa: E402
from tests.conftest import ScriptedQA, fake_embed  # noqa: E402


def completion(text):
    return NS(choices=[NS(message=NS(content=text))])


class SchemaAwareClient:
    """Answers DeepEval's prompts by looking at the JSON schema the judge asks for.
    `verdict` sets every verdict, so a test can make the judge pass or fail answers."""

    def __init__(self, verdict="yes", fail_first=None):
        self.verdict, self.fail_first = verdict, list(fail_first or [])
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        if self.fail_first:
            raise self.fail_first.pop(0)
        system = kw["messages"][0]["content"]
        title = json.loads(system.split("\n", 1)[1]).get("title", "")
        if title == "Truths":
            body = {"truths": ["The daily ATM cash withdrawal limit is Rs 40,000."]}
        elif title == "Claims":
            body = {"claims": ["The limit is Rs 40,000."]}
        elif title == "Statements":
            body = {"statements": ["The limit is Rs 40,000."]}
        elif title == "Verdicts":
            body = {"verdicts": [{"verdict": self.verdict, "reason": "checked"}]}
        elif title.endswith("ScoreReason"):
            body = {"reason": "scored by the scripted judge"}
        else:
            raise AssertionError(f"unexpected schema {title}")
        return completion("```json\n" + json.dumps(body) + "\n```")


def judge_with(client):
    return DE._make_judge_class()(client, "judge-test", sleep=lambda s: None)


# ---------------------------------------------------------------------- #
def test_judge_returns_validated_schema_objects():
    from deepeval.metrics.faithfulness.schema import Claims

    judge = judge_with(SchemaAwareClient())
    out = judge.generate("Extract the claims", schema=Claims)
    assert isinstance(out, Claims) and out.claims == ["The limit is Rs 40,000."]
    assert judge.client.requests[0]["response_format"] == {"type": "json_object"}
    assert judge.client.requests[0]["temperature"] == 0.0


def test_judge_retries_rate_limits_and_drops_json_mode_if_unsupported():
    from deepeval.metrics.faithfulness.schema import Claims

    client = SchemaAwareClient(fail_first=[Exception("Error code: 429 - rate limit reached"),
                                           Exception("response_format json_object is not supported")])
    judge = judge_with(client)
    assert judge.generate("x", schema=Claims).claims
    assert "response_format" in client.requests[1] and "response_format" not in client.requests[2]


def test_judge_gives_up_on_other_errors():
    judge = judge_with(SchemaAwareClient(fail_first=[TimeoutError("down")]))
    with pytest.raises(TimeoutError):
        judge.generate("x")


def test_judge_needs_settings(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_MODEL", "JUDGE_BASE_URL", "JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        DE.load_judge()


def test_plan_skips_refusal_cases_and_abstentions():
    rows = [
        {"id": "a", "category": "numeric", "abstained": False, "expected": ["x"]},
        {"id": "b", "category": "numeric", "abstained": True, "expected": ["x"]},
        {"id": "c", "category": "unanswerable", "abstained": False, "expected": []},
        {"id": "d", "category": "adversarial", "abstained": False, "expected": []},
    ]
    assert [(r["id"], m) for r, m in DE.plan(rows)] == [
        ("a", "faithfulness"), ("a", "answer_relevancy"), ("a", "contextual_recall"),
        ("b", "contextual_recall")]


@pytest.fixture
def harness_run(tmp_path):
    golden = load_golden(harness.DEFAULT_GOLDEN)
    qa = ScriptedQA({c["question"]: (c["answers"][0], 0.9) for c in golden if c["answers"]},
                    default=("Rs 999", 0.05))
    build = lambda reader="extractive", **kw: RagPipeline(fake_embed, qa, **kw)  # noqa: E731
    harness.main(["--out", str(tmp_path / "runs"), "--min-score", "0.3"], build_pipeline=build)
    return next((tmp_path / "runs").glob("*.json"))


def test_end_to_end_with_real_deepeval_metrics(tmp_path, harness_run):
    judge = judge_with(SchemaAwareClient(verdict="yes"))
    assert DE.main([str(harness_run), "--out", str(tmp_path), "--limit", "3"], judge=judge) == 0
    data = json.loads(next(tmp_path.glob("deepeval_*.json")).read_text())
    s = data["summary"]
    assert s["faithfulness"]["mean"] == 1.0 and s["faithfulness"]["errors"] == 0
    assert s["answer_relevancy"]["pass_rate"] == 1.0
    assert s["contextual_recall"]["cases"] == 3
    assert data["meta"]["judge_calls"] > 0
    report = next(tmp_path.glob("deepeval_*.md")).read_text()
    assert "| faithfulness |" in report and "Agreement" in report


def test_disagreement_with_harness_is_reported(tmp_path, harness_run):
    judge = judge_with(SchemaAwareClient(verdict="no"))  # judge rejects answers the harness accepts
    DE.main([str(harness_run), "--out", str(tmp_path), "--limit", "2"], judge=judge)
    s = json.loads(next(tmp_path.glob("deepeval_*.json")).read_text())["summary"]
    assert s["faithfulness"]["pass_rate"] == 0.0
    assert s["faithfulness"]["agreement_with_harness"] == 0.0
    assert len(s["faithfulness"]["disagreements"]) == 2
    assert "## Where the judge and the harness disagree" in next(tmp_path.glob("deepeval_*.md")).read_text()


def test_a_broken_judge_reply_is_recorded_not_fatal(tmp_path, harness_run):
    class Garbage(SchemaAwareClient):
        def _create(self, **kw):
            self.requests.append(kw)
            return completion("I think it is fine!")
    DE.main([str(harness_run), "--out", str(tmp_path), "--limit", "1"], judge=judge_with(Garbage()))
    s = json.loads(next(tmp_path.glob("deepeval_*.json")).read_text())["summary"]
    assert s["faithfulness"]["errors"] == 1 and s["faithfulness"]["mean"] is None
