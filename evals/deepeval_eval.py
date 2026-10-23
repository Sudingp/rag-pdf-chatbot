"""
Scores a finished harness run with DeepEval's LLM-as-judge metrics and
checks how far the judge agrees with the harness's deterministic checks.

    python -m evals.harness --reader llm                       # 1. answer the golden set
    python -m evals.deepeval_eval evals/results/<run>.json     # 2. judge those answers

It re-uses the answers already in the run file, so the reader is not called
again: only the judge model is. The retrieved chunks are rebuilt from the PDF
with the run's chunk settings.

Metrics
  faithfulness       every claim in the answer is supported by the retrieved text
                     (only for answers the reader actually gave)
  answer_relevancy   the answer addresses the question (same cases)
  contextual_recall  the retrieved text contains what the expected answer needs
                     (every case with an expected answer: this judges the retriever)

Agreement with the harness
  faithfulness       vs  `grounded`           (answer text found verbatim in context)
  contextual_recall  vs  `evidence_retrieved` (evidence sentence found in context)
A judge that disagrees often with a cheap exact check is itself worth a look,
so every disagreement is listed in the report.

The judge is any OpenAI-compatible chat model, read from the same .env as the
`llm` reader (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL), or JUDGE_* variables to use
a different model as judge than as reader.
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from evals.harness import DEFAULT_PDF, RESULTS_DIR
from evals.metrics import REFUSAL_CATEGORIES

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

THRESHOLDS = {"faithfulness": 0.7, "answer_relevancy": 0.7, "contextual_recall": 0.7}
HARNESS_CHECK = {"faithfulness": "grounded", "contextual_recall": "evidence_retrieved"}


# ---------------------------------------------------------------------- #
# Judge model: any OpenAI-compatible chat API
# ---------------------------------------------------------------------- #
def _make_judge_class():
    from deepeval.models import DeepEvalBaseLLM

    from ragbot.azure_reader import _parse_json, _schema_not_supported

    class OpenAICompatibleJudge(DeepEvalBaseLLM):
        """DeepEval judge backed by an OpenAI-style client (Groq, Gemini, Ollama, Azure...)."""

        def __init__(self, client, model: str, max_retries: int = 4, sleep=time.sleep):
            self.client, self.model_id = client, model
            self.max_retries, self.sleep = max_retries, sleep
            self.json_mode = True
            self.calls = 0
            super().__init__(model)

        def load_model(self):
            return self.client

        def get_model_name(self):
            return f"{self.model_id} (OpenAI-compatible)"

        def _chat(self, messages):
            kwargs = {"model": self.model_id, "messages": messages, "temperature": 0.0}
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            for attempt in range(self.max_retries + 1):
                try:
                    self.calls += 1
                    resp = self.client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""
                except Exception as exc:  # noqa: BLE001
                    text = str(exc).lower()
                    if self.json_mode and _schema_not_supported(exc):
                        self.json_mode = False
                        kwargs.pop("response_format", None)
                        continue
                    if attempt < self.max_retries and ("429" in text or "rate limit" in text):
                        self.sleep(min(60, 5 * 2 ** attempt))  # free tiers: back off and retry
                        continue
                    raise
            raise RuntimeError("judge retries exhausted")

        def generate(self, prompt: str, schema=None, **_):
            if schema is None:
                return self._chat([{"role": "user", "content": prompt}])
            system = ("You are an evaluation judge. Reply ONLY with a JSON object that matches "
                      "this JSON schema:\n" + json.dumps(schema.model_json_schema()))
            data = _parse_json(self._chat([{"role": "system", "content": system},
                                            {"role": "user", "content": prompt}]))
            return schema.model_validate(data)

        async def a_generate(self, prompt: str, schema=None, **kw):
            return await asyncio.to_thread(self.generate, prompt, schema, **kw)

    return OpenAICompatibleJudge


def load_judge(client=None):
    """Builds the judge from JUDGE_* or LLM_* settings (a .env file is read too)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    env = lambda k: os.environ.get("JUDGE_" + k) or os.environ.get("LLM_" + k)  # noqa: E731
    if not env("BASE_URL") or not env("MODEL"):
        raise RuntimeError("Set LLM_BASE_URL and LLM_MODEL (or JUDGE_BASE_URL and JUDGE_MODEL) "
                           "in .env to choose the judge model. See .env.example.")
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=env("BASE_URL"), api_key=env("API_KEY") or "not-needed-for-local-ollama")
    return _make_judge_class()(client, env("MODEL"))


# ---------------------------------------------------------------------- #
# Building DeepEval test cases from a harness run
# ---------------------------------------------------------------------- #
def retrieval_context(row: dict, chunks: List[str]) -> List[str]:
    return [chunks[i] for i in row.get("chunk_ids", []) if 0 <= i < len(chunks)]


def plan(rows: List[dict]) -> List[tuple]:
    """(row, metric) pairs worth judging. Refusals and adversarial cases are skipped:
    the harness already scores those exactly, and a judge adds nothing there."""
    jobs = []
    for r in rows:
        if r["category"] in REFUSAL_CATEGORIES:
            continue
        if not r["abstained"]:
            jobs += [(r, "faithfulness"), (r, "answer_relevancy")]
        if r["expected"]:
            jobs.append((r, "contextual_recall"))
    return jobs


def default_metric_factory(judge) -> Callable[[str], object]:
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric

    classes = {"faithfulness": FaithfulnessMetric, "answer_relevancy": AnswerRelevancyMetric,
               "contextual_recall": ContextualRecallMetric}
    return lambda name: classes[name](threshold=THRESHOLDS[name], model=judge,
                                      async_mode=False, include_reason=True)


def judge_rows(rows: List[dict], chunks: List[str], make_metric: Callable[[str], object],
               limit: Optional[int] = None, pause: float = 0.0) -> List[dict]:
    from deepeval.test_case import LLMTestCase

    jobs = plan(rows)
    if limit:
        keep = {r["id"] for r in rows if r["category"] not in REFUSAL_CATEGORIES}
        keep = set(sorted(keep)[:limit])
        jobs = [j for j in jobs if j[0]["id"] in keep]
    results = []
    for i, (r, name) in enumerate(jobs, 1):
        tc = LLMTestCase(input=r["question"], actual_output=r["answer"],
                         expected_output=(r["expected"] or [""])[0],
                         retrieval_context=retrieval_context(r, chunks) or [""])
        metric = make_metric(name)
        try:
            metric.measure(tc)
            score, reason, error = float(metric.score), metric.reason, None
        except Exception as exc:  # noqa: BLE001  one bad judge reply must not end the run
            score, reason, error = None, None, f"{type(exc).__name__}: {exc}"[:300]
        check = HARNESS_CHECK.get(name)
        results.append({
            "id": r["id"], "category": r["category"], "metric": name, "score": score,
            "passed": None if score is None else score >= THRESHOLDS[name],
            "reason": reason, "error": error, "harness_failure_mode": r["failure_mode"],
            "harness_check": check, "harness_value": r.get(check) if check else None,
        })
        print(f"[{i}/{len(jobs)}] {r['id']:<9} {name:<18} "
              + (f"{score:.2f}" if score is not None else "ERROR"))
        if pause:
            time.sleep(pause)
    return results


# ---------------------------------------------------------------------- #
# Summary and report
# ---------------------------------------------------------------------- #
def summarize(results: List[dict]) -> Dict:
    out = {}
    for name in THRESHOLDS:
        rs = [r for r in results if r["metric"] == name]
        scored = [r for r in rs if r["score"] is not None]
        m = {"cases": len(rs), "errors": len(rs) - len(scored),
             "mean": round(sum(r["score"] for r in scored) / len(scored), 4) if scored else None,
             "pass_rate": round(sum(r["passed"] for r in scored) / len(scored), 4) if scored else None}
        if name in HARNESS_CHECK:
            pairs = [r for r in scored if r["harness_value"] is not None]
            agree = [r for r in pairs if r["passed"] == bool(r["harness_value"])]
            m["agreement_with_harness"] = round(len(agree) / len(pairs), 4) if pairs else None
            m["disagreements"] = [r["id"] for r in pairs if r not in agree]
        by_cat = defaultdict(list)
        for r in scored:
            by_cat[r["category"]].append(r["score"])
        m["by_category"] = {c: round(sum(v) / len(v), 4) for c, v in sorted(by_cat.items())}
        out[name] = m
    return out


def _f(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_report(meta: dict, summary: Dict, results: List[dict]) -> str:
    lines = [f"# DeepEval judge report: {meta['run_id']}", "",
             f"- Reader: {meta['reader']} (chunk {meta['chunk_size']}, overlap {meta['overlap']}, k {meta['k']})",
             f"- Judge: {meta['judge']}", f"- Judge calls: {meta['judge_calls']}", "",
             "| Metric | Cases | Mean | Pass rate | Agreement with harness | Judge errors |",
             "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in summary.items():
        agree = _f(m["agreement_with_harness"]) if "agreement_with_harness" in m else "-"
        lines.append(f"| {name} | {m['cases']} | {_f(m['mean'])} | {_f(m['pass_rate'])} | {agree} | {m['errors']} |")
    lines += ["", "Agreement compares the judge's pass/fail with the harness's exact checks: "
              "faithfulness vs `grounded`, contextual recall vs `evidence_retrieved`.", ""]
    lines.append("## Where the judge and the harness disagree\n")
    dis = [r for r in results if r["harness_check"] and r["score"] is not None
           and r["harness_value"] is not None and r["passed"] != bool(r["harness_value"])]
    if not dis:
        lines.append("None.")
    for r in dis:
        lines.append(f"- **{r['id']}** {r['metric']} {r['score']:.2f} vs {r['harness_check']}="
                     f"{r['harness_value']} (harness: {r['harness_failure_mode']}). {r['reason'] or ''}")
    errs = [r for r in results if r["error"]]
    if errs:
        lines += ["", "## Judge errors\n"] + [f"- {r['id']} {r['metric']}: {r['error']}" for r in errs]
    return "\n".join(lines) + "\n"


def main(argv=None, judge=None, make_metric=None) -> int:
    p = argparse.ArgumentParser(description="Judge a harness run with DeepEval metrics.")
    p.add_argument("run", help="a harness result file, e.g. evals/results/<run>.json")
    p.add_argument("--pdf", default=DEFAULT_PDF)
    p.add_argument("--limit", type=int, default=None, help="judge only the first N cases (saves quota)")
    p.add_argument("--pause", type=float, default=0.0, help="seconds between judge calls (free tiers)")
    p.add_argument("--out", default=RESULTS_DIR)
    a = p.parse_args(argv)

    from ragbot.pipeline import chunk_text, read_pdf

    with open(a.run, encoding="utf-8") as f:
        run = json.load(f)
    cfg = run["config"]
    chunks = chunk_text(read_pdf(a.pdf), cfg["chunk_size"], cfg["overlap"])
    if cfg.get("chunks") not in (None, len(chunks)):
        print(f"The PDF gives {len(chunks)} chunks but the run had {cfg['chunks']}: "
              "pass the same --pdf the run used.")
        return 2

    if make_metric is None:
        judge = judge or load_judge()
        make_metric = default_metric_factory(judge)
    results = judge_rows(run["cases"], chunks, make_metric, a.limit, a.pause)
    summary = summarize(results)
    meta = {**cfg, "judge": judge.get_model_name() if judge else "custom",
            "judge_calls": getattr(judge, "calls", None)}
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"deepeval_{cfg['run_id']}")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "results": results}, f, indent=2)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(render_report(meta, summary, results))
    for name, m in summary.items():
        print(f"{name:<18} mean {_f(m['mean'])}  pass {_f(m['pass_rate'])}"
              + (f"  agrees with harness {_f(m['agreement_with_harness'])}" if "agreement_with_harness" in m else ""))
    print("written:", stem + ".md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
