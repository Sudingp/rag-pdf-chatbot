"""
Runs the golden set through a RagPipeline and scores every answer.

    python -m evals.harness                          # default app settings
    python -m evals.harness --overlap 150 --min-score 0.3
    python -m evals.harness --sweep                  # compare a grid of settings
    python -m evals.harness --save-baseline          # record current scores
    python -m evals.harness --check-baseline         # fail (exit 1) on regression
    python -m evals.harness --reader generative      # Flan-T5 instead of DistilBERT
    python -m evals.harness --compare-readers        # both readers, side by side

Outputs, in evals/results/:
    <run>.json   every case: question, expected, answer, score, EM, F1,
                 whether the evidence was retrieved, failure mode
    <run>.md     summary report: overall scores, per-category scores,
                 failure-mode counts, and every failing case
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from evals import metrics as M
from evals.validate_golden import load_golden, validate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(HERE, "data", "kestrel_savings_terms.pdf")
DEFAULT_GOLDEN = os.path.join(HERE, "data", "golden.jsonl")
RESULTS_DIR = os.path.join(HERE, "results")
BASELINE = os.path.join(HERE, "baseline.json")

# Allowed drop before --check-baseline fails the run
TOLERANCE = {"answerable_f1": 0.02, "answerable_em": 0.02, "pass_rate": 0.02}


def evaluate(pipeline, cases: List[dict]) -> List[dict]:
    rows = []
    for n, c in enumerate(cases, 1):
        print(f"  [{n}/{len(cases)}] {c['id']}", end="\r", flush=True)  # hosted models take a while
        t0 = time.perf_counter()
        ans = pipeline.ask(c["question"])
        latency_ms = (time.perf_counter() - t0) * 1000
        answers = c.get("answers", [])
        em = M.exact_match(ans.text, answers) if answers else 0.0
        f1 = M.f1_score(ans.text, answers) if answers else 0.0
        retrieved = M.evidence_retrieved(c.get("evidence", ""), ans.context)
        is_grounded = M.grounded(ans.raw or ans.text, ans.context)
        rows.append({
            "id": c["id"], "category": c["category"], "question": c["question"],
            "expected": answers, "answer": ans.text, "score": round(ans.score, 4),
            "raw_output": ans.raw, "abstained": ans.abstained, "grounded": is_grounded,
            "truncated": ans.truncated, "em": em, "f1": round(f1, 4),
            "evidence_retrieved": retrieved, "chunk_ids": ans.chunk_ids,
            "trace": ans.trace,
            "latency_ms": round(latency_ms, 1),
            "failure_mode": M.classify(c, ans.text, ans.abstained, retrieved, em, f1, is_grounded),
        })
    return rows


def summarize(rows: List[dict]) -> Dict:
    answerable = [r for r in rows if r["category"] not in M.REFUSAL_CATEGORIES]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    adversarial = [r for r in rows if r["category"] == "adversarial"]
    with_evidence = [r for r in rows if r["evidence_retrieved"] is not None]

    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 4) if xs else None

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    return {
        "cases": len(rows),
        "pass_rate": mean(r["failure_mode"] in M.PASSING for r in rows),
        "answerable_em": mean(r["em"] for r in answerable),
        "answerable_f1": mean(r["f1"] for r in answerable),
        "retrieval_hit_rate": mean(r["evidence_retrieved"] for r in with_evidence),
        "hallucination_rate": mean(r["failure_mode"] == M.HALLUCINATION for r in unanswerable),
        "guardrail_breach_rate": mean(r["failure_mode"] == M.GUARDRAIL_BREACH for r in adversarial),
        "wrong_abstain_rate": mean(r["failure_mode"] == M.WRONG_ABSTAIN for r in answerable),
        # Of the answers actually given, how many are supported by the retrieved text?
        "grounded_rate": mean(r["grounded"] for r in rows if not r["abstained"]),
        "truncation_rate": mean(r["truncated"] for r in rows),
        "median_latency_ms": sorted(r["latency_ms"] for r in rows)[len(rows) // 2] if rows else None,
        "by_category": {
            cat: {"cases": len(rs),
                  "pass_rate": mean(r["failure_mode"] in M.PASSING for r in rs),
                  "f1": mean(r["f1"] for r in rs) if cat not in M.REFUSAL_CATEGORIES else None}
            for cat, rs in sorted(by_cat.items())
        },
        "failure_modes": dict(Counter(r["failure_mode"] for r in rows).most_common()),
    }


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_report(config: dict, summary: dict, rows: List[dict]) -> str:
    s = summary
    lines = [
        f"# Evaluation report: {config['run_id']}",
        "",
        f"Config: reader={config['reader']}, chunk_size={config['chunk_size']}, overlap={config['overlap']}, "
        f"k={config['k']}, min_score={config['min_score']}  |  {s['cases']} cases",
        "",
        "| Metric | Value |", "| --- | --- |",
        f"| Pass rate (all cases) | {_pct(s['pass_rate'])} |",
        f"| Exact match (answerable) | {_pct(s['answerable_em'])} |",
        f"| Token F1 (answerable) | {_pct(s['answerable_f1'])} |",
        f"| Retrieval hit rate (evidence in top-k) | {_pct(s['retrieval_hit_rate'])} |",
        f"| Hallucination rate (unanswerable answered) | {_pct(s['hallucination_rate'])} |",
        f"| Guardrail breach rate (prompt injection obeyed) | {_pct(s['guardrail_breach_rate'])} |",
        f"| Wrong-abstain rate (answerable refused) | {_pct(s['wrong_abstain_rate'])} |",
        f"| Grounded rate (answer text found in retrieved context) | {_pct(s['grounded_rate'])} |",
        f"| Context truncated to fit model | {_pct(s['truncation_rate'])} |",
        f"| Median latency per question | {s['median_latency_ms']} ms |",
        "",
        "## By category", "",
        "| Category | Cases | Pass rate | F1 |", "| --- | --- | --- | --- |",
    ]
    for cat, v in s["by_category"].items():
        lines.append(f"| {cat} | {v['cases']} | {_pct(v['pass_rate'])} | {_pct(v['f1'])} |")
    lines += ["", "## Failure modes", "", "| Mode | Count |", "| --- | --- |"]
    for mode, n in s["failure_modes"].items():
        lines.append(f"| {mode} | {n} |")
    failing = [r for r in rows if r["failure_mode"] not in M.PASSING]
    lines += ["", f"## Failing cases ({len(failing)})", "",
              "| id | mode | question | expected | got | score |", "| --- | --- | --- | --- | --- | --- |"]
    for r in failing:
        exp = " / ".join(r["expected"]) or "(should abstain)"
        got = r["answer"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {r['id']} | {r['failure_mode']} | {r['question']} | {exp} | {got} | {r['score']} |")
    return "\n".join(lines) + "\n"


def compare_to_baseline(summary: dict, baseline: dict) -> List[str]:
    """Return regressions (empty = OK)."""
    problems = []
    for key, tol in TOLERANCE.items():
        old, new = baseline.get(key), summary.get(key)
        if old is not None and new is not None and new < old - tol:
            problems.append(f"{key} dropped {old:.3f} -> {new:.3f} (tolerance {tol})")
    old_h, new_h = baseline.get("hallucination_rate"), summary.get("hallucination_rate")
    if old_h is not None and new_h is not None and new_h > old_h + 0.02:
        problems.append(f"hallucination_rate rose {old_h:.3f} -> {new_h:.3f}")
    return problems


def run_once(build_pipeline: Callable, text: str, cases: List[dict], chunk_size: int,
             overlap: int, k: int, min_score: Optional[float], out_dir: Optional[str],
             reader: str = "extractive") -> dict:
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
              + f"_{reader[:3]}_c{chunk_size}_o{overlap}_k{k}_t{min_score}")
    pipe = build_pipeline(reader=reader, chunk_size=chunk_size, overlap=overlap, k=k, min_score=min_score)
    pipe.index_text(text)
    rows = evaluate(pipe, cases)
    summary = summarize(rows)
    config = {"run_id": run_id, "reader": reader, "chunk_size": chunk_size, "overlap": overlap, "k": k,
              "min_score": min_score, "chunks": len(pipe.chunks)}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, run_id + ".json"), "w", encoding="utf-8") as f:
            json.dump({"config": config, "summary": summary, "cases": rows}, f, indent=2)
        with open(os.path.join(out_dir, run_id + ".md"), "w", encoding="utf-8") as f:
            f.write(render_report(config, summary, rows))
    return {"config": config, "summary": summary, "rows": rows}


SWEEP_GRIDS = {"extractive": [
    # (chunk_size, overlap, k, min_score)
    (900, 0, 6, None),    # original app
    (900, 150, 6, None),
    (500, 100, 3, None),
    (500, 100, 6, None),
    (300, 75, 3, None),
    (900, 150, 6, 0.1),
    (900, 150, 6, 0.3),
    (500, 100, 3, 0.3),
], "generative": [
    # Flan-T5 reads 512 tokens (~2,000 characters), so keep k * chunk_size small
    (500, 100, 3, None),
    (500, 100, 2, None),
    (300, 75, 4, None),
    (300, 75, 3, None),
    (500, 100, 3, 0.5),
    (300, 75, 4, 0.5),
], "azure": [
    # Hosted model with a large context window: try more chunks; each run costs API calls
    (900, 150, 4, None),
    (900, 150, 6, None),
    (500, 100, 4, None),
    (900, 150, 4, 0.6),   # drop "low" confidence answers
]}
SWEEP_GRIDS["llm"] = SWEEP_GRIDS["azure"]
SWEEP_GRID = SWEEP_GRIDS["extractive"]
# Settings used for --compare-readers: each reader at a fair configuration
COMPARE = {"extractive": (900, 150, 6, 0.3), "generative": (500, 100, 3, None),
           "azure": (900, 150, 4, None), "llm": (900, 150, 4, None)}
# Hosted readers only run when their settings exist (a .env file is read too)
NEEDS_ENV = {"azure": "AZURE_OPENAI_ENDPOINT", "llm": "LLM_BASE_URL"}

SUMMARY_HEADER = ("| reader | chunk | overlap | k | min_score | pass | EM | F1 | retrieval | "
                  "halluc. | guardrail breach | wrong abstain | grounded |\n"
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def summary_row(r: dict) -> str:
    c, s = r["config"], r["summary"]
    return (f"| {c['reader']} | {c['chunk_size']} | {c['overlap']} | {c['k']} | {c['min_score']} | "
            f"{_pct(s['pass_rate'])} | {_pct(s['answerable_em'])} | {_pct(s['answerable_f1'])} | "
            f"{_pct(s['retrieval_hit_rate'])} | {_pct(s['hallucination_rate'])} | "
            f"{_pct(s['guardrail_breach_rate'])} | {_pct(s['wrong_abstain_rate'])} | {_pct(s['grounded_rate'])} |")


def default_builder() -> Callable:
    """Loads real models lazily, once per reader."""
    from ragbot.pipeline import RagPipeline, load_embedder, load_reader

    cache = {}

    def build(reader="extractive", **kw):
        if "embed" not in cache:
            print("Loading embedding model...")
            cache["embed"] = load_embedder()
        if reader not in cache:
            print(f"Loading {reader} reader" + (" (first run downloads it)..." if reader in ("extractive", "generative") else "..."))
            cache[reader] = load_reader(reader)
        return RagPipeline(cache["embed"], cache[reader], **kw)

    return build


def main(argv=None, build_pipeline: Optional[Callable] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the RAG chatbot against the golden set.")
    p.add_argument("--pdf", default=DEFAULT_PDF)
    p.add_argument("--golden", default=DEFAULT_GOLDEN)
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--reader", choices=["extractive", "generative", "azure", "llm"], default="extractive")
    p.add_argument("--compare-readers", action="store_true",
                   help="Run both readers at their recommended settings and compare")
    p.add_argument("--sweep", action="store_true", help="Run the settings grid and compare")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--check-baseline", action="store_true")
    p.add_argument("--out", default=RESULTS_DIR)
    a = p.parse_args(argv)

    from ragbot.pipeline import read_pdf

    text = read_pdf(a.pdf)
    cases = load_golden(a.golden)
    errors = validate(cases, text)
    if errors:
        print("Golden set is invalid; fix it before trusting any score:")
        for e in errors:
            print("  ", e)
        return 2

    if build_pipeline is None:
        build_pipeline = default_builder()

    if a.sweep or a.compare_readers:
        if a.compare_readers:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            readers = [r for r in COMPARE if r not in NEEDS_ENV or os.environ.get(NEEDS_ENV[r])]
            for r in COMPARE:
                if r not in readers:
                    print(f"Skipping the {r} reader: {NEEDS_ENV[r]} is not set (see README).")
            configs = [(reader, COMPARE[reader]) for reader in readers]
            name, title = "readers.md", "Reader comparison"
        else:
            configs = [(a.reader, cfg) for cfg in SWEEP_GRIDS[a.reader]]
            name, title = f"sweep_{a.reader}.md", f"Settings sweep: {a.reader} reader"
        results = [run_once(build_pipeline, text, cases, *cfg, a.out, reader=reader) for reader, cfg in configs]
        table = SUMMARY_HEADER + "\n" + "\n".join(summary_row(r) for r in results)
        modes = "\n".join(f"- {r['config']['reader']} ({r['config']['run_id']}): {r['summary']['failure_modes']}"
                           for r in results)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{len(cases)} golden cases.\n\n{table}\n\n## Failure modes\n\n{modes}\n")
        print(table)
        print("written:", os.path.join(a.out, name))
        return 0

    result = run_once(build_pipeline, text, cases, a.chunk_size, a.overlap, a.k, a.min_score, a.out,
                      reader=a.reader)
    s = result["summary"]
    print(f"pass={_pct(s['pass_rate'])} EM={_pct(s['answerable_em'])} F1={_pct(s['answerable_f1'])} "
          f"retrieval={_pct(s['retrieval_hit_rate'])} hallucination={_pct(s['hallucination_rate'])}")
    print("failure modes:", s["failure_modes"])
    print("report:", os.path.join(a.out, result["config"]["run_id"] + ".md"))

    if a.save_baseline:
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump({**s, "config": result["config"]}, f, indent=2)
        print("Saved baseline:", BASELINE)
    if a.check_baseline:
        if not os.path.exists(BASELINE):
            print("No baseline yet; run with --save-baseline first.")
            return 2
        with open(BASELINE, encoding="utf-8") as f:
            problems = compare_to_baseline(s, json.load(f))
        if problems:
            print("REGRESSION:")
            for pr in problems:
                print("  ", pr)
            return 1
        print("No regression against baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
