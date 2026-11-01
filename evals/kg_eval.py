"""
Evaluates the knowledge graph (ragbot/kg.py) two ways.

1. Extraction quality: the facts the LLM extracted vs 20 hand-checked charges and
   3 exemptions (evals/data/golden_kg.json). A charge counts as correct only when
   both the service and the amount match; a right service with a wrong amount is
   reported separately, because "found it but misread the number" and "never found
   it" need different fixes. Precision, recall and F1 for charges and exemptions.

2. Graph vs RAG: 6 questions whose answers are spread across the document
   ("which services are free?"). The graph side answers with graph queries; the RAG
   side answers from the top-k retrieved chunks. Both answers are scored the same way:
   which golden charges each answer mentions (regex per charge), giving precision and
   recall per question.

    python -m evals.kg_eval              # extraction + graph answers
    python -m evals.kg_eval --rag        # also ask the same questions through RAG

Needs the LLM_* settings in .env (same as the `llm` reader).
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(HERE, "data", "kestrel_savings_terms.pdf")
DEFAULT_GOLDEN = os.path.join(HERE, "data", "golden_kg.json")
RESULTS_DIR = os.path.join(HERE, "results")


def load_golden(path: str = DEFAULT_GOLDEN) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def golden_value(g: dict):
    if g.get("percent") is not None:
        return ("pct", round(float(g["percent"]), 2))
    return ("rs", round(float(g["amount_rs"]), 2))


def best_name_score(name: str, golden: dict) -> float:
    from ragbot.kg import name_similarity
    return max(name_similarity(name, n) for n in [golden["service"], *golden.get("aliases", [])])


def prf(tp: int, n_pred: int, n_true: int) -> Dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "predicted": n_pred, "expected": n_true,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def match_charges(extracted: List[dict], golden: List[dict], threshold: float = 0.6) -> Tuple[Dict, Dict]:
    """One-to-one matching, best name match first. Returns (full matches, name-only matches),
    each {extracted index: golden id}."""
    from ragbot.kg import value_of

    def assign(require_value: bool, taken_e=(), taken_g=()):
        cands = []
        for i, e in enumerate(extracted):
            if i in taken_e:
                continue
            for g in golden:
                if g["id"] in taken_g:
                    continue
                if require_value and value_of(e) != golden_value(g):
                    continue
                s = best_name_score(e["service"], g)
                if s >= threshold:
                    cands.append((s, i, g["id"]))
        out, used_e, used_g = {}, set(), set()
        for s, i, gid in sorted(cands, reverse=True):
            if i not in used_e and gid not in used_g:
                out[i] = gid
                used_e.add(i)
                used_g.add(gid)
        return out

    full = assign(True)
    name_only = assign(False, set(full), set(full.values()))
    return full, name_only


def score_extraction(extracted: List[dict], graph, golden: dict) -> Dict:
    from ragbot.kg import name_similarity
    gold_charges = golden["charges"]
    full, name_only = match_charges(extracted, gold_charges)
    by_id = {g["id"]: g for g in gold_charges}

    # Exemptions: group must match, and the charge it points to must be the right golden charge
    node_to_gold = {}
    services = {d["service"]: n for n, d in graph.nodes(data=True) if d.get("kind") == "charge"}
    for i, gid in full.items():
        node = services.get(extracted[i]["service"])
        if node:
            node_to_gold[node] = gid
    pred_ex = [(graph.nodes[u]["name"], v) for u, v, d in graph.edges(data=True) if d.get("relation") == "EXEMPT_FROM"]
    used, tp_ex = set(), 0
    for group, node in pred_ex:
        for gex in golden["exemptions"]:
            if gex["id"] in used:
                continue
            names = [gex["group"], *gex.get("aliases", [])]
            if max(name_similarity(group, n) for n in names) >= 0.5 and node_to_gold.get(node) == gex["charge"]:
                used.add(gex["id"])
                tp_ex += 1
                break

    return {
        "charges": prf(len(full), len(extracted), len(gold_charges)),
        "charges_name_only": prf(len(full) + len(name_only), len(extracted), len(gold_charges)),
        "exemptions": prf(tp_ex, len(pred_ex), len(golden["exemptions"])),
        "wrong_value": [{"extracted": extracted[i]["service"], "got": extracted[i].get("amount_rs", extracted[i].get("percent")),
                         "golden": gid, "expected": golden_value(by_id[gid])[1]} for i, gid in name_only.items()],
        "missed": [g["id"] + ": " + g["service"] for g in gold_charges
                   if g["id"] not in set(full.values()) | set(name_only.values())],
        "extra": [extracted[i]["service"] for i in range(len(extracted)) if i not in full and i not in name_only],
        "exemptions_missed": [e["id"] for e in golden["exemptions"] if e["id"] not in used],
    }


_DASHES = re.compile("[\u2010-\u2015\u2212]")


def normalize_answer(text: str) -> str:
    """Models write 'Rs\u202f500', '\u20b9100', 'stop\u2011payment' and curly apostrophes. Without this
    the scorer missed correct answers (first live run: INC-004)."""
    t = unicodedata.normalize("NFKC", text or "")          # narrow/no-break spaces -> space
    t = _DASHES.sub("-", t).replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u20b9", "Rs ")
    return re.sub(r"[ \t]+", " ", t)


def mentioned(answer: str, golden: dict) -> set:
    text = normalize_answer(answer)
    return {c["id"] for c in golden["charges"] if re.search(c["answer_pattern"], text, re.IGNORECASE)}


def score_answer(answer: str, expected: List[str], golden: dict) -> Dict:
    found = mentioned(answer, golden)
    tp = len(found & set(expected))
    out = prf(tp, len(found), len(expected))
    out.update(missing=sorted(set(expected) - found), wrong=sorted(found - set(expected)))
    return out


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render(run_id: str, model: str, ext_stats: Dict, extraction, qa: List[Dict], rag: bool) -> str:
    c, n, e = ext_stats["charges"], ext_stats["charges_name_only"], ext_stats["exemptions"]
    lines = [f"# Knowledge-graph evaluation: {run_id}", "", f"Model: {model}  |  chunks: {extraction.chunks}  |  "
             f"LLM calls for extraction: {extraction.calls}  |  facts dropped by the quote guardrail: "
             f"{sum(1 for d in extraction.dropped if d.get('reason') == 'quote not in chunk')}", "",
             "## Extraction vs hand-checked facts", "",
             "| Facts | Extracted | Expected | Correct | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- | --- | --- |",
             f"| Charges (service and amount) | {c['predicted']} | {c['expected']} | {c['tp']} | {_pct(c['precision'])} | {_pct(c['recall'])} | {_pct(c['f1'])} |",
             f"| Charges (service only) | {n['predicted']} | {n['expected']} | {n['tp']} | {_pct(n['precision'])} | {_pct(n['recall'])} | {_pct(n['f1'])} |",
             f"| Exemptions | {e['predicted']} | {e['expected']} | {e['tp']} | {_pct(e['precision'])} | {_pct(e['recall'])} | {_pct(e['f1'])} |",
             ""]
    for title, key in (("Right service, wrong amount", "wrong_value"), ("Missed", "missed"),
                       ("Extracted but not in the golden set", "extra"), ("Exemptions missed", "exemptions_missed")):
        items = ext_stats[key]
        lines.append(f"**{title} ({len(items)}):** " + ("; ".join(map(str, items)) if items else "none"))
        lines.append("")
    lines += ["## Graph answers" + (" vs RAG answers" if rag else ""), ""]
    head = "| Question | Graph P | Graph R |" + (" RAG P | RAG R |" if rag else "")
    lines += [head, "| --- " * (head.count("|") - 1) + "|"]
    for q in qa:
        row = f"| {q['id']}: {q['question']} | {_pct(q['graph']['precision'])} | {_pct(q['graph']['recall'])} |"
        if rag:
            row += f" {_pct(q['rag']['precision'])} | {_pct(q['rag']['recall'])} |"
        lines.append(row)
    def avg(side, k):
        return sum(q[side][k] for q in qa) / len(qa) if qa else None
    total = f"| **Mean** | {_pct(avg('graph', 'precision'))} | {_pct(avg('graph', 'recall'))} |"
    if rag:
        total += f" {_pct(avg('rag', 'precision'))} | {_pct(avg('rag', 'recall'))} |"
    lines += [total, ""]
    for q in qa:
        lines.append(f"### {q['id']}: {q['question']}")
        lines.append(f"- Graph ({len(q['graph_tool_calls'])} tool calls): {q['graph_answer']}  ")
        lines.append(f"  missing {q['graph']['missing'] or 'none'}, wrong {q['graph']['wrong'] or 'none'}")
        if rag:
            lines.append(f"- RAG: {q['rag_answer']}  ")
            lines.append(f"  missing {q['rag']['missing'] or 'none'}, wrong {q['rag']['wrong'] or 'none'}")
        lines.append("")
    return "\n".join(lines)


def rescore(path: str, golden: dict) -> Dict:
    """Re-scores a saved run with the current golden set and scorer, without calling the LLM.
    Used after fixing the scorer (INC-004) so the same model answers are judged again."""
    import networkx as nx

    from ragbot.kg import Extraction
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    graph = nx.node_link_graph(saved["graph"], directed=True, multigraph=True, edges="edges")
    extracted = [{"service": d["service"], "amount_rs": d.get("amount_rs"), "percent": d.get("percent"),
                  "free": bool(d.get("free")), "unit": d.get("unit", ""), "quote": d.get("quote", "")}
                 for _, d in graph.nodes(data=True) if d.get("kind") == "charge"]
    stats = score_extraction(extracted, graph, golden)
    expected = {q["id"]: q["expected"] for q in golden["questions"]}
    qa = []
    for q in saved["questions"]:
        q = dict(q, expected=expected.get(q["id"], q["expected"]))
        q["graph"] = score_answer(q["graph_answer"], q["expected"], golden)
        if "rag_answer" in q:
            q["rag"] = score_answer(q["rag_answer"], q["expected"], golden)
        qa.append(q)
    extraction = Extraction(dropped=saved.get("dropped", []), chunks=saved.get("chunks", "?"),
                            calls=saved.get("calls", "?"))
    run_id = os.path.splitext(os.path.basename(path))[0] + " (re-scored)"
    report = render(run_id, saved["model"], stats, extraction, qa, any("rag" in q for q in qa))
    out = os.path.splitext(path)[0] + "_rescored.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    return {"stats": stats, "qa": qa, "report": report, "path": out}


def make_client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    missing = [k for k in ("LLM_BASE_URL", "LLM_MODEL") if not os.environ.get(k)]
    if missing:
        raise SystemExit("Set " + ", ".join(missing) + " in .env (see README, 'Hosted LLM reader').")
    from openai import OpenAI

    from ragbot.llm_retry import RetryingClient
    raw = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ.get("LLM_API_KEY") or "x")
    return RetryingClient(raw), os.environ["LLM_MODEL"]


def run(client, model: str, text: str, golden: dict, chunk_size: int = 900, overlap: int = 150,
        rag_pipeline=None, out_dir: str = RESULTS_DIR) -> Dict:
    from ragbot.kg import GraphTools, answer_with_graph, build_graph, extract_facts, graph_to_json, merge_charges
    from ragbot.pipeline import chunk_text

    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Extracting facts from {len(chunks)} chunks...")
    extraction = extract_facts(client, model, chunks)
    graph = build_graph(extraction)
    extracted = merge_charges(extraction.charges)
    stats = score_extraction(extracted, graph, golden)

    tools, qa = GraphTools(graph), []
    for q in golden["questions"]:
        print(f"  {q['id']}: {q['question']}")
        g = answer_with_graph(client, model, tools, q["question"])
        row = {"id": q["id"], "question": q["question"], "expected": q["expected"],
               "graph_answer": g["answer"], "graph_tool_calls": g["tool_calls"],
               "graph": score_answer(g["answer"], q["expected"], golden)}
        if rag_pipeline is not None:
            ans = rag_pipeline.ask(q["question"])
            row["rag_answer"] = ans.text
            row["rag"] = score_answer(ans.text, q["expected"], golden)
        qa.append(row)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_kg"
    report = render(run_id, model, stats, extraction, qa, rag_pipeline is not None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, run_id + ".md"), "w", encoding="utf-8") as f:
            f.write(report)
        with open(os.path.join(out_dir, run_id + ".json"), "w", encoding="utf-8") as f:
            json.dump({"model": model, "chunks": extraction.chunks, "calls": extraction.calls,
                       "extraction": stats, "questions": qa,
                       "dropped": extraction.dropped, "graph": graph_to_json(graph)}, f, indent=2, default=str)
    return {"run_id": run_id, "stats": stats, "qa": qa, "graph": graph, "report": report}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the knowledge graph against hand-checked facts")
    p.add_argument("--pdf", default=DEFAULT_PDF)
    p.add_argument("--golden", default=DEFAULT_GOLDEN)
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--rag", action="store_true", help="also answer the questions with RAG (llm reader, k=4)")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--rescore", metavar="RUN_JSON", help="re-score a saved run (no LLM calls)")
    a = p.parse_args(argv)

    if a.rescore:
        r = rescore(a.rescore, load_golden(a.golden))
        s = r["stats"]
        print(f"charges F1={_pct(s['charges']['f1'])} (P {_pct(s['charges']['precision'])}, R {_pct(s['charges']['recall'])})  "
              f"exemptions F1={_pct(s['exemptions']['f1'])}")
        print("report:", r["path"])
        return 0

    from ragbot.pipeline import read_pdf
    client, model = make_client()
    text = read_pdf(a.pdf)
    rag_pipeline = None
    if a.rag:
        from ragbot.azure_reader import make_azure_reader
        from ragbot.pipeline import RagPipeline, load_embedder
        print("Loading embedding model for the RAG comparison...")
        # Same reader as `--reader llm`, sharing the rate-limit-aware client
        rag_pipeline = RagPipeline(load_embedder(), make_azure_reader(client, model), chunk_size=a.chunk_size,
                                   overlap=a.overlap, k=a.k).index_text(text)
    result = run(client, model, text, load_golden(a.golden), a.chunk_size, a.overlap, rag_pipeline)
    s = result["stats"]
    print(f"charges F1={_pct(s['charges']['f1'])} (P {_pct(s['charges']['precision'])}, R {_pct(s['charges']['recall'])})  "
          f"exemptions F1={_pct(s['exemptions']['f1'])}")
    print("report:", os.path.join(RESULTS_DIR, result["run_id"] + ".md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
