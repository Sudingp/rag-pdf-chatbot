"""Knowledge graph: extraction guardrails, graph queries, tool-using answers and the scoring.
Uses a fake OpenAI-compatible client: no network, no API key."""

import json
import os
from types import SimpleNamespace as NS

import pytest

from evals import kg_eval as K
from evals.metrics import squash
from ragbot import kg

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(HERE, "evals", "data", "kestrel_savings_terms.pdf")
GOLDEN = K.load_golden()

CHUNK = ("A replacement debit card costs Rs 200 and is delivered within 7 working days. "
         "Email alerts are free. No shortfall charge applies to junior accounts, or to accounts held "
         "by customers aged 60 or above. If the monthly average balance falls below this amount, a "
         "shortfall charge of Rs 150 plus applicable taxes is applied on the first working day of the following month.")


def tool_reply(name, args, call_id="c1"):
    tc = NS(id=call_id, type="function", function=NS(name=name, arguments=json.dumps(args)))
    return NS(choices=[NS(finish_reason="tool_calls", message=NS(content=None, tool_calls=[tc]))])


def text_reply(text):
    return NS(choices=[NS(finish_reason="stop", message=NS(content=text, tool_calls=None))])


class FakeClient:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def charge(service, amount=None, quote="", free=False, percent=None, unit=""):
    return {"service": service, "amount_rs": amount, "percent": percent, "unit": unit, "free": free, "quote": quote}


FACTS = {"charges": [
    charge("Replacement debit card", 200, "A replacement debit card costs Rs 200"),
    charge("Email alerts", 0, "Email alerts are free.", free=True),
    charge("Minimum balance shortfall charge", 150, "a shortfall charge of Rs 150 plus applicable taxes", unit="per month"),
    charge("Made-up courier fee", 99, "courier fee of Rs 99"),          # quote not in the text
], "exemptions": [
    {"group": "Customers aged 60 or above", "charge_service": "shortfall charge",
     "quote": "No shortfall charge applies to junior accounts, or to accounts held by customers aged 60 or above."},
]}


def test_golden_quotes_are_verbatim_in_the_pdf():
    from ragbot.pipeline import read_pdf
    doc = squash(read_pdf(PDF))
    for item in GOLDEN["charges"] + GOLDEN["exemptions"]:
        assert squash(item["quote"]) in doc, item["id"]
    ids = {c["id"] for c in GOLDEN["charges"]}
    assert all(set(q["expected"]) <= ids for q in GOLDEN["questions"])


def test_name_similarity():
    assert kg.name_similarity("replacement debit card", "Debit card replacement") == 1.0
    assert kg.name_similarity("SMS alerts", "SMS alert charge") == 1.0
    assert kg.name_similarity("printed statement", "email alerts") == 0.0


def test_extraction_drops_facts_whose_quote_is_not_in_the_chunk():
    client = FakeClient(tool_reply("submit_facts", FACTS))
    ex = kg.extract_facts(client, "m", [CHUNK])
    assert [c["service"] for c in ex.charges] == ["Replacement debit card", "Email alerts", "Minimum balance shortfall charge"]
    assert ex.dropped[0]["service"] == "Made-up courier fee" and ex.dropped[0]["reason"] == "quote not in chunk"
    req = client.requests[0]
    assert req["tool_choice"]["function"]["name"] == "submit_facts" and req["temperature"] == 0


def test_extraction_recovers_from_groq_tool_use_failed():
    exc = Exception("400 tool_use_failed")
    exc.body = {"error": {"code": "tool_use_failed",
                          "failed_generation": json.dumps({"name": "submit_facts", "arguments": FACTS})}}
    ex = kg.extract_facts(FakeClient(exc), "m", [CHUNK])
    assert len(ex.charges) == 3


def test_overlapping_chunks_are_merged():
    dup = [charge("Replacement debit card", 200, "q"), charge("replacement card", 200, "longer quote"),
           charge("Replacement debit card", 250, "q")]
    merged = kg.merge_charges(dup)
    assert len(merged) == 2 and merged[0]["quote"] == "longer quote"


@pytest.fixture
def graph():
    return kg.build_graph(kg.extract_facts(FakeClient(tool_reply("submit_facts", FACTS)), "m", [CHUNK]))


def test_graph_shape_and_queries(graph):
    tools = kg.GraphTools(graph)
    assert [c["service"] for c in tools.list_charges(min_rs=151)] == ["Replacement debit card"]
    assert [c["service"] for c in tools.list_charges(free_only=True)] == ["Email alerts"]
    ex = tools.find_exemptions("senior citizens aged 60")
    assert ex and ex[0]["exempt_from"]["service"] == "Minimum balance shortfall charge"
    assert tools.search_charges("debit card")[0]["amount_rs"] == 200


def test_answer_with_graph_uses_tools(graph):
    client = FakeClient(tool_reply("list_charges", {"min_rs": 101}),
                        text_reply("Replacement debit card (Rs 200) and the shortfall charge (Rs 150)."))
    out = kg.answer_with_graph(client, "m", kg.GraphTools(graph), "Which charges cost more than Rs 100?")
    assert out["tool_calls"][0] == {"name": "list_charges", "args": {"min_rs": 101}, "results": 2}
    sent = json.loads(client.requests[1]["messages"][-1]["content"])
    assert {c["service"] for c in sent} == {"Replacement debit card", "Minimum balance shortfall charge"}
    assert "Rs 200" in out["answer"]


def test_bad_tool_arguments_do_not_crash(graph):
    client = FakeClient(tool_reply("drop_table", {}), text_reply("Nothing found."))
    out = kg.answer_with_graph(client, "m", kg.GraphTools(graph), "q")
    assert out["answer"] == "Nothing found." and "unknown tool" in str(out["tool_calls"][0]["results"])


def golden_as_extraction():
    return [charge(g["service"], g.get("amount_rs"), g["quote"], free=g.get("amount_rs") == 0,
                   percent=g.get("percent")) for g in GOLDEN["charges"]]


def test_perfect_extraction_scores_100_percent():
    extracted = golden_as_extraction()
    ex = kg.Extraction(charges=extracted, exemptions=[
        {"group": e["group"], "charge_service": next(g["service"] for g in GOLDEN["charges"] if g["id"] == e["charge"]),
         "quote": e["quote"]} for e in GOLDEN["exemptions"]])
    g = kg.build_graph(ex)
    stats = K.score_extraction(kg.merge_charges(extracted), g, GOLDEN)
    assert stats["charges"]["f1"] == 1.0, stats
    assert stats["exemptions"]["f1"] == 1.0, stats


def test_wrong_amount_counts_as_found_but_not_correct():
    extracted = golden_as_extraction()
    extracted[8] = dict(extracted[8], amount_rs=250)       # replacement debit card: 200 -> 250
    extracted.append(charge("Courier fee", 99, "x"))
    full, name_only = K.match_charges(extracted, GOLDEN["charges"])
    assert len(full) == len(GOLDEN["charges"]) - 1 and list(name_only.values()) == ["c09"]
    stats = K.score_extraction(extracted, kg.build_graph(kg.Extraction(charges=extracted)), GOLDEN)
    n = len(GOLDEN["charges"])
    assert stats["charges"]["recall"] == round((n - 1) / n, 4) and stats["wrong_value"][0]["golden"] == "c09"
    assert stats["extra"] == ["Courier fee"]


def test_answer_scoring_patterns():
    q = next(x for x in GOLDEN["questions"] if x["id"] == "g1")
    good = ("The minimum balance shortfall charge (Rs 150), the closure charge of Rs 500, "
            "a replacement debit card (Rs 200) and each failed standing instruction (Rs 250).")
    s = K.score_answer(good, q["expected"], GOLDEN)
    assert s["recall"] == 1.0 and s["precision"] == 1.0
    partial = "A replacement debit card costs Rs 200. A stop-payment instruction costs Rs 100."
    s = K.score_answer(partial, q["expected"], GOLDEN)
    assert s["recall"] == 0.25 and s["wrong"] == ["c06"]
    assert "c02" not in K.mentioned("Rs 200 for a card", GOLDEN)     # Rs 20 must not match Rs 200
    assert K.mentioned("Email alerts are free; statements are emailed monthly", GOLDEN) >= {"c13", "c16"}


class PerfectExtractor:
    """Fake LLM that 'extracts' exactly the golden facts whose quote is in the chunk,
    and answers every graph question by listing all charges."""

    def __init__(self):
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        user = kw["messages"][-1]["content"]
        if kw.get("tool_choice"):
            chunk = squash(user)
            by_id = {g["id"]: g for g in GOLDEN["charges"]}
            facts = {"charges": [charge(g["service"], g.get("amount_rs"), g["quote"], free=g.get("amount_rs") == 0,
                                        percent=g.get("percent")) for g in GOLDEN["charges"] if squash(g["quote"]) in chunk],
                     "exemptions": [{"group": e["group"], "charge_service": by_id[e["charge"]]["service"], "quote": e["quote"]}
                                    for e in GOLDEN["exemptions"] if squash(e["quote"]) in chunk]}
            return tool_reply("submit_facts", facts)
        if kw["messages"][-1]["role"] == "user":
            return tool_reply("list_charges", {"min_rs": 101})
        results = json.loads(kw["messages"][-1]["content"])
        return text_reply("; ".join(f"{r['service']} Rs {r.get('amount_rs')}" for r in results))


def test_full_run_with_a_perfect_extractor(tmp_path, embed, scripted_qa):
    from ragbot.pipeline import RagPipeline, read_pdf
    text = read_pdf(PDF)
    rag = RagPipeline(embed, scripted_qa(default=("Rs 200", 0.9)), chunk_size=900, overlap=150, k=4).index_text(text)
    out = K.run(PerfectExtractor(), "fake", text, GOLDEN, rag_pipeline=rag, out_dir=str(tmp_path))
    assert out["stats"]["charges"]["recall"] == 1.0 and out["stats"]["charges"]["precision"] == 1.0
    assert out["stats"]["exemptions"]["f1"] == 1.0
    g1 = out["qa"][0]
    assert g1["graph"]["recall"] == 1.0 and g1["graph"]["precision"] == 1.0
    assert g1["rag"]["recall"] == 0.25
    assert (tmp_path / (out["run_id"] + ".md")).exists() and "Graph P" in out["report"]


def test_scorer_handles_the_unicode_models_write():
    """INC-004: gpt-oss wrote 'Rs\u202f500', '\u20b9100', 'stop\u2011payment' and curly apostrophes,
    and the first scorer missed those correct answers."""
    answer = ("- Account closure after 14\u202fdays \u2013 Rs\u202f500\n"
              "- Stop\u2011payment instruction on a cheque \u2013 \u20b9100 per cheque\n"
              "- Withdrawal at other banks\u2019 ATMs (after first three) \u2013 each")
    found = K.mentioned(answer, GOLDEN)
    assert {"c08", "c06", "c03"} <= found
    assert K.normalize_answer("\u20b975") == "Rs 75"


def test_rescore_uses_saved_answers_without_calling_the_llm(tmp_path):
    import networkx as nx
    extracted = golden_as_extraction()
    g = kg.build_graph(kg.Extraction(charges=extracted))
    saved = {"model": "m", "graph": kg.graph_to_json(g), "dropped": [], "questions": [
        {"id": "g5", "question": "q", "expected": ["c08"], "graph_answer": "Closure: Rs\u202f500",
         "graph_tool_calls": [], "rag_answer": "Not found in the document."}]}
    path = tmp_path / "run_kg.json"
    path.write_text(json.dumps(saved))
    r = K.rescore(str(path), GOLDEN)
    assert r["stats"]["charges"]["f1"] == 1.0
    assert r["qa"][0]["graph"]["recall"] == 1.0 and r["qa"][0]["rag"]["recall"] == 0.0
    assert (tmp_path / "run_kg_rescored.md").exists()
    assert isinstance(nx.node_link_graph(saved["graph"], directed=True, multigraph=True, edges="edges"), nx.MultiDiGraph)
