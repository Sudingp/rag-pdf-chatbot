"""
Named-entity recognition for the charge amounts in the bank PDF, scored against the
hand-checked facts in evals/data/golden_kg.json.

Task: find every rupee amount that is a charge (11 paid charges in the golden file; free
services and the percentage penalty are out of scope for an amount extractor). A
prediction is (amount, sentence); it is correct when the amount equals a golden charge's
amount and the sentence overlaps that charge's quote. Each golden charge can be matched
once, so a duplicate counts as a false positive.

Systems compared on exactly the same task and matcher:
  1. spaCy en_core_web_sm, MONEY entities as they come
  2. the same model with an EntityRuler that teaches it Indian currency ("Rs 150", "INR 20", "₹ 75")
  3. (2) plus a relation rule: keep an amount only if its sentence talks about a cost and the
     words just before it don't make it a limit or a threshold
  4. the LLM extraction (gpt-oss-120b via forced function calls) from a saved kg_eval run

    python -m evals.ner_eval [--kg-run evals/results/<stamp>_kg.json]
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(__file__)
PDF = os.path.join(HERE, "data", "kestrel_savings_terms.pdf")
GOLDEN = os.path.join(HERE, "data", "golden_kg.json")

CURRENCY = ["rs", "rs.", "inr", "₹"]
COST_CUES = {"cost", "costs", "charge", "charges", "charged", "fee", "fees", "penalty", "attracts",
             "levied", "levy", "payable"}
NOT_A_CHARGE_CUES = {"limit", "limits", "minimum", "maximum", "least", "up", "upto", "every", "balance",
                     "balances", "above", "below", "exceeding", "over", "deposit", "deposits", "of"}
LOOKBACK = 3   # words before the amount checked for NOT_A_CHARGE_CUES


def normalize(text: str) -> str:
    text = text.replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_amount(text: str) -> Optional[float]:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return round(float(m.group(0).replace(",", "")), 2) if m else None


def rupees(amount: float) -> str:
    return "Rs " + f"{amount:,.2f}".rstrip("0").rstrip(".")


def golden_paid_charges(path: str = GOLDEN) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    return [{"id": c["id"], "amount": round(float(c["amount_rs"]), 2), "quote": normalize(c["quote"])}
            for c in g["charges"] if c.get("amount_rs")]


def overlaps(sentence: str, quote: str) -> bool:
    s, q = normalize(sentence).lower().rstrip("."), normalize(quote).lower().rstrip(".")
    if q in s or s in q:
        return True
    sw, qw = set(re.findall(r"\w+", s)), set(re.findall(r"\w+", q))
    return bool(qw) and len(sw & qw) / len(qw) >= 0.6


@dataclass
class Result:
    name: str
    tp: int
    predicted: int
    expected: int
    false_positives: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.expected if self.expected else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score(name: str, predictions: List[Tuple[float, str]], golden: List[dict]) -> Result:
    matched, fps = set(), []
    for amount, sentence in predictions:
        hit = next((g["id"] for g in golden if g["id"] not in matched and g["amount"] == amount
                    and overlaps(sentence, g["quote"])), None)
        if hit:
            matched.add(hit)
        else:
            fps.append(f"{rupees(amount)}: {normalize(sentence)[:90]}")
    missed = [f'{g["id"]} {rupees(g["amount"])}' for g in golden if g["id"] not in matched]
    return Result(name, len(matched), len(predictions), len(golden), fps, missed)


# ---------------------------------------------------------------- spaCy systems

def load_nlp(with_currency_ruler: bool):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if with_currency_ruler:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([{"label": "MONEY",
                             "pattern": [{"LOWER": {"IN": CURRENCY}}, {"LIKE_NUM": True}]}])
    return nlp


def money_entities(doc) -> List[Tuple[float, str, object]]:
    out = []
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            amount = parse_amount(ent.text)
            if amount is not None:
                out.append((amount, ent.sent.text, ent))
    return out


def is_charge_amount(ent) -> bool:
    """Relation rule: the sentence must talk about a cost, and the few words right before
    the amount must not turn it into a limit or threshold ('up to Rs 40,000', 'a balance of
    Rs 5,000', 'for every Rs 100')."""
    sent_words = {t.lower_ for t in ent.sent}
    if not sent_words & COST_CUES:
        return False
    before = [t.lower_ for t in ent.doc[max(ent.sent.start, ent.start - LOOKBACK):ent.start]]
    if before and before[-1] == "of" and len(before) >= 2 and before[-2] in COST_CUES:
        return True                               # "a charge of Rs 150", "a penalty of Rs 1.50"
    return not set(before) & NOT_A_CHARGE_CUES


def spacy_predictions(text: str, ruler: bool, relation_rule: bool) -> List[Tuple[float, str]]:
    doc = load_nlp(ruler)(text)
    return [(a, s) for a, s, ent in money_entities(doc) if not relation_rule or is_charge_amount(ent)]


# ---------------------------------------------------------------- LLM extraction

def llm_predictions(kg_run: str) -> Tuple[str, List[Tuple[float, str]]]:
    with open(kg_run, encoding="utf-8") as f:
        run = json.load(f)
    nodes = [n for n in run["graph"]["nodes"] if n.get("kind") == "charge" and n.get("amount_rs")]
    return run.get("model", "LLM"), [(round(float(n["amount_rs"]), 2), n.get("quote", "")) for n in nodes]


def latest_kg_run() -> Optional[str]:
    runs = sorted(glob.glob(os.path.join(HERE, "results", "*_kg.json")))
    return runs[-1] if runs else None


def pdf_text(path: str = PDF) -> str:
    from pypdf import PdfReader
    return normalize(" ".join(p.extract_text() or "" for p in PdfReader(path).pages))


def run(kg_run: Optional[str] = None) -> List[Result]:
    text, golden = pdf_text(), golden_paid_charges()
    results = [
        score("spaCy en_core_web_sm (pretrained)", spacy_predictions(text, False, False), golden),
        score("+ currency EntityRuler", spacy_predictions(text, True, False), golden),
        score("+ currency EntityRuler + charge relation rule", spacy_predictions(text, True, True), golden),
    ]
    kg_run = kg_run or latest_kg_run()
    if kg_run:
        model, preds = llm_predictions(kg_run)
        results.append(score(f"LLM extraction ({model}, forced function calls)", preds, golden))
    return results


def to_markdown(results: List[Result]) -> str:
    lines = ["# NER for charge amounts", "",
             f"Golden: {results[0].expected} paid charges in rupees (evals/data/golden_kg.json).", "",
             "| System | Found | Correct | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {r.predicted} | {r.tp} | {r.precision:.1%} | {r.recall:.1%} | {r.f1:.1%} |")
    for r in results:
        if r.false_positives or r.missed:
            lines += ["", f"**{r.name}**"]
            lines += [f"- false positive: {fp}" for fp in r.false_positives]
            lines += [f"- missed: {m}" for m in r.missed]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-run", help="saved kg_eval JSON with the LLM's graph (default: latest)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ner_eval.md"))
    args = ap.parse_args()
    md = to_markdown(run(args.kg_run))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
