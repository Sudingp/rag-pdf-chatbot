import pytest

from evals.ner_eval import (golden_paid_charges, is_charge_amount, load_nlp, money_entities,
                            overlaps, parse_amount, rupees, score)


def test_amounts_parse_indian_digit_grouping_and_decimals():
    assert parse_amount("Rs 1,00,000") == 100000
    assert parse_amount("Rs 1.50") == 1.5
    assert rupees(100000) == "Rs 100,000" and rupees(1.5) == "Rs 1.5"


def test_golden_has_only_paid_rupee_charges():
    paid = golden_paid_charges()
    assert len(paid) == 11 and all(g["amount"] > 0 for g in paid)


def test_sentence_must_overlap_the_golden_quote():
    q = "Each additional cash withdrawal costs Rs 20."
    assert overlaps("Each additional cash withdrawal costs Rs 20.", q)
    assert not overlaps("The daily ATM cash withdrawal limit is Rs 40,000.", q)


def test_scoring_counts_wrong_amount_and_duplicates_as_false_positives():
    golden = [{"id": "c1", "amount": 20.0, "quote": "Each additional cash withdrawal costs Rs 20."}]
    preds = [(20.0, "Each additional cash withdrawal costs Rs 20."),
             (20.0, "Each additional cash withdrawal costs Rs 20."),      # duplicate
             (21.0, "Each additional cash withdrawal costs Rs 21.")]      # wrong amount
    r = score("x", preds, golden)
    assert (r.tp, r.predicted, r.expected) == (1, 3, 1)
    assert r.precision == pytest.approx(1 / 3) and r.recall == 1.0


def _nlp(ruler):
    spacy = pytest.importorskip("spacy")
    try:
        return load_nlp(ruler)
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not installed")


TEXT = ("A replacement debit card costs Rs 200. The daily ATM cash withdrawal limit is Rs 40,000. "
        "A missed instalment attracts a penalty of Rs 1.50 for every Rs 100 of the instalment.")


def test_pretrained_model_does_not_know_rupees_but_the_ruler_does():
    plain = money_entities(_nlp(False)(TEXT))
    ruled = money_entities(_nlp(True)(TEXT))
    assert not [a for a, _, _ in plain if a in (200, 1.5)]
    assert {200.0, 40000.0, 1.5, 100.0} <= {a for a, _, _ in ruled}


def test_relation_rule_keeps_charges_and_drops_limits_and_thresholds():
    kept = {a for a, _, ent in money_entities(_nlp(True)(TEXT)) if is_charge_amount(ent)}
    assert kept == {200.0, 1.5}
