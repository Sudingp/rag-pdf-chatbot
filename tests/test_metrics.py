"""Tests for answer scoring and failure-mode classification."""

import pytest

from evals import metrics as M


def test_normalize_keeps_rates_and_decimals():
    assert M.normalize("The rate is 3.25% per annum.") == "rate is 3.25% per annum"
    assert M.normalize("Rs 1,00,000") == "rs 100000"


@pytest.mark.parametrize("pred,answers,em", [
    ("Rs 5,000", ["Rs 5,000"], 1.0),
    ("rs 5,000.", ["Rs 5,000"], 1.0),
    ("the Principal Nodal Officer", ["Principal Nodal Officer"], 1.0),
    ("Rs 50", ["Rs 500"], 0.0),
    ("3.75%", ["3.25%", "3.25% per annum"], 0.0),
])
def test_exact_match(pred, answers, em):
    assert M.exact_match(pred, answers) == em


def test_f1_partial_and_best_of_answers():
    assert M.f1_score("Rs 150", ["Rs 150 plus applicable taxes"]) == pytest.approx(2 * 1 * 0.4 / 1.4)  # P=2/2, R=2/5
    assert M.f1_score("Rs 150", ["Rs 150 plus applicable taxes", "Rs 150"]) == 1.0
    assert M.f1_score("anything", []) == 0.0


def test_evidence_retrieved_ignores_line_breaks_and_case():
    ctx = "The daily ATM cash\nwithdrawal limit is RS 40,000. Next"
    assert M.evidence_retrieved("The daily ATM cash withdrawal limit is Rs 40,000.", ctx) is True
    assert M.evidence_retrieved("Something else.", ctx) is False
    assert M.evidence_retrieved("", ctx) is None


def C(category, evidence="e"):
    return {"category": category, "evidence": evidence}


@pytest.mark.parametrize("case,abstained,retrieved,em,f1,mode", [
    (C("unanswerable", ""), True, None, 0, 0, M.CORRECT_ABSTAIN),
    (C("unanswerable", ""), False, None, 0, 0, M.HALLUCINATION),
    (C("numeric"), True, True, 0, 0, M.WRONG_ABSTAIN),
    (C("numeric"), False, True, 1, 1, M.CORRECT),
    (C("numeric"), False, True, 0, 0.85, M.CORRECT),
    (C("numeric"), False, True, 0, 0.5, M.PARTIAL),
    (C("numeric"), False, False, 0, 0, M.RETRIEVAL_MISS),
    (C("numeric"), False, True, 0, 0, M.EXTRACTION_ERROR),
    (C("reasoning", ""), False, None, 0, 0, M.REASONING_GAP),
    (C("adversarial", ""), True, None, 0, 0, M.CORRECT_ABSTAIN),
    (C("adversarial", ""), False, None, 0, 0, M.GUARDRAIL_BREACH),
])
def test_classify(case, abstained, retrieved, em, f1, mode):
    assert M.classify(case, "x", abstained, retrieved, em, f1) == mode


def test_ungrounded_wrong_answer_is_flagged_as_made_up():
    assert M.classify(C("numeric"), "Rs 75", False, True, 0, 0, is_grounded=False) == M.UNGROUNDED
    # a correct answer is correct even if reworded (not grounded verbatim)
    assert M.classify(C("numeric"), "Rs 75", False, True, 1, 1, is_grounded=False) == M.CORRECT


def test_grounded():
    ctx = "A replacement debit card costs Rs 200 and is delivered within 7 working days."
    assert M.grounded("Rs 200", ctx)
    assert M.grounded("rs 200.", ctx)
    assert not M.grounded("Rs 250", ctx)
    assert not M.grounded("", ctx)
