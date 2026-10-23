"""The golden set must be correct, or every score built on it is wrong."""

import copy
import os

import pytest

from evals.validate_golden import load_golden, validate
from ragbot.pipeline import read_pdf

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals", "data")


@pytest.fixture(scope="module")
def doc_text():
    return read_pdf(os.path.join(DATA, "kestrel_savings_terms.pdf"))


@pytest.fixture(scope="module")
def cases():
    return load_golden(os.path.join(DATA, "golden.jsonl"))


def test_golden_set_is_valid_against_the_pdf(cases, doc_text):
    assert validate(cases, doc_text) == []


def test_golden_set_covers_every_category(cases):
    counts = {}
    for c in cases:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    assert counts.get("unanswerable", 0) >= 5
    assert counts.get("distractor", 0) >= 5
    assert counts.get("paraphrase", 0) >= 5
    assert len(cases) >= 40


def test_validator_catches_tampered_evidence(cases, doc_text):
    bad = copy.deepcopy(cases)
    bad[0]["evidence"] = bad[0]["evidence"].replace("18", "21")
    assert any("not found verbatim" in e for e in validate(bad, doc_text))


def test_validator_catches_answer_missing_from_evidence(cases, doc_text):
    bad = copy.deepcopy(cases)
    bad[0]["answers"] = ["21 years"]
    assert any("no expected answer" in e for e in validate(bad, doc_text))


def test_validator_catches_duplicates_and_bad_unanswerable(cases, doc_text):
    bad = copy.deepcopy(cases) + [copy.deepcopy(cases[0])]
    bad.append({"id": "u-x", "category": "unanswerable", "question": "q", "answers": ["x"], "evidence": ""})
    errors = validate(bad, doc_text)
    assert any("duplicate id" in e for e in errors)
    assert any("must have no answers" in e for e in errors)
