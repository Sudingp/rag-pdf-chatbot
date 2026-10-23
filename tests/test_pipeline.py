"""Tests for chunking, retrieval and abstention in the RAG pipeline."""

import random
import string

import pytest

from ragbot.pipeline import NO_ANSWER, RagPipeline, build_prompt, chunk_text, is_refusal, load_models


def original_chunk_text(text, size=900):
    """The exact function from the original app.py."""
    return [text[i:i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize("n", [0, 1, 899, 900, 901, 5000])
def test_default_chunking_matches_original_app(n):
    text = "".join(random.Random(n).choices(string.ascii_letters + " ", k=n))
    assert chunk_text(text) == original_chunk_text(text)


def test_overlap_keeps_boundary_sentence_whole():
    sentence = "The daily ATM cash withdrawal limit is Rs 40,000."
    text = "x" * 880 + sentence + "y" * 500
    assert not any(sentence in c for c in chunk_text(text, 900, 0))   # split by the boundary
    assert any(sentence in c for c in chunk_text(text, 900, 150))     # whole with overlap


def test_overlap_chunks_cover_all_text():
    text = "".join(str(i % 10) for i in range(2000))
    chunks = chunk_text(text, 300, 50)
    assert chunks[0] == text[:300]
    assert chunks[1].startswith(text[250:260])
    assert chunks[-1].endswith(text[-20:])


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_settings_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", size, overlap)


def test_retrieves_relevant_chunk(embed, scripted_qa):
    text = ("Cheque books cost Rs 75 each. " * 10) + ("The UPI daily limit is Rs 1,00,000. " * 10)
    rag = RagPipeline(embed, scripted_qa(), chunk_size=300, k=1).index_text(text)
    ids = rag.retrieve("What is the UPI daily limit?")
    assert "UPI" in rag.chunks[ids[0]]


def test_k_larger_than_chunk_count_is_safe(embed, scripted_qa):
    rag = RagPipeline(embed, scripted_qa(), chunk_size=900, k=6).index_text("short text")
    assert rag.retrieve("anything") == [0]


def test_always_answers_by_default(embed, scripted_qa):
    rag = RagPipeline(embed, scripted_qa(default=("Rs 20", 0.01))).index_text("some text")
    ans = rag.ask("q")
    assert ans.text == "Rs 20" and not ans.abstained


def test_abstains_below_min_score(embed, scripted_qa):
    rag = RagPipeline(embed, scripted_qa(default=("Rs 20", 0.2)), min_score=0.3).index_text("t")
    ans = rag.ask("q")
    assert ans.abstained and ans.text == NO_ANSWER and ans.score == 0.2


def test_empty_document_raises(embed, scripted_qa):
    with pytest.raises(ValueError, match="No text"):
        RagPipeline(embed, scripted_qa()).index_text("")


def test_ask_before_index_raises(embed, scripted_qa):
    with pytest.raises(RuntimeError):
        RagPipeline(embed, scripted_qa()).ask("q")


@pytest.mark.parametrize("text,refusal", [
    ("unanswerable", True), ("Unanswerable.", True), ("Not in the context", True),
    ("I don't know", True), ("unanswerable from the context", True),
    ("Rs 200", False), ("No charge is levied for reactivation", False), ("quarterly", False),
])
def test_is_refusal(text, refusal):
    assert is_refusal(text) is refusal


def test_prompt_contains_context_question_and_refusal_instruction():
    p = build_prompt("What is the ATM limit?", "The daily ATM cash withdrawal limit is Rs 40,000.")
    assert "Rs 40,000" in p and "What is the ATM limit?" in p and "unanswerable" in p
    assert p.index("Context:") < p.index("Question:")


def test_generative_refusal_becomes_abstention(embed):
    reader = lambda q, c: {"answer": "unanswerable", "score": 0.95, "refused": True}  # noqa: E731
    ans = RagPipeline(embed, reader, chunk_size=500, k=3).index_text("text").ask("q")
    assert ans.abstained and ans.text == NO_ANSWER and ans.raw == "unanswerable"


def test_truncation_flag_is_passed_through(embed):
    reader = lambda q, c: {"answer": "Rs 20", "score": 0.9, "truncated": True}  # noqa: E731
    assert RagPipeline(embed, reader).index_text("text").ask("q").truncated


def test_unknown_reader_rejected():
    with pytest.raises(ValueError, match="Unknown reader"):
        load_models("gpt")
