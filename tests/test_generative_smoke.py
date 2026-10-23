"""
Smoke test for the real Flan-T5 code path, without downloading Flan-T5.

A tiny randomly initialised T5 (a few thousand parameters) stands in for the
real model, and a toy tokenizer stands in for the real one. The answers are
gibberish, but the test proves the generate -> decode -> confidence ->
truncation code runs and returns the shape RagPipeline expects.

Skipped automatically when torch/transformers are not installed (e.g. in CI).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ragbot import pipeline as P  # noqa: E402


class ToyTokenizer:
    """Maps characters to ids; enough for generate() and decode()."""

    vocab_size = 64

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        ids = [3 + (ord(ch) % (self.vocab_size - 3)) for ch in text]
        if truncation:
            ids = ids[:max_length]
        ids = ids + [1]  # EOS
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (int(i) % 26)) for i in ids if int(i) > 2)


@pytest.fixture
def tiny_t5(monkeypatch):
    torch.manual_seed(0)
    cfg = transformers.T5Config(vocab_size=64, d_model=16, d_ff=32, d_kv=8, num_layers=1,
                                num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
                                pad_token_id=0, eos_token_id=1)
    model = transformers.T5ForConditionalGeneration(cfg)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: ToyTokenizer())
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", lambda *a, **k: model)
    return P.load_generative_reader("tiny-test-model", max_new_tokens=5)


def test_generative_reader_returns_expected_fields(tiny_t5):
    out = tiny_t5("What is the ATM limit?", "The daily ATM cash withdrawal limit is Rs 40,000.")
    assert set(out) == {"answer", "score", "refused", "truncated"}
    assert isinstance(out["answer"], str)
    assert 0.0 < out["score"] <= 1.0          # geometric-mean token probability
    assert out["truncated"] is False


def test_long_context_is_flagged_as_truncated(tiny_t5):
    out = tiny_t5("q?", "x" * 5000)
    assert out["truncated"] is True


def test_generative_reader_plugs_into_pipeline(tiny_t5, embed):
    rag = P.RagPipeline(embed, tiny_t5, chunk_size=500, k=3).index_text("Cheque books cost Rs 75. " * 40)
    ans = rag.ask("How much is a cheque book?")
    assert isinstance(ans.text, str) and 0.0 < ans.score <= 1.0


# ---------------------------------------------------------------------- #
# Extractive reader (regression for INC-001: transformers 5 removed the
# "question-answering" pipeline, so the app now runs the model directly)
# ---------------------------------------------------------------------- #
@pytest.fixture
def tiny_extractive(tmp_path):
    words = ("the daily atm cash withdrawal limit is rs 40 , 000 . what ? cheque books cost 75 "
             "each how much a").split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(words))
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=True)
    torch.manual_seed(0)
    model = transformers.DistilBertForQuestionAnswering(transformers.DistilBertConfig(
        vocab_size=len(vocab), dim=16, hidden_dim=32, n_layers=1, n_heads=2, max_position_embeddings=512))
    model.eval()
    return tok, model


def test_extract_answer_returns_a_span_of_the_context(tiny_extractive):
    tok, model = tiny_extractive
    ctx = "The daily ATM cash withdrawal limit is Rs 40,000."
    out = P.extract_answer(tok, model, "What is the daily ATM limit?", ctx)
    assert out["answer"] == ctx[out["start"]:out["end"]].strip()
    assert out["answer"] and out["answer"] in ctx
    assert 0.0 < out["score"] <= 1.0


def test_extract_answer_handles_context_longer_than_one_window(tiny_extractive):
    tok, model = tiny_extractive
    ctx = "Cheque books cost Rs 75 each. " * 120   # far beyond 384 tokens
    out = P.extract_answer(tok, model, "How much is a cheque book?", ctx, max_length=64, stride=16)
    assert out["answer"] in ctx and 0.0 < out["score"] <= 1.0


def test_answer_never_includes_question_tokens(tiny_extractive):
    tok, model = tiny_extractive
    out = P.extract_answer(tok, model, "what what what what", "cheque books cost 75")
    assert out["answer"] in "cheque books cost 75"
