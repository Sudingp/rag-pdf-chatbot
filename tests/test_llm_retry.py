"""Rate-limited calls are retried after the wait the provider asks for."""

import pytest

from ragbot.llm_retry import RetryingClient, suggested_wait


class RateLimit(Exception):
    status_code = 429


GROQ_MSG = ("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` ... tokens per minute (TPM): "
            "Limit 8000, Used 5704, Requested 3521. Please try again in 9.1875s.")


class Flaky:
    def __init__(self, failures):
        self.failures, self.calls = list(failures), 0
        from types import SimpleNamespace as NS
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def test_parses_groq_wait_times():
    assert suggested_wait(Exception(GROQ_MSG)) == 9.1875
    assert suggested_wait(Exception("Please try again in 1m30.5s")) == 90.5
    assert suggested_wait(Exception("try again in 450ms")) == 0.45
    assert suggested_wait(Exception("server error")) == 0.0


def test_waits_as_told_then_succeeds():
    waits = []
    c = RetryingClient(Flaky([RateLimit(GROQ_MSG), RateLimit("no hint")]), sleep=waits.append, log=lambda m: None)
    assert c.chat.completions.create(model="m") == "ok"
    assert waits == [pytest.approx(10.1875), 5.0]       # provider's hint + 1s, then backoff 2*2^1 + 1
    assert c.retries == 2


def test_other_errors_and_exhausted_retries_raise():
    c = RetryingClient(Flaky([ValueError("bad request")]), sleep=lambda s: None, log=lambda m: None)
    with pytest.raises(ValueError):
        c.chat.completions.create()
    c = RetryingClient(Flaky([RateLimit(GROQ_MSG)] * 3), max_retries=2, sleep=lambda s: None, log=lambda m: None)
    with pytest.raises(RateLimit):
        c.chat.completions.create()
