import socket
import urllib.request
from types import SimpleNamespace

from ragbot.metrics import AppMetrics, NOT_IN_CONTEXT, classify, short_reason, start_exporter


def ans(abstained=False, guardrail=None):
    return SimpleNamespace(abstained=abstained, trace={"guardrail": guardrail} if guardrail else {})


def value(m, name, **labels):
    v = m.registry.get_sample_value(name, labels)
    return 0.0 if v is None else v


def test_outcomes_separate_not_found_from_guardrail_refusals():
    assert classify(ans()) == "answered"
    assert classify(ans(True)) == "not_found"                         # low score / no evidence
    assert classify(ans(True, NOT_IN_CONTEXT)) == "not_found"         # the model said so itself
    assert classify(ans(True, "possible prompt injection")) == "refused"
    assert classify(ans(True, "evidence quote not found in retrieved context")) == "refused"


def test_guardrail_reasons_map_to_a_fixed_label_set():
    assert short_reason("possible prompt injection") == "prompt_injection"
    assert short_reason("evidence quote not found in retrieved context") == "quote_not_in_context"
    assert short_reason("blocked by provider content filter (prompt)") == "content_filter"
    assert short_reason("question longer than 500 characters") == "bad_input"
    assert short_reason("something new") == "other"


def test_recording_counts_outcomes_latency_and_reasons():
    m = AppMetrics()
    m.record_answer("llm", ans(), 3.2)
    m.record_answer("llm", ans(True, "possible prompt injection"), 0.4)
    m.record_answer("llm", ans(True, NOT_IN_CONTEXT), 7.0)
    m.record_error("llm", 1.0)
    m.record_upload(True)
    m.record_upload(False)
    assert value(m, "ragbot_questions_total", reader="llm", outcome="answered") == 1
    assert value(m, "ragbot_questions_total", reader="llm", outcome="refused") == 1
    assert value(m, "ragbot_questions_total", reader="llm", outcome="not_found") == 1
    assert value(m, "ragbot_questions_total", reader="llm", outcome="error") == 1
    assert value(m, "ragbot_guardrail_blocks_total", reader="llm", reason="prompt_injection") == 1
    assert value(m, "ragbot_answer_latency_seconds_count", reader="llm") == 4
    assert value(m, "ragbot_answer_latency_seconds_bucket", reader="llm", le="4.0") == 3
    assert value(m, "ragbot_pdf_uploads_total", result="too_large") == 1


def test_exporter_serves_metrics_and_never_the_question_text():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    m = AppMetrics()
    secret_question = "What is my account number 1234?"
    m.record_answer("llm", SimpleNamespace(abstained=False, trace={}, text=secret_question), 2.0)
    assert start_exporter(m, port) == port
    assert start_exporter(m, port) == port                 # second call is a no-op, not a crash
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    assert 'ragbot_questions_total{outcome="answered",reader="llm"} 1.0' in body
    assert "account number" not in body and "1234" not in body


def test_port_zero_turns_the_exporter_off():
    assert start_exporter(AppMetrics(), 0) is None
