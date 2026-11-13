"""
Prometheus metrics for the web app (backlog story S6: usage analytics without storing questions).

Only counts, timings and categories are recorded. The question text, the PDF and the answer
never become metric labels, so nothing a user typed leaves the app through monitoring.

    questions_total{reader, outcome}          outcome: answered | not_found | refused | error
    answer_latency_seconds{reader}            histogram
    guardrail_blocks_total{reader, reason}    why an answer was withheld
    pdf_uploads_total{result}                 accepted | too_large

Exposed on METRICS_PORT (default 9100; set 0 to turn off) at /metrics.
"""

import os
import threading
from typing import Optional

from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

OUTCOMES = ("answered", "not_found", "refused", "error")
# Buckets around what the golden set showed: median 8.4 s, slowest 25 s.
LATENCY_BUCKETS = (0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 60)


class AppMetrics:
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.questions = Counter("ragbot_questions_total", "Questions answered by the app",
                                 ["reader", "outcome"], registry=self.registry)
        self.latency = Histogram("ragbot_answer_latency_seconds", "Time to produce an answer",
                                 ["reader"], buckets=LATENCY_BUCKETS, registry=self.registry)
        self.guardrail = Counter("ragbot_guardrail_blocks_total", "Answers withheld by a guardrail",
                                 ["reader", "reason"], registry=self.registry)
        self.uploads = Counter("ragbot_pdf_uploads_total", "PDF uploads", ["result"],
                               registry=self.registry)
        for reader in ("extractive", "generative", "llm", "azure", "vertex"):   # show zeros, not "no data"
            for outcome in OUTCOMES:
                self.questions.labels(reader, outcome)
            self.latency.labels(reader)
        for result in ("accepted", "too_large"):
            self.uploads.labels(result)

    def record_answer(self, reader: str, answer, seconds: float) -> str:
        outcome = classify(answer)
        self.questions.labels(reader, outcome).inc()
        self.latency.labels(reader).observe(seconds)
        reason = (getattr(answer, "trace", None) or {}).get("guardrail")
        if outcome == "refused" and reason:
            self.guardrail.labels(reader, short_reason(reason)).inc()
        return outcome

    def record_error(self, reader: str, seconds: float) -> None:
        self.questions.labels(reader, "error").inc()
        self.latency.labels(reader).observe(seconds)

    def record_upload(self, accepted: bool) -> None:
        self.uploads.labels("accepted" if accepted else "too_large").inc()


# The reader also writes this into trace["guardrail"] when the model itself says the answer is
# not in the document. That is an ordinary "not found", not a guardrail stepping in.
NOT_IN_CONTEXT = "model said not in context"

# Reason strings written by ragbot/azure_reader.py, mapped to a small fixed label set
# (free text as a label would create a new time series per message).
REASONS = (("possible prompt injection", "prompt_injection"),
           ("evidence quote not found", "quote_not_in_context"),
           ("content filter", "content_filter"),
           ("not valid json", "invalid_json"),
           ("too many tool calls", "too_many_tool_calls"),
           ("empty question", "bad_input"),
           ("question longer than", "bad_input"))


def classify(answer) -> str:
    if not getattr(answer, "abstained", False):
        return "answered"
    reason = (getattr(answer, "trace", None) or {}).get("guardrail")
    if reason and reason != NOT_IN_CONTEXT:
        return "refused"
    return "not_found"


def short_reason(reason: str) -> str:
    r = str(reason).lower()
    return next((label for text, label in REASONS if text in r), "other")


_lock = threading.Lock()
_started = {}


def start_exporter(metrics: AppMetrics, port: Optional[int] = None) -> Optional[int]:
    """Start the /metrics HTTP server once per process. Returns the port, or None if disabled."""
    port = int(os.environ.get("METRICS_PORT", "9100")) if port is None else port
    if port == 0:
        return None
    with _lock:
        if port not in _started:
            start_http_server(port, registry=metrics.registry)
            _started[port] = True
    return port
