"""
Retries rate-limited LLM calls (HTTP 429) instead of crashing the run.

Groq's free tier allows 8,000 tokens per minute; the knowledge-graph run sends one
chunk per call, and the first live run stopped at question g3 with
"Rate limit reached ... tokens per minute ... Please try again in 9.1875s".
The error says how long to wait, so we wait that long (plus a little), and fall
back to exponential backoff when it doesn't.

Wrap any OpenAI-compatible client:  client = RetryingClient(OpenAI(...))
"""

import re
import time
from types import SimpleNamespace
from typing import Callable

_WAIT = re.compile(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)\s*(ms|s)", re.IGNORECASE)


def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "rate limit" in str(exc).lower()


def suggested_wait(exc: Exception) -> float:
    """Seconds the provider asked us to wait, or 0 if it didn't say."""
    m = _WAIT.search(str(exc))
    if not m:
        return 0.0
    minutes, value, unit = m.groups()
    seconds = float(value) / 1000 if unit.lower() == "ms" else float(value)
    return seconds + 60 * int(minutes or 0)


class RetryingClient:
    def __init__(self, client, max_retries: int = 6, base_delay: float = 2.0, max_delay: float = 90.0,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print):
        self._client, self.max_retries = client, max_retries
        self.base_delay, self.max_delay = base_delay, max_delay
        self._sleep, self._log = sleep, log
        self.retries = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if not is_rate_limit(exc) or attempt == self.max_retries:
                    raise
                wait = suggested_wait(exc) or self.base_delay * 2 ** attempt
                wait = min(wait + 1.0, self.max_delay)
                self.retries += 1
                self._log(f"    rate limited; waiting {wait:.1f}s (retry {attempt + 1}/{self.max_retries})")
                self._sleep(wait)
