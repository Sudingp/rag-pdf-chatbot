"""
Guards for running the app on a public URL.

A deployed app answers with a paid or rate-limited LLM key, so anyone who finds
the URL could use it up. Two controls, both set by environment variables on the
cloud service (never in the image or the repo):

    APP_PASSWORD               if set, visitors must enter it before using the app
    MAX_QUESTIONS_PER_SESSION  questions one browser session may ask (default 20)
    MAX_UPLOAD_MB              largest PDF accepted (default 10)
    DEFAULT_READER             reader selected when the page opens (default extractive)
"""

import hmac
import os
from typing import Mapping, Optional

READERS = ("extractive", "generative", "llm", "azure")


def setting(name: str, default: str, env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    value = env.get(name, "").strip()
    return value or default


def password_required(env: Optional[Mapping[str, str]] = None) -> bool:
    return bool(setting("APP_PASSWORD", "", env))


def password_ok(attempt: str, env: Optional[Mapping[str, str]] = None) -> bool:
    expected = setting("APP_PASSWORD", "", env)
    if not expected:
        return True
    # Constant-time comparison, so response time doesn't leak how much of the guess was right.
    return hmac.compare_digest((attempt or "").encode(), expected.encode())


def question_limit(env: Optional[Mapping[str, str]] = None) -> int:
    try:
        return max(1, int(setting("MAX_QUESTIONS_PER_SESSION", "20", env)))
    except ValueError:
        return 20


def upload_limit_mb(env: Optional[Mapping[str, str]] = None) -> int:
    try:
        return max(1, int(setting("MAX_UPLOAD_MB", "10", env)))
    except ValueError:
        return 10


def default_reader(env: Optional[Mapping[str, str]] = None) -> str:
    r = setting("DEFAULT_READER", "extractive", env)
    return r if r in READERS else "extractive"
