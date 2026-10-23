"""
Fake models so the harness and pipeline can be tested without downloading
anything: a deterministic bag-of-words embedder and a scripted QA model.
"""

import hashlib
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIM = 256


def fake_embed(texts):
    out = np.zeros((len(texts), DIM), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


class ScriptedQA:
    """Returns a scripted answer per question; unknown questions get a default."""

    def __init__(self, script=None, default=("Rs 999", 0.05)):
        self.script = script or {}
        self.default = default
        self.calls = []

    def __call__(self, question, context):
        self.calls.append((question, context))
        answer, score = self.script.get(question, self.default)
        return {"answer": answer, "score": score}


@pytest.fixture
def embed():
    return fake_embed


@pytest.fixture
def scripted_qa():
    return ScriptedQA
