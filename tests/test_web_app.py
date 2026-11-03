"""The public-deployment guards: password gate, question and upload limits, default reader.
Uses Streamlit's AppTest to run the real app.py without a browser or models."""

import os

import pytest

from ragbot import web_guard as guard

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_settings_read_from_environment():
    env = {"APP_PASSWORD": "s3cret", "MAX_QUESTIONS_PER_SESSION": "5", "MAX_UPLOAD_MB": "4", "DEFAULT_READER": "llm"}
    assert guard.password_required(env) and guard.password_ok("s3cret", env) and not guard.password_ok("s3cre", env)
    assert guard.question_limit(env) == 5 and guard.upload_limit_mb(env) == 4 and guard.default_reader(env) == "llm"


def test_safe_defaults_for_missing_or_bad_values():
    assert not guard.password_required({}) and guard.password_ok("anything", {})
    bad = {"MAX_QUESTIONS_PER_SESSION": "lots", "MAX_UPLOAD_MB": "-3", "DEFAULT_READER": "gpt-9"}
    assert guard.question_limit(bad) == 20 and guard.upload_limit_mb(bad) == 1 and guard.default_reader(bad) == "extractive"
    assert guard.question_limit({"MAX_QUESTIONS_PER_SESSION": "0"}) == 1


streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def run_app(monkeypatch, **env):
    for k in ("APP_PASSWORD", "MAX_UPLOAD_MB", "DEFAULT_READER"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_without_password_the_uploader_shows(monkeypatch):
    at = run_app(monkeypatch, MAX_UPLOAD_MB="7")
    assert not at.exception
    assert len(at.text_input) == 0                       # no password box
    assert at.radio[0].value == "extractive"


def test_password_gate_blocks_until_correct(monkeypatch):
    at = run_app(monkeypatch, APP_PASSWORD="open-sesame", DEFAULT_READER="llm")
    assert at.text_input[0].label == "Password" and len(at.radio) == 0
    at.text_input[0].input("wrong").run()
    assert at.error[0].value == "Wrong password." and len(at.radio) == 0
    at.text_input[0].input("open-sesame").run()
    assert not at.exception and at.radio[0].value == "llm"   # unlocked, settings visible
