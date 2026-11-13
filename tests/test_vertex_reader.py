import json
from types import SimpleNamespace

import pytest

from ragbot import pipeline, web_guard
from ragbot.vertex_reader import VertexClient, load_vertex_reader, vertex_base_url
from ragbot.azure_reader import make_azure_reader


class FakeCreds:
    """Stands in for google.auth credentials: invalid until refreshed, then valid."""

    def __init__(self):
        self.token, self.valid, self.refresh_count = None, False, 0

    def refresh(self, request):
        self.refresh_count += 1
        self.token, self.valid = f"token-{self.refresh_count}", True


class FakeOpenAI:
    def __init__(self, reply):
        self.api_key, self.seen_keys, self.reply = "unset", [], reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.seen_keys.append(self.api_key)
        return self.reply


def json_reply(answer, quote, confidence="high"):
    content = json.dumps({"answer": answer, "found_in_context": True, "quote": quote, "confidence": confidence})
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                                                    message=SimpleNamespace(content=content, tool_calls=None))])


def test_base_url_global_and_regional():
    assert vertex_base_url("p1") == \
        "https://aiplatform.googleapis.com/v1/projects/p1/locations/global/endpoints/openapi"
    assert vertex_base_url("p1", "asia-south1").startswith("https://asia-south1-aiplatform.googleapis.com/")


def test_token_is_fetched_once_and_refreshed_only_when_expired():
    creds, fake = FakeCreds(), FakeOpenAI(reply="ok")
    client = VertexClient(creds, "https://x", openai_client=fake, request_factory=object)
    client.chat.completions.create(model="m", messages=[])
    client.chat.completions.create(model="m", messages=[])
    assert fake.seen_keys == ["token-1", "token-1"] and client.refreshes == 1
    creds.valid = False                                   # an hour later: token expired
    client.chat.completions.create(model="m", messages=[])
    assert fake.seen_keys[-1] == "token-2" and client.refreshes == 2


def test_reader_logic_is_shared_with_the_other_hosted_readers():
    context = "A late payment fee of Rs 500 applies to bills paid after the due date."
    fake = FakeOpenAI(reply=json_reply("Rs 500", "A late payment fee of Rs 500 applies"))
    client = VertexClient(FakeCreds(), "https://x", openai_client=fake, request_factory=object)
    reader = make_azure_reader(client, "google/gemini-3.5-flash")
    out = reader("What is the late payment fee?", context)
    assert out["answer"] == "Rs 500" and not out["refused"]
    assert out["trace"]["model"] == "google/gemini-3.5-flash" and out["trace"]["grounding"] == "quote"
    assert fake.seen_keys == ["token-1"]                  # the request carried the OAuth token


def test_missing_credentials_give_a_clear_message(monkeypatch):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def no_creds(**kw):
        raise DefaultCredentialsError("none")
    monkeypatch.setattr(google.auth, "default", no_creds)
    with pytest.raises(RuntimeError, match="application-default login"):
        load_vertex_reader()


def test_vertex_is_offered_everywhere_readers_are_listed():
    assert "vertex" in pipeline.READERS and "vertex" in web_guard.READERS
    assert web_guard.default_reader({"DEFAULT_READER": "vertex"}) == "vertex"


def test_gemini_thought_signature_is_sent_back_with_the_tool_call():
    """Gemini 3 rejects a follow-up turn whose tool call lost its thought signature (HTTP 400)."""
    sig = {"google": {"thought_signature": "abc123"}}
    call = SimpleNamespace(id="c1", type="function", model_extra={"extra_content": sig},
                           function=SimpleNamespace(name="calculate", arguments=json.dumps({"expression": "2*250"})))
    first = SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls",
                                                     message=SimpleNamespace(content=None, tool_calls=[call]))])
    second = json_reply("Rs 500", "A late payment fee of Rs 500 applies")
    sent = []

    class Seq(FakeOpenAI):
        def create(self, **kw):
            sent.append(json.loads(json.dumps(kw["messages"], default=str)))
            return [first, second][len(sent) - 1]

    client = VertexClient(FakeCreds(), "https://x", openai_client=Seq(reply=None), request_factory=object)
    reader = make_azure_reader(client, "google/gemini-3.5-flash")
    out = reader("What is the late payment fee?", "A late payment fee of Rs 500 applies to bills paid late.")
    assert out["answer"] == "Rs 500"
    replayed = next(m for m in sent[1] if m["role"] == "assistant")["tool_calls"][0]
    assert replayed["extra_content"] == sig and replayed["function"]["name"] == "calculate"
