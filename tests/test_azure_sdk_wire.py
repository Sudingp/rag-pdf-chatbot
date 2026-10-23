"""
Runs the Azure reader through the REAL openai SDK (AzureOpenAI client) with a
mocked HTTP transport, so the exact request Azure would receive is checked and
real SDK response objects are parsed. Still no network and no API key.
"""

import json

import pytest

openai = pytest.importorskip("openai")
try:  # openai SDK 3.x ships httpx2; older versions use httpx
    import httpx2 as httpx
except ImportError:
    httpx = pytest.importorskip("httpx")

from ragbot.azure_reader import make_azure_reader  # noqa: E402

CONTEXT = ("The fixed deposit rate for a tenure of 1 year to 2 years is 6.80% per annum. "
           "Senior citizens aged 60 or above receive an additional 0.50% per annum.")


def completion(message, finish):
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def test_real_sdk_request_and_tool_round_trip():
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append((str(request.url), body))
        if len(seen) == 1:
            return httpx.Response(200, json=completion({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "calculate", "arguments": "{\"expression\": \"6.80 + 0.50\"}"}}]},
                "tool_calls"))
        answer = {"answer": "7.30% per annum", "found_in_context": True, "confidence": "high",
                  "quote": "Senior citizens aged 60 or above receive an additional 0.50% per annum"}
        return httpx.Response(200, json=completion({"role": "assistant", "content": json.dumps(answer)}, "stop"))

    client = openai.AzureOpenAI(azure_endpoint="https://example.openai.azure.com/", api_key="test",
                                api_version="2024-10-21",
                                http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = make_azure_reader(client, "gpt-4o-mini")("FD rate for a 65-year-old, 1 year?", CONTEXT)

    assert out["answer"] == "7.30% per annum" and not out["refused"]
    url, first = seen[0]
    assert "/openai/deployments/gpt-4o-mini/chat/completions" in url and "api-version=2024-10-21" in url
    assert first["response_format"]["type"] == "json_schema"
    assert first["response_format"]["json_schema"]["strict"] is True
    assert first["tools"][0]["function"]["strict"] is True
    # second request carries the assistant tool call and our tool result
    msgs = seen[1][1]["messages"]
    assert msgs[-2]["tool_calls"][0]["id"] == "call_1"
    assert msgs[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "7.3"}
