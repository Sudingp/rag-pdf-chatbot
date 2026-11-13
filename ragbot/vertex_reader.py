"""
Gemini on Google Cloud Vertex AI as a reader ("vertex").

Vertex AI serves Gemini through an OpenAI-compatible endpoint, so the same reader logic as the
other hosted readers (structured JSON, the calculator tool, evidence-quote guardrail, fallbacks)
is reused unchanged. What differs is authentication: there is no API key. The reader uses
Application Default Credentials and sends a short-lived OAuth access token, refreshed before
it expires (tokens last about an hour, a Cloud Run instance lives longer).

    Local:      gcloud auth application-default login
    Cloud Run:  the service's runtime service account needs the "Vertex AI User" role
                (roles/aiplatform.user); nothing secret is stored anywhere.

Settings (environment variables, or a local .env):
    VERTEX_PROJECT    Google Cloud project ID (falls back to the credentials' project)
    VERTEX_LOCATION   default "global"
    VERTEX_MODEL      default "google/gemini-3.5-flash"
"""

import os
from types import SimpleNamespace
from typing import Callable, Optional

from ragbot.azure_reader import _load_env, make_azure_reader

DEFAULT_MODEL = "google/gemini-3.5-flash"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def vertex_base_url(project: str, location: str = "global") -> str:
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


class VertexClient:
    """Looks like an OpenAI client (chat.completions.create) and puts a fresh access token
    on every call, so a long-running server never sends an expired one."""

    def __init__(self, credentials, base_url: str, openai_client=None, request_factory=None):
        if openai_client is None:
            from openai import OpenAI
            openai_client = OpenAI(base_url=base_url, api_key="set-per-call")
        if request_factory is None:
            from google.auth.transport.requests import Request
            request_factory = Request
        self._creds, self._client, self._request = credentials, openai_client, request_factory
        self.refreshes = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _token(self) -> str:
        if not self._creds.valid or not self._creds.token:
            self._creds.refresh(self._request())
            self.refreshes += 1
        return self._creds.token

    def _create(self, **kwargs):
        self._client.api_key = self._token()
        return self._client.chat.completions.create(**kwargs)


def load_vertex_reader(model: Optional[str] = None) -> Callable[[str, str], dict]:
    _load_env()
    try:
        import google.auth
    except ImportError as e:
        raise RuntimeError("The 'vertex' reader needs google-auth: pip install google-auth") from e
    try:
        credentials, default_project = google.auth.default(scopes=SCOPES)
    except Exception as e:  # DefaultCredentialsError: no login on this machine
        raise RuntimeError("No Google Cloud credentials found. Run "
                           "'gcloud auth application-default login' (locally) or give the Cloud Run "
                           "service account the Vertex AI User role.") from e
    project = os.environ.get("VERTEX_PROJECT") or default_project
    if not project:
        raise RuntimeError("Set VERTEX_PROJECT to your Google Cloud project ID.")
    location = os.environ.get("VERTEX_LOCATION", "global")
    client = VertexClient(credentials, vertex_base_url(project, location))
    return make_azure_reader(client, model or os.environ.get("VERTEX_MODEL", DEFAULT_MODEL))
