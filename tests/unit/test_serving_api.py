"""The Serving API client used at launch time: identity lookup for
`--authorization private`, and the pre-launch served-name policy check.

Both calls decide who can reach a model, so every failure mode has to be a
loud ServingApiError — never a silent empty answer that a caller might read
as "no restriction".
"""

import asyncio

import httpx
import pytest

from swiss_ai_model_launch import serving_api
from swiss_ai_model_launch.serving_api import (
    BASE_URL,
    ServingApiError,
    served_model_authorizations,
    whoami,
)


def _run(coro):
    return asyncio.run(coro)


class _FakeClient:
    """Stands in for httpx.AsyncClient: records the request and replays a
    caller-supplied response (or raises the given exception)."""

    def __init__(self, handler, calls):
        self._handler = handler
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, headers=None, timeout=None):
        self._calls.append({"url": url, "headers": headers or {}, "timeout": timeout})
        return self._handler()


def _patch_http(monkeypatch, handler):
    """Returns the list the fake client appends each request to."""
    calls: list[dict] = []
    monkeypatch.setattr(
        serving_api.httpx,
        "AsyncClient",
        lambda *a, **k: _FakeClient(handler, calls),
    )
    return calls


def _responds(status, payload=None):
    def handler():
        return httpx.Response(status, json=payload if payload is not None else {})

    return handler


def _raises(exc):
    def handler():
        raise exc

    return handler


# ── whoami ──────────────────────────────────────────────────────────────────


def test_whoami_returns_the_email_and_sends_the_bearer(monkeypatch):
    calls = _patch_http(monkeypatch, _responds(200, {"email": "alice@epfl.ch"}))

    assert _run(whoami("sk-rc-key")) == "alice@epfl.ch"
    assert calls[0]["url"] == f"{BASE_URL}/v1/whoami"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-rc-key"


def test_whoami_401_names_the_key_as_the_problem(monkeypatch):
    """The most likely cause is an unconfigured or rotated key, so say so
    rather than surfacing a bare status code."""
    _patch_http(monkeypatch, _responds(401))

    with pytest.raises(ServingApiError, match="rejected your Swiss AI Research API key"):
        _run(whoami("sk-rc-stale"))


def test_whoami_other_http_error_is_reported(monkeypatch):
    _patch_http(monkeypatch, _responds(503))

    with pytest.raises(ServingApiError, match="HTTP 503"):
        _run(whoami("sk-rc-key"))


def test_whoami_network_failure_is_reported(monkeypatch):
    _patch_http(monkeypatch, _raises(httpx.ConnectError("no route")))

    with pytest.raises(ServingApiError, match="Could not reach the Serving API"):
        _run(whoami("sk-rc-key"))


def test_whoami_timeout_is_reported(monkeypatch):
    _patch_http(monkeypatch, _raises(httpx.ReadTimeout("slow")))

    with pytest.raises(ServingApiError, match="Could not reach the Serving API"):
        _run(whoami("sk-rc-key"))


def test_whoami_without_an_email_is_an_error_not_an_empty_string(monkeypatch):
    """An empty email would normalize into an invalid label and quietly
    restrict the model to nobody."""
    _patch_http(monkeypatch, _responds(200, {}))

    with pytest.raises(ServingApiError, match="no email address"):
        _run(whoami("sk-rc-key"))


# ── served_model_authorizations ─────────────────────────────────────────────


_MODELS = {
    "data": [
        {"id": "alice/org/m", "labels": {"authorization": "a@epfl.ch"}},
        {"id": "alice/org/m", "labels": {"authorization": "a@epfl.ch"}},
        {"id": "alice/org/other", "labels": {"authorization": "public"}},
        {"id": "bob/org/m", "labels": {"authorization": "b@ethz.ch"}},
    ]
}


def test_authorizations_returns_only_the_matching_names_labels(monkeypatch):
    calls = _patch_http(monkeypatch, _responds(200, _MODELS))

    assert _run(served_model_authorizations("sk-rc-key", "alice/org/m")) == [
        "a@epfl.ch",
        "a@epfl.ch",
    ]
    assert calls[0]["url"] == f"{BASE_URL}/v1/models_detailed"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-rc-key"


def test_authorizations_is_empty_when_the_name_is_not_served(monkeypatch):
    _patch_http(monkeypatch, _responds(200, _MODELS))

    assert _run(served_model_authorizations("sk-rc-key", "alice/org/unlaunched")) == []


def test_authorizations_reads_a_missing_label_as_empty(monkeypatch):
    """A peer with no authorization label is public; the caller compares
    normalized policies, so "" has to survive the trip."""
    _patch_http(monkeypatch, _responds(200, {"data": [{"id": "alice/org/m"}]}))

    assert _run(served_model_authorizations("sk-rc-key", "alice/org/m")) == [""]


def test_authorizations_tolerates_an_empty_payload(monkeypatch):
    _patch_http(monkeypatch, _responds(200, {}))

    assert _run(served_model_authorizations("sk-rc-key", "alice/org/m")) == []


def test_authorizations_http_error_is_reported(monkeypatch):
    _patch_http(monkeypatch, _responds(500))

    with pytest.raises(ServingApiError, match="HTTP 500"):
        _run(served_model_authorizations("sk-rc-key", "alice/org/m"))


def test_authorizations_network_failure_is_reported(monkeypatch):
    """The caller downgrades this to a warning — it must be distinguishable
    from "no conflicting peers", which is why it raises rather than returns []."""
    _patch_http(monkeypatch, _raises(httpx.ConnectError("no route")))

    with pytest.raises(ServingApiError, match="Could not reach the Serving API"):
        _run(served_model_authorizations("sk-rc-key", "alice/org/m"))
