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
    ServedAccess,
    ServingApiError,
    served_model_access,
    whoami,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_whoami_cache():
    """whoami memoizes per API key for the life of the process, which is right
    for a CLI run and wrong for a test session: without this, the first test to
    resolve a key would decide the answer for every later one."""
    serving_api._reset_whoami_cache_for_tests()
    yield
    serving_api._reset_whoami_cache_for_tests()


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


# ── served_model_access ─────────────────────────────────────────────────────


def _access(effective, *, can_edit=True, overridden=False, conflict=False):
    """A GET /v1/model-access body, trimmed to what the client reads."""
    return {
        "model_id": "alice/org/m",
        "effective_authorization": effective,
        "is_overridden": overridden,
        "conflict": conflict,
        "can_edit": can_edit,
    }


def test_access_asks_the_model_access_endpoint_for_the_name(monkeypatch):
    calls = _patch_http(monkeypatch, _responds(200, _access("a@epfl.ch")))

    assert _run(served_model_access("sk-rc-key", "alice/org/m")) == ServedAccess(
        policy="a@epfl.ch", overridden=False, conflict=False
    )
    assert calls[0]["url"] == f"{BASE_URL}/v1/model-access/alice/org/m"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-rc-key"


def test_access_reports_a_change_made_after_launch(monkeypatch):
    """The listing only knew the launch label; the conflict rule compares
    the policy actually enforced, which an override replaces."""
    _patch_http(monkeypatch, _responds(200, _access("b@ethz.ch", overridden=True)))

    assert _run(served_model_access("sk-rc-key", "alice/org/m")) == ServedAccess(
        policy="b@ethz.ch", overridden=True, conflict=False
    )


def test_access_is_none_when_nothing_serves_the_name(monkeypatch):
    _patch_http(monkeypatch, _responds(404, {"error": {"message": "No running model"}}))

    assert _run(served_model_access("sk-rc-key", "alice/org/unlaunched")) is None


def test_access_policy_is_not_comparable_for_someone_who_cannot_edit(monkeypatch):
    """Only an editor gets addresses back; anyone else gets display names,
    and a name compared with an email list would read as a conflict."""
    _patch_http(monkeypatch, _responds(200, _access("Alice Example", can_edit=False)))

    assert _run(served_model_access("sk-rc-key", "alice/org/m")).policy is None


def test_access_public_is_comparable_for_anyone(monkeypatch):
    _patch_http(monkeypatch, _responds(200, _access("public", can_edit=False)))

    assert _run(served_model_access("sk-rc-key", "alice/org/m")).policy == "public"


def test_access_reports_launches_that_already_disagree(monkeypatch):
    _patch_http(monkeypatch, _responds(200, _access(None, conflict=True)))

    assert _run(served_model_access("sk-rc-key", "alice/org/m")) == ServedAccess(
        policy=None, overridden=False, conflict=True
    )


def test_access_rejected_key_is_reported(monkeypatch):
    _patch_http(monkeypatch, _responds(401))

    with pytest.raises(ServingApiError, match="rejected your Swiss AI Research API key"):
        _run(served_model_access("sk-rc-key", "alice/org/m"))


def test_access_http_error_is_reported(monkeypatch):
    _patch_http(monkeypatch, _responds(500))

    with pytest.raises(ServingApiError, match="HTTP 500"):
        _run(served_model_access("sk-rc-key", "alice/org/m"))


def test_access_network_failure_is_reported(monkeypatch):
    """The caller downgrades this to a warning — it must be distinguishable
    from "nothing serves this name", which is why it raises rather than
    returning None."""
    _patch_http(monkeypatch, _raises(httpx.ConnectError("no route")))

    with pytest.raises(ServingApiError, match="Could not reach the Serving API"):
        _run(served_model_access("sk-rc-key", "alice/org/m"))
