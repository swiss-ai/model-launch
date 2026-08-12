import firecrest as f7t
import httpx
import pytest

from swiss_ai_model_launch.launchers.firecrest_auth import (
    API_KEY_HEADER,
    ApiKeyAuth,
    build_client,
    build_client_from_env,
)

_URL = "https://f7t-pat.api.svc.cscs.ch/mlp"
_KEY = "svc-account-key"


def _record_headers(seen: list[httpx.Headers]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={})

    return httpx.MockTransport(handler)


async def _send(client: f7t.v2.AsyncFirecrest, url: str) -> httpx.Headers:
    """Send one request through the client's session, returning the headers as sent."""
    seen: list[httpx.Headers] = []
    # Swap the transport rather than the session: the API-key hook lives on the
    # session's event hooks, and the point is to exercise it on the real send path.
    client._session._transport = _record_headers(seen)
    await client._session.get(url, headers={"Authorization": "Bearer some-token"})
    return seen[0]


async def test_api_key_replaces_bearer_auth_on_firecrest_requests() -> None:
    client = build_client(_URL, api_key=_KEY)
    headers = await _send(client, f"{_URL}/status/systems")
    assert headers[API_KEY_HEADER] == _KEY
    assert "authorization" not in headers


async def test_api_key_is_not_sent_to_presigned_storage_urls() -> None:
    client = build_client(_URL, api_key=_KEY)
    headers = await _send(client, "https://object-store.cscs.ch/bucket/part?X-Amz-Signature=abc")
    assert API_KEY_HEADER not in headers
    assert headers["authorization"] == "Bearer some-token"


async def test_api_key_hook_survives_a_new_session() -> None:
    client = build_client(_URL, api_key=_KEY)
    await client.create_new_session()
    headers = await _send(client, f"{_URL}/status/systems")
    assert headers[API_KEY_HEADER] == _KEY


def test_api_key_wins_over_leftover_client_credentials() -> None:
    client = build_client(
        _URL,
        api_key=_KEY,
        client_id="id",
        client_secret="secret",  # noqa: S106
        token_uri="https://auth.cscs.ch/token",  # noqa: S106
    )
    assert isinstance(client._authorization, ApiKeyAuth)


def test_client_credentials_are_still_used_when_no_api_key_is_set() -> None:
    client = build_client(
        _URL,
        client_id="id",
        client_secret="secret",  # noqa: S106
        token_uri="https://auth.cscs.ch/token",  # noqa: S106
    )
    assert isinstance(client._authorization, f7t.ClientCredentialsAuth)


def test_incomplete_credentials_are_rejected() -> None:
    with pytest.raises(ValueError, match="not configured"):
        build_client(_URL, client_id="id")


def test_build_client_from_env_reads_either_credential_kind() -> None:
    api_key_client = build_client_from_env(_URL, {"SML_FIRECREST_API_KEY": _KEY})
    assert isinstance(api_key_client._authorization, ApiKeyAuth)

    client_credentials_client = build_client_from_env(
        _URL,
        {
            "SML_FIRECREST_CLIENT_ID": "id",
            "SML_FIRECREST_CLIENT_SECRET": "secret",
            "SML_FIRECREST_TOKEN_URI": "https://auth.cscs.ch/token",
        },
    )
    assert isinstance(client_credentials_client._authorization, f7t.ClientCredentialsAuth)
