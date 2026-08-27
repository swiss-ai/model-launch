"""FirecREST client construction for both credential kinds CSCS hands out.

A personal account gets an OIDC client ID/secret pair from the Developer Portal and
authenticates with a bearer token. A service account gets an API key instead, sent
as an ``X-API-Key`` header to the PAT gateway (e.g. ``https://f7t-pat.api.svc.cscs.ch/mlp``).

pyfirecrest only implements the bearer flow, so API-key auth is layered on top with
an httpx request hook that swaps the header out on the way to the FirecREST host.
"""

import os
from collections.abc import Awaitable, Callable, Mapping

import firecrest as f7t
import httpx

API_KEY_HEADER = "X-API-Key"

# pyfirecrest always sets an Authorization header before sending; the hook drops it,
# so the value below never leaves the process.
_UNUSED_TOKEN = "unused-api-key-auth"  # noqa: S105
# FirecREST rejects a token with < 30 s left by *its* clock (401 "Access token
# expires too soon"); pyfirecrest measures the remaining life by ours. Leave a
# margin that survives a couple of minutes of clock skew between the two.
_MIN_TOKEN_VALIDITY = 180


class ApiKeyAuth:
    """Authorization object for API-key auth.

    pyfirecrest only requires an object exposing ``get_access_token()``. The token it
    returns is a placeholder — :func:`_api_key_hook` replaces the resulting bearer
    header with the API key.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_access_token(self) -> str:
        return _UNUSED_TOKEN


def _api_key_hook(api_key: str, firecrest_url: str) -> Callable[[httpx.Request], Awaitable[None]]:
    """Return an httpx request hook that swaps bearer auth for the API key."""
    firecrest_host = httpx.URL(firecrest_url).host

    # Awaits nothing, but must stay `async`: httpx's AsyncClient awaits every request
    # event hook, so a plain function raises "object NoneType can't be awaited".
    async def hook(request: httpx.Request) -> None:  # NOSONAR
        # Uploads and downloads reuse this session to talk to presigned S3 URLs,
        # which carry their own credentials in the query string. Only rewrite
        # requests that actually go to FirecREST.
        if request.url.host != firecrest_host:
            return
        request.headers.pop("Authorization", None)
        request.headers[API_KEY_HEADER] = api_key

    return hook


def _install_hook(client: f7t.v2.AsyncFirecrest, hook: Callable[[httpx.Request], Awaitable[None]]) -> None:
    client._session.event_hooks["request"].append(hook)

    # create_new_session() builds a fresh httpx client, which would come without our
    # hook, so re-install it on every new session.
    original_create_new_session = client.create_new_session

    async def create_new_session_with_hook() -> None:
        await original_create_new_session()
        client._session.event_hooks["request"].append(hook)

    client.create_new_session = create_new_session_with_hook  # type: ignore[method-assign]


def build_client(
    firecrest_url: str,
    *,
    api_key: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    token_uri: str | None = None,
) -> f7t.v2.AsyncFirecrest:
    """Build an ``AsyncFirecrest`` client.

    An API key wins when both credential kinds are present, so a service-account key
    in the environment overrides personal client credentials left over in a config.
    """
    if api_key:
        client = f7t.v2.AsyncFirecrest(
            firecrest_url=firecrest_url,
            authorization=ApiKeyAuth(api_key),
        )
        _install_hook(client, _api_key_hook(api_key, firecrest_url))
        return client

    if not (client_id and client_secret and token_uri):
        raise ValueError(
            "FirecREST authentication is not configured: provide an API key "
            "(service account), or a client ID, client secret and token URI "
            "(personal account)."
        )

    return f7t.v2.AsyncFirecrest(
        firecrest_url=firecrest_url,
        authorization=f7t.ClientCredentialsAuth(
            client_id=client_id,
            client_secret=client_secret,
            token_uri=token_uri,
            min_token_validity=_MIN_TOKEN_VALIDITY,
        ),
    )


def build_client_from_env(
    firecrest_url: str,
    env: Mapping[str, str] | None = None,
) -> f7t.v2.AsyncFirecrest:
    """Build a client from ``SML_FIRECREST_*`` credentials in the environment."""
    env = os.environ if env is None else env
    return build_client(
        firecrest_url,
        api_key=env.get("SML_FIRECREST_API_KEY"),
        client_id=env.get("SML_FIRECREST_CLIENT_ID"),
        client_secret=env.get("SML_FIRECREST_CLIENT_SECRET"),
        token_uri=env.get("SML_FIRECREST_TOKEN_URI"),
    )
