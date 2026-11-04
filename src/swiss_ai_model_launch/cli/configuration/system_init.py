"""``sml init --system NAME``: credentials for one more system, on its own instance.

``sml init`` configures the main FirecREST instance. A system served by another
instance (Beverin, say) needs a Developer Portal app subscribed to that
instance; this stores that app's client ID and secret in the keyring under keys
of the system, so they sit next to the main credentials instead of replacing
them. Afterwards ``--system NAME`` reaches the system through its own instance.
"""

from pathlib import Path

from swiss_ai_model_launch.cli.configuration.models import ChainConfiguration, PasswordConfiguration
from swiss_ai_model_launch.launchers.firecrest_endpoints import (
    BUILTIN_ENDPOINTS,
    DEFAULT_TOKEN_URI,
    FirecrestEndpoint,
    credential_key,
    load_endpoints,
    save_endpoint,
)


def _credentials_config(system: str) -> ChainConfiguration:
    return ChainConfiguration(
        name=f"firecrest_credentials@{system}",
        chain=[
            PasswordConfiguration(
                name=credential_key("firecrest_client_id", system),
                prompt=f"FirecREST client ID of the app subscribed to {system}'s instance?",
                env_var="SML_FIRECREST_CLIENT_ID",
                expose_as_arg=False,
            ),
            PasswordConfiguration(
                name=credential_key("firecrest_client_secret", system),
                prompt=f"FirecREST client secret for {system}?",
                env_var="SML_FIRECREST_CLIENT_SECRET",
                expose_as_arg=False,
            ),
        ],
    )


def _remember_endpoint(system: str, url: str, token_uri: str | None, path: Path | None) -> None:
    """Write the URL/token URI to clusters.yml only when they differ from what sml knows."""
    custom_url = url != BUILTIN_ENDPOINTS.get(system)
    custom_token_uri = token_uri is not None and token_uri != DEFAULT_TOKEN_URI
    if not (custom_url or custom_token_uri):
        return
    entry = load_endpoints(path).get(system) or FirecrestEndpoint()
    update: dict[str, str] = {"firecrest_url": url}
    if custom_token_uri and token_uri is not None:
        update["token_uri"] = token_uri
    save_endpoint(system, entry.model_copy(update=update), path)


async def register_system(
    system: str,
    firecrest_url: str | None = None,
    token_uri: str | None = None,
    endpoints_path: Path | None = None,
) -> str:
    """Store ``system``'s own credentials; returns the URL it will be reached through."""
    url = firecrest_url or BUILTIN_ENDPOINTS.get(system)
    if url is None:
        known = ", ".join(BUILTIN_ENDPOINTS)
        raise ValueError(f"sml does not know {system!r}'s FirecREST URL (it knows {known}): pass --firecrest-url")
    print(f"\nAdding {system}, reached through {url}.")
    print("Its app must be subscribed to that instance in https://developer.svc.cscs.ch/devportal/apis\n")
    await _credentials_config(system).aconfigure()
    _remember_endpoint(system, url, token_uri, endpoints_path)
    return url
