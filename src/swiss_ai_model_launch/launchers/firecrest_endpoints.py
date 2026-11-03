"""Per-system FirecREST endpoints, so one sml setup can reach several instances.

CSCS serves systems through separate FirecREST instances, each with its own URL
and its own Developer Portal subscription: Clariden on the ML platform
(``api.cscs.ch/ml/firecrest/v2``), Beverin, which is AMD, on its own
(``api.svc.cscs.ch/beverin/firecrest/v2``). ``sml init`` configures exactly one
of them. ``clusters.yml`` next to the init config lists the others::

    systems:
      beverin:
        firecrest_url: https://api.svc.cscs.ch/beverin/firecrest/v2
        client_id_env: BEVERIN_CLIENT_ID
        client_secret_env: BEVERIN_CLIENT_SECRET

A system listed there is reached through its own endpoint and credentials;
every other system goes through the ``sml init`` configuration as before.
Credentials are read from the environment variables the entry names, never
stored in the file or the keyring.
"""

import os
from collections.abc import Callable, Mapping
from pathlib import Path

import firecrest as f7t
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from swiss_ai_model_launch.launchers.firecrest_auth import build_client

ENDPOINTS_FILE_NAME = "clusters.yml"
DEFAULT_TOKEN_URI = "https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token"  # noqa: S105


class EndpointConfigError(ValueError):
    """``clusters.yml`` is malformed, or an entry's credentials are not set."""


class FirecrestEndpoint(BaseModel):
    """How to reach one system: its instance URL and where its credentials are."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    firecrest_url: str
    token_uri: str = DEFAULT_TOKEN_URI
    client_id_env: str | None = None
    client_secret_env: str | None = None
    api_key_env: str | None = None


def default_endpoints_path() -> Path:
    config_dir = os.environ.get("SML_CONFIG_DIR")
    return (Path(config_dir) if config_dir else Path.home() / ".sml") / ENDPOINTS_FILE_NAME


def load_endpoints(path: Path | None = None) -> dict[str, FirecrestEndpoint]:
    """The systems listed in ``clusters.yml``; none if the file does not exist."""
    path = path or default_endpoints_path()
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    systems = data.get("systems") if isinstance(data, dict) else None
    if not isinstance(systems, dict):
        raise EndpointConfigError(f"{path}: expected a top-level 'systems:' mapping")
    try:
        return {str(name): FirecrestEndpoint.model_validate(entry) for name, entry in systems.items()}
    except ValidationError as exc:
        raise EndpointConfigError(f"{path}: {exc}") from exc


def build_endpoint_client(system: str, endpoint: FirecrestEndpoint, env: Mapping[str, str]) -> f7t.v2.AsyncFirecrest:
    """A client for ``endpoint``, with credentials from the variables it names."""
    api_key = env.get(endpoint.api_key_env) if endpoint.api_key_env else None
    client_id = env.get(endpoint.client_id_env) if endpoint.client_id_env else None
    client_secret = env.get(endpoint.client_secret_env) if endpoint.client_secret_env else None
    if not api_key and not (client_id and client_secret):
        names = [n for n in (endpoint.api_key_env, endpoint.client_id_env, endpoint.client_secret_env) if n]
        raise EndpointConfigError(
            f"system {system!r} is configured in {ENDPOINTS_FILE_NAME} but its credentials are not set: "
            f"export {' or '.join(names) if names else 'client_id_env/client_secret_env in its entry'}"
        )
    return build_client(
        endpoint.firecrest_url,
        api_key=api_key,
        client_id=client_id,
        client_secret=client_secret,
        token_uri=endpoint.token_uri,
    )


def client_for_system(
    system: str | None,
    default_client: Callable[[], f7t.v2.AsyncFirecrest],
    env: Mapping[str, str] | None = None,
    endpoints: Mapping[str, FirecrestEndpoint] | None = None,
) -> f7t.v2.AsyncFirecrest:
    """The client for ``system``: its own endpoint if listed, else ``default_client()``."""
    endpoints = load_endpoints() if endpoints is None else endpoints
    if system is not None and system in endpoints:
        return build_endpoint_client(system, endpoints[system], os.environ if env is None else env)
    return default_client()
