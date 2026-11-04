"""Per-system FirecREST endpoints, so one sml setup can reach several instances.

CSCS serves systems through separate FirecREST instances, each with its own URL
and its own Developer Portal subscription: Clariden on the ML platform, Beverin,
which is AMD, on an instance of its own. ``sml init`` configures exactly one.

A system gets its own endpoint when it has its own credentials:

- ``sml init --system beverin`` stores them in the keyring under keys of that
  system (``firecrest_client_id@beverin``), next to, not over, the init ones;
- or ``clusters.yml`` (next to the init config) names environment variables::

      systems:
        beverin:
          client_id_env: BEVERIN_CLIENT_ID
          client_secret_env: BEVERIN_CLIENT_SECRET

The URL of a known system is built in (``BUILTIN_ENDPOINTS``); ``clusters.yml``
can set ``firecrest_url`` for any other. A system without credentials of its own
goes through the ``sml init`` configuration, exactly as before.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import firecrest as f7t
import keyring
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from swiss_ai_model_launch.launchers.firecrest_auth import build_client

ENDPOINTS_FILE_NAME = "clusters.yml"
DEFAULT_TOKEN_URI = "https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token"  # noqa: S105
# Same service as the init config's secrets (cli/configuration/models.py).
KEYRING_SERVICE = "swiss_ai_model_launch"

_HPC = "https://api.cscs.ch/hpc/firecrest/v2"
_ML = "https://api.cscs.ch/ml/firecrest/v2"
BUILTIN_ENDPOINTS: Mapping[str, str] = {
    "daint": _HPC,
    "eiger": _HPC,
    "clariden": _ML,
    "bristen": _ML,
    "beverin": "https://api.svc.cscs.ch/beverin/firecrest/v2",
}


class EndpointConfigError(ValueError):
    """``clusters.yml`` is malformed, or a listed system cannot be reached with it."""


class FirecrestEndpoint(BaseModel):
    """A ``clusters.yml`` entry: where a system's instance and credentials are."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    firecrest_url: str | None = None
    token_uri: str = DEFAULT_TOKEN_URI
    client_id_env: str | None = None
    client_secret_env: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class SystemCredentials:
    client_id: str
    client_secret: str


CredentialLookup = Callable[[str], SystemCredentials | None]


def credential_key(name: str, system: str) -> str:
    """Keyring key of a system's own secret, e.g. ``firecrest_client_id@beverin``."""
    return f"{name}@{system}"


def stored_credentials(system: str) -> SystemCredentials | None:
    """The credentials ``sml init --system SYSTEM`` stored, if any."""
    client_id = keyring.get_password(KEYRING_SERVICE, credential_key("firecrest_client_id", system))
    client_secret = keyring.get_password(KEYRING_SERVICE, credential_key("firecrest_client_secret", system))
    return SystemCredentials(client_id, client_secret) if client_id and client_secret else None


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
        return {str(name): FirecrestEndpoint.model_validate(entry or {}) for name, entry in systems.items()}
    except ValidationError as exc:
        raise EndpointConfigError(f"{path}: {exc}") from exc


def save_endpoint(system: str, endpoint: FirecrestEndpoint, path: Path | None = None) -> None:
    """Add or replace ``system``'s entry in ``clusters.yml``, keeping the others."""
    path = path or default_endpoints_path()
    entries = {name: e.model_dump(exclude_defaults=True) for name, e in load_endpoints(path).items()}
    entries[system] = endpoint.model_dump(exclude_defaults=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"systems": entries}, sort_keys=False))


def _url_of(system: str, listed: FirecrestEndpoint | None) -> str | None:
    return (listed.firecrest_url if listed else None) or BUILTIN_ENDPOINTS.get(system)


def _env_credentials(listed: FirecrestEndpoint, env: Mapping[str, str]) -> tuple[str | None, SystemCredentials | None]:
    """``(api_key, client credentials)`` from the variables an entry names."""
    api_key = env.get(listed.api_key_env) if listed.api_key_env else None
    client_id = env.get(listed.client_id_env) if listed.client_id_env else None
    client_secret = env.get(listed.client_secret_env) if listed.client_secret_env else None
    pair = SystemCredentials(client_id, client_secret) if client_id and client_secret else None
    return api_key, pair


def _missing_credentials(system: str, listed: FirecrestEndpoint) -> EndpointConfigError:
    names = [n for n in (listed.api_key_env, listed.client_id_env, listed.client_secret_env) if n]
    how = f"export {' / '.join(names)}, or " if names else ""
    return EndpointConfigError(
        f"system {system!r} is listed in {ENDPOINTS_FILE_NAME} but has no credentials: "
        f"{how}run `sml init --system {system}`"
    )


def client_for_system(
    system: str | None,
    default_client: Callable[[], f7t.v2.AsyncFirecrest],
    env: Mapping[str, str] | None = None,
    endpoints: Mapping[str, FirecrestEndpoint] | None = None,
    lookup: CredentialLookup = stored_credentials,
) -> f7t.v2.AsyncFirecrest:
    """The client for ``system``: its own endpoint if it has credentials, else ``default_client()``."""
    if system is None:
        return default_client()
    endpoints = load_endpoints() if endpoints is None else endpoints
    listed = endpoints.get(system)
    api_key, credentials = _env_credentials(listed, os.environ if env is None else env) if listed else (None, None)
    credentials = credentials or lookup(system)

    if not api_key and credentials is None:
        if listed is not None:
            raise _missing_credentials(system, listed)
        return default_client()  # no credentials of its own: the init config, as before

    url = _url_of(system, listed)
    if url is None:
        raise EndpointConfigError(
            f"system {system!r} has credentials but no FirecREST URL: set firecrest_url for it in "
            f"{ENDPOINTS_FILE_NAME}, or pass --firecrest-url to `sml init --system {system}`"
        )
    return build_client(
        url,
        api_key=api_key,
        client_id=credentials.client_id if credentials else None,
        client_secret=credentials.client_secret if credentials else None,
        token_uri=listed.token_uri if listed else DEFAULT_TOKEN_URI,
    )
