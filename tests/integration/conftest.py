import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

_REQUIRED_ENV_VARS_FOR_SML_CONFIG = [
    "SML_SWISSAI_RESEARCH_API_KEY",
    "SML_FIRECREST_URL",
]

# Credentials come in two shapes: a service-account API key, which the CLI reads
# straight from SML_FIRECREST_API_KEY and which therefore needs nothing written
# into the config, or a personal account's client ID/secret plus token URI, which
# does. Only one set needs to be present.
_CLIENT_CREDENTIALS_ENV_VARS = [
    "SML_FIRECREST_CLIENT_ID",
    "SML_FIRECREST_CLIENT_SECRET",
    "SML_FIRECREST_TOKEN_URI",
]
_HAS_API_KEY = bool(os.environ.get("SML_FIRECREST_API_KEY"))
_HAS_CLIENT_CREDENTIALS = all(os.environ.get(v) for v in _CLIENT_CREDENTIALS_ENV_VARS)


# Set SML_CONFIG_DIR at conftest import time — before pytest collects any test
# file that transitively imports `swiss_ai_model_launch.cli` (which loads
# init_wizard, whose module-level `_CONFIG_DIR` snapshots this env var once).
_BOOTSTRAP_DIR: Path | None = None
if (_HAS_API_KEY or _HAS_CLIENT_CREDENTIALS) and all(os.environ.get(v) for v in _REQUIRED_ENV_VARS_FOR_SML_CONFIG):
    _BOOTSTRAP_DIR = Path(tempfile.mkdtemp(prefix="sml-cfg-"))
    os.environ["SML_CONFIG_DIR"] = str(_BOOTSTRAP_DIR)

# Imported after the bootstrap above: `tests.integration.utils` pulls in the CLI
# package, whose init wizard snapshots SML_CONFIG_DIR at import time.
from swiss_ai_model_launch.cli.configuration import InitConfig  # noqa: E402
from swiss_ai_model_launch.launchers.firecrest_auth import build_client_from_env  # noqa: E402
from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher  # noqa: E402
from tests.integration.utils import firecrest_auth_env  # noqa: E402


@pytest.fixture(scope="session", autouse=True)  # type: ignore[misc]
def sml_config_dir() -> Iterator[Path]:
    """Write a throwaway InitConfig into _BOOTSTRAP_DIR so `sml advanced` can run without `sml init`."""
    if _BOOTSTRAP_DIR is None:
        yield Path("/dev/null")
        return

    config = InitConfig()
    config.set_value("launcher", "firecrest")
    config.set_value("firecrest_url", os.environ["SML_FIRECREST_URL"])
    if _HAS_CLIENT_CREDENTIALS:
        for env_var in _CLIENT_CREDENTIALS_ENV_VARS:
            config.set_value(env_var.removeprefix("SML_").lower(), os.environ[env_var])
    config.set_value("swissai_research_api_key", os.environ["SML_SWISSAI_RESEARCH_API_KEY"])
    config.save()

    yield _BOOTSTRAP_DIR


# Every FirecREST-backed test needs these; the launcher fixture below turns them
# into a ready client, so a test only asks for `launcher`.
_REQUIRED_ENV_VARS = [
    "SML_SWISSAI_RESEARCH_API_KEY",
    "SML_SYSTEM",
    "SML_FIRECREST_URL",
    "SML_PARTITION",
    "SML_RESERVATION",
]


@pytest.fixture(scope="function")  # type: ignore[misc]
def env() -> dict[str, str]:
    missing = [v for v in _REQUIRED_ENV_VARS if os.environ.get(v) is None]
    if missing:
        pytest.fail(
            "Missing required environment variables: " + ", ".join(missing),
            pytrace=False,
        )
    return {v: os.environ[v] for v in _REQUIRED_ENV_VARS} | firecrest_auth_env()


@pytest.fixture(scope="function")  # type: ignore[misc]
async def launcher(env: dict[str, str]) -> AsyncIterator[FirecRESTLauncher]:
    client = build_client_from_env(env["SML_FIRECREST_URL"], env)
    try:
        yield await FirecRESTLauncher.from_client(
            client=client,
            system_name=env["SML_SYSTEM"],
            partition=env["SML_PARTITION"],
            reservation=env["SML_RESERVATION"] or None,
        )
    finally:
        await client.close_session()


@pytest.fixture(scope="function")  # type: ignore[misc]
def swissai_research_api_key(env: dict[str, str]) -> str:
    return env["SML_SWISSAI_RESEARCH_API_KEY"]
