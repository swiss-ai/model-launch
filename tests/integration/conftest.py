import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_REQUIRED_ENV_VARS_FOR_SML_CONFIG = [
    "SML_CSCS_API_KEY",
    "SML_FIRECREST_CLIENT_ID",
    "SML_FIRECREST_CLIENT_SECRET",
    "SML_FIRECREST_TOKEN_URI",
    "SML_FIRECREST_URL",
]

# Set SML_CONFIG_DIR at conftest import time — before pytest collects any test
# file that transitively imports `swiss_ai_model_launch.cli` (which loads
# init_wizard, whose module-level `_CONFIG_DIR` snapshots this env var once).
_BOOTSTRAP_DIR: Path | None = None
if all(os.environ.get(v) for v in _REQUIRED_ENV_VARS_FOR_SML_CONFIG):
    _BOOTSTRAP_DIR = Path(tempfile.mkdtemp(prefix="sml-cfg-"))
    os.environ["SML_CONFIG_DIR"] = str(_BOOTSTRAP_DIR)


@pytest.fixture(scope="session", autouse=True)  # type: ignore[misc]
def sml_config_dir() -> Iterator[Path]:
    """Write a throwaway InitConfig into _BOOTSTRAP_DIR so `sml advanced` can run without `sml init`."""
    if _BOOTSTRAP_DIR is None:
        yield Path("/dev/null")
        return

    from swiss_ai_model_launch.cli.configuration import InitConfig

    config = InitConfig()
    config.set_value("launcher", "firecrest")
    config.set_value("firecrest_url", os.environ["SML_FIRECREST_URL"])
    config.set_value("firecrest_token_uri", os.environ["SML_FIRECREST_TOKEN_URI"])
    config.set_value("firecrest_client_id", os.environ["SML_FIRECREST_CLIENT_ID"])
    config.set_value("firecrest_client_secret", os.environ["SML_FIRECREST_CLIENT_SECRET"])
    config.set_value("cscs_api_key", os.environ["SML_CSCS_API_KEY"])
    config.set_value("telemetry_endpoint", "")
    config.save()

    yield _BOOTSTRAP_DIR
