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


@pytest.fixture(scope="session", autouse=True)  # type: ignore[misc]
def sml_config_dir() -> Iterator[Path]:
    """Bootstrap a throwaway InitConfig so `sml advanced` can run without `sml init`.

    Only activates when all env vars required for a firecrest config are present —
    unit-style tests in this repo don't need it.
    """
    missing = [v for v in _REQUIRED_ENV_VARS_FOR_SML_CONFIG if not os.environ.get(v)]
    if missing:
        yield Path("/dev/null")
        return

    with tempfile.TemporaryDirectory(prefix="sml-cfg-") as tmp:
        config_dir = Path(tmp)
        os.environ["SML_CONFIG_DIR"] = str(config_dir)

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

        yield config_dir
