import os
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import Field

from swiss_ai_model_launch.cli.configuration.models import (
    BranchConfiguration,
    ChainConfiguration,
    Configuration,
    OptionsConfiguration,
    PasswordConfiguration,
    TextConfiguration,
    migrate_keyring_entry,
)

_ENV_CONFIG_DIR = os.environ.get("SML_CONFIG_DIR")
_CONFIG_DIR = Path(_ENV_CONFIG_DIR) if _ENV_CONFIG_DIR else Path.home() / ".sml"
_CONFIG_FILE = _CONFIG_DIR / "config.yml"

# Configuration nodes renamed after release, old name -> new name. Configs written
# by an earlier version still carry the old names, so `load` rewrites them (and
# carries the keyring secret across) instead of failing with a KeyError.
_RENAMED_KEYS = {"cscs_api_key": "swissai_research_api_key"}


def _rename_legacy_keys(data: Any, renamed: set[str]) -> Any:
    """Return `data` with legacy node names replaced, recording which ones were hit."""
    if isinstance(data, list):
        return [_rename_legacy_keys(item, renamed) for item in data]
    if not isinstance(data, dict):
        return data
    node = {key: _rename_legacy_keys(value, renamed) for key, value in data.items()}
    name = node.get("name")
    if isinstance(name, str) and name in _RENAMED_KEYS:
        renamed.add(name)
        node["name"] = _RENAMED_KEYS[name]
    return node


class InitConfig(ChainConfiguration):
    name: str = "init_config"
    chain: list[Configuration] = Field(
        default_factory=lambda: cast(
            list[Configuration],
            [
                BranchConfiguration(
                    name="launcher_configuration",
                    head_configuration=OptionsConfiguration(
                        name="launcher",
                        prompt="How should jobs be submitted?",
                        options={
                            "firecrest": (
                                "FirecREST",
                                "5-10min setup instructions at https://docs.cscs.ch/services/devportal/#getting-started",
                            ),
                            "slurm": (
                                "SLURM Commands",
                                "Assumes you are already SSH'd into the cluster.",
                            ),
                        },
                    ),
                    branches={
                        "firecrest": ChainConfiguration(
                            name="firecrest_launcher_configuration",
                            chain=[
                                TextConfiguration(
                                    name="firecrest_url",
                                    prompt="What is your FirecREST URL?",
                                    default="https://api.cscs.ch/ml/firecrest/v2",
                                ),
                                TextConfiguration(
                                    name="firecrest_token_uri",
                                    prompt="What is your FirecREST token URI?",
                                    default="https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token",
                                ),
                                PasswordConfiguration(
                                    name="firecrest_client_id",
                                    prompt="What is your FirecREST client ID?",
                                    intro=(
                                        "\nFirecREST client ID & secret come from your CSCS Developer Portal app.\n"
                                        "Get them at: https://developer.svc.cscs.ch/devportal/apis\n"
                                        "(See https://docs.cscs.ch/services/devportal/#manage-your-applications "
                                        "for the walkthrough)\n"
                                    ),
                                    env_var="SML_FIRECREST_CLIENT_ID",
                                    expose_as_arg=False,
                                ),
                                PasswordConfiguration(
                                    name="firecrest_client_secret",
                                    prompt="What is your FirecREST client secret?",
                                    env_var="SML_FIRECREST_CLIENT_SECRET",
                                    expose_as_arg=False,
                                ),
                                TextConfiguration(
                                    name="cluster_ssh_host",
                                    prompt="(Optional) SSH host/alias for opening node terminals from the TUI",
                                    intro=(
                                        "\nUsed by the TUI's per-replica 'open' button to SSH into a node and "
                                        "attach a shell.\nLeave blank to auto-detect from the FirecREST system "
                                        "(or to disable the button).\n"
                                    ),
                                    default="",
                                    env_var="SML_CLUSTER_SSH_HOST",
                                ),
                            ],
                        ),
                        "slurm": None,
                    },
                ),
                PasswordConfiguration(
                    name="swissai_research_api_key",
                    prompt="What is your Swiss AI Research API Key?",
                    intro=(
                        "\nThe Swiss AI Research API Key is used for health checks against your served model.\n"
                        "Get one at: https://serving.swissai.svc.cscs.ch  (log in -> View API Keys)\n"
                    ),
                    env_var="SML_SWISSAI_RESEARCH_API_KEY",
                    expose_as_arg=False,
                ),
            ],
        )
    )

    @classmethod
    def exists(cls) -> bool:
        return _CONFIG_FILE.exists()

    @classmethod
    def load(cls) -> "InitConfig":
        with _CONFIG_FILE.open() as f:
            data: dict[str, Any] = yaml.safe_load(f)
        renamed: set[str] = set()
        data = _rename_legacy_keys(data, renamed)
        # Secrets are keyed by node name in the keyring, so migrate them before
        # validation reads them back.
        for old_name in renamed:
            migrate_keyring_entry(old_name, _RENAMED_KEYS[old_name])
        return cls.model_validate(data)

    def save(self) -> None:
        _CONFIG_DIR.mkdir(exist_ok=True)
        with _CONFIG_FILE.open("w") as f:
            yaml.dump(self.model_dump(mode="json"), f)
