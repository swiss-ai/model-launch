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
)

_ENV_CONFIG_DIR = os.environ.get("SML_CONFIG_DIR")
_CONFIG_DIR = Path(_ENV_CONFIG_DIR) if _ENV_CONFIG_DIR else Path.home() / ".sml"
_CONFIG_FILE = _CONFIG_DIR / "config.yml"


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
                                "If you have FirecREST credentials of the cluster.",
                            ),
                            "slurm": (
                                "SLURM Commands",
                                "If you are running the CLI on the cluster head node "
                                "and want to directly submit jobs using SLURM.",
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
                                    env_var="SML_FIRECREST_CLIENT_ID",
                                    expose_as_arg=False,
                                ),
                                PasswordConfiguration(
                                    name="firecrest_client_secret",
                                    prompt="What is your FirecREST client secret?",
                                    env_var="SML_FIRECREST_CLIENT_SECRET",
                                    expose_as_arg=False,
                                ),
                            ],
                        ),
                        "slurm": None,
                    },
                ),
                PasswordConfiguration(
                    name="cscs_api_key",
                    prompt="What is your CSCS API key? "
                    "(https://serving.swissai.svc.cscs.ch)",
                    env_var="SML_CSCS_API_KEY",
                    expose_as_arg=False,
                ),
                TextConfiguration(
                    name="telemetry_endpoint",
                    prompt="Where to send telemetry reports?",
                    default="https://sml-dev.swissai.svc.cscs.ch/launches",
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
        return cls.model_validate(data)

    def save(self) -> None:
        _CONFIG_DIR.mkdir(exist_ok=True)
        with _CONFIG_FILE.open("w") as f:
            yaml.dump(self.model_dump(mode="json"), f)
