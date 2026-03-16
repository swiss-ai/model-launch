import os
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import Field

from .models import (
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
                            "remote": (
                                "Remote Launcher",
                                "If you have deployed a launcher elsewhere which is "
                                "accessible over the network.",
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
                                ),
                                TextConfiguration(
                                    name="firecrest_token_uri",
                                    prompt="What is your FirecREST token URI?",
                                ),
                                PasswordConfiguration(
                                    name="firecrest_client_id",
                                    prompt="What is your FirecREST client ID?",
                                ),
                                PasswordConfiguration(
                                    name="firecrest_client_secret",
                                    prompt="What is your FirecREST client secret?",
                                ),
                            ],
                        ),
                        "remote": ChainConfiguration(
                            name="remote_launcher_configuration",
                            chain=[
                                TextConfiguration(
                                    name="remote_launcher_address",
                                    prompt="What is your remote launcher address?",
                                ),
                                TextConfiguration(
                                    name="remote_launcher_auth_token",
                                    prompt="What is your token for authenticating in "
                                    "remote launcher?",
                                ),
                            ],
                        ),
                        "slurm": None,
                    },
                ),
                PasswordConfiguration(
                    name="cscs_api_key",
                    prompt="What is your CSCS API key? "
                    "(https://serving.swissai.cscs.ch)",
                ),
                BranchConfiguration(
                    name="telemetry_configuration",
                    head_configuration=OptionsConfiguration(
                        name="telemetry",
                        prompt="Do you want to enable telemetry to help us improve "
                        "the product? (you can change this later in the config file)",
                        options={
                            "default": (
                                "Yes",
                                "Anonymized data will be sent to telemetry endpoint.",
                            ),
                            "disabled": (
                                "No",
                                "No data will be recorded out of your usage.",
                            ),
                        },
                    ),
                    branches={
                        "default": ChainConfiguration(
                            name="telemetry_endpoint_configuration",
                            chain=[
                                TextConfiguration(
                                    name="telemetry_endpoint",
                                    prompt="What is the endpoint to which telemetry "
                                    "data should be sent?",
                                ),
                            ],
                        ),
                        "disabled": None,
                    },
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
