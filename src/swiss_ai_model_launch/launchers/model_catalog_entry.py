from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelCatalogEntry(BaseModel):
    # Unknown keys are rejected: a misspelt field (e.g. "nodes_per_worker") would
    # otherwise be dropped silently and the entry would launch with the default.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_replica: int = 1
    framework_args: str | None = None
    pre_launch_cmds: str | None = None
    model_path: str | None = None
    # Marks the entry as part of the lightweight CI matrix; underscore-prefixed in
    # models.json to keep it visually apart from the launch fields.
    include_in_lightweight_ci: bool = Field(default=False, alias="_include_in_lightweight_ci")
