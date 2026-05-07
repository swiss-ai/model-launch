import warnings
from typing import Any, Literal

from pydantic import BaseModel, model_validator


class ModelCatalogEntry(BaseModel):
    """A model entry from the catalogue — describes what the model needs, not how to run it."""

    model: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_replica: int = 1
    framework_args: str | None = None
    pre_launch_cmds: str | None = None
    model_path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: Any) -> Any:
        if isinstance(data, dict) and "nodes_per_worker" in data and "nodes_per_replica" not in data:
            warnings.warn(
                "`nodes_per_worker` is deprecated; use `nodes_per_replica` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["nodes_per_replica"] = data.pop("nodes_per_worker")
        return data
