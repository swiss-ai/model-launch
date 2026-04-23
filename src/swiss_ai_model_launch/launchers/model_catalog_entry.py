from typing import Any, Literal

from pydantic import BaseModel, model_validator


class ModelCatalogEntry(BaseModel):
    """A model entry from the catalogue — describes what the model needs, not how to run it."""

    vendor: str
    model_name: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_worker: int = 1
    framework_args: str | None = None
    pre_launch_cmds: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_model_field(cls, data: Any) -> Any:
        if isinstance(data, dict) and "model" in data:
            vendor, model_name = data.pop("model").split("/", 1)
            data.setdefault("vendor", vendor)
            data.setdefault("model_name", model_name)
        return data
