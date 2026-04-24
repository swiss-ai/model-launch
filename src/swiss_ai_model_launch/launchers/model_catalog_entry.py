from typing import Literal

from pydantic import BaseModel


class ModelCatalogEntry(BaseModel):
    """A model entry from the catalogue — describes what the model needs, not how to run it."""

    model: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_worker: int = 1
    framework_args: str | None = None
    pre_launch_cmds: str | None = None
    model_path: str | None = None
