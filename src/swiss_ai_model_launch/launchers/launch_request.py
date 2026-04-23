from typing import Literal

from pydantic import BaseModel


class ModelCatalogEntry(BaseModel):
    """A model entry from the catalogue — describes what the model needs, not how to run it."""

    vendor: str
    model_name: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_worker: int = 1
    framework_args: str | None = None
    pre_launch_cmds: str | None = None


class LaunchRequest(BaseModel):
    """A fully-specified launch request — catalogue fields plus user-supplied runtime parameters."""

    vendor: str
    model_name: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_worker: int
    workers: int
    time: str
    served_model_name: str | None = None
    framework_args: str | None = None
    pre_launch_cmds: str | None = None
    use_router: bool = False
