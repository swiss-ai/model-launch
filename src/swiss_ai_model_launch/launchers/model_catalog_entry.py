from typing import Literal

from pydantic import BaseModel


class ModelCatalogEntry(BaseModel):
    model: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_replica: int = 1
    framework_args: str | None = None
    pre_launch_cmds: str | None = None
    model_path: str | None = None
