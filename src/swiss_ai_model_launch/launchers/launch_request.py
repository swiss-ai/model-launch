from typing import Literal

from pydantic import BaseModel


class LaunchRequest(BaseModel):
    vendor: str
    model_name: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    workers: int
    nodes_per_worker: int
    time: str
    served_model_name: str | None = None
