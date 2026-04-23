from typing import Literal, Self

from pydantic import BaseModel

from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry


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

    @classmethod
    def from_catalog_entry(
        cls,
        entry: ModelCatalogEntry,
        *,
        workers: int,
        time: str,
        served_model_name: str | None = None,
        use_router: bool = False,
    ) -> Self:
        return cls(
            vendor=entry.vendor,
            model_name=entry.model_name,
            framework=entry.framework,
            environment=entry.environment,
            nodes_per_worker=entry.nodes_per_worker,
            framework_args=entry.framework_args,
            pre_launch_cmds=entry.pre_launch_cmds,
            workers=workers,
            time=time,
            served_model_name=served_model_name,
            use_router=use_router,
        )
