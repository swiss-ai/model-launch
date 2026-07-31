from typing import Literal, Self

from pydantic import BaseModel

from swiss_ai_model_launch.launchers.launch_args import ROUTER_OPENTELA, RouterMode
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry


class LaunchRequest(BaseModel):
    model: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_replica: int
    replicas: int
    time: str
    served_model_name: str | None = None
    framework_args: str | None = None
    pre_launch_cmds: str | None = None
    router: RouterMode = ROUTER_OPENTELA
    model_path: str | None = None

    @classmethod
    def from_catalog_entry(
        cls,
        entry: ModelCatalogEntry,
        *,
        replicas: int,
        time: str,
        served_model_name: str | None = None,
        router: RouterMode = ROUTER_OPENTELA,
    ) -> Self:
        return cls(
            model=entry.model,
            framework=entry.framework,
            environment=entry.environment,
            nodes_per_replica=entry.nodes_per_replica,
            framework_args=entry.framework_args,
            pre_launch_cmds=entry.pre_launch_cmds,
            replicas=replicas,
            time=time,
            served_model_name=served_model_name,
            router=router,
            model_path=entry.model_path,
        )
