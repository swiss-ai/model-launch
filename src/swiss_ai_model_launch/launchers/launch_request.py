from typing import Literal, Self

from pydantic import BaseModel

from swiss_ai_model_launch.launchers.authorization import PRIVATE
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
    # Optional SLURM job name. Set it to something deterministic (an id of
    # your own) and the launch becomes idempotent: if FirecREST reports an
    # error after the sbatch actually went through, the launcher finds the
    # job by this name and returns it instead of submitting a second one.
    # Unset, a unique random name is generated.
    job_name: str | None = None
    # Access policy for the served model: "public", a comma-separated email
    # list, or "private" (the default) — only the launcher, resolved to their
    # email by the CLI, or by the Launcher before it submits.
    authorization: str = PRIVATE
    # The launcher's platform identity (from /v1/whoami), stamped on the mesh
    # so the gateway knows whose model this is when someone asks to change its
    # access later. Empty is allowed and means admin-managed.
    launched_by_email: str = ""

    @classmethod
    def from_catalog_entry(
        cls,
        entry: ModelCatalogEntry,
        *,
        replicas: int,
        time: str,
        served_model_name: str | None = None,
        router: RouterMode = ROUTER_OPENTELA,
        authorization: str = PRIVATE,
        launched_by_email: str = "",
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
            authorization=authorization,
            launched_by_email=launched_by_email,
        )
