import warnings
from typing import Any, Literal, Self

from pydantic import BaseModel, model_validator

from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry


class LaunchRequest(BaseModel):
    """A fully-specified launch request — catalogue fields plus user-supplied runtime parameters."""

    model: str
    framework: Literal["sglang", "vllm"]
    environment: str | None = None
    nodes_per_replica: int
    replicas: int
    time: str
    served_model_name: str | None = None
    framework_args: str | None = None
    pre_launch_cmds: str | None = None
    use_router: bool = False
    model_path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for legacy, new in (("workers", "replicas"), ("nodes_per_worker", "nodes_per_replica")):
            if legacy in data and new not in data:
                warnings.warn(
                    f"`{legacy}` is deprecated; use `{new}` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                data[new] = data.pop(legacy)
        return data

    @classmethod
    def from_catalog_entry(
        cls,
        entry: ModelCatalogEntry,
        *,
        replicas: int,
        time: str,
        served_model_name: str | None = None,
        use_router: bool = False,
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
            use_router=use_router,
            model_path=entry.model_path,
        )
