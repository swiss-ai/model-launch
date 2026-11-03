"""Container runtimes used to start each replica inside its srun step.

``edf`` is CSCS's container engine: ``srun --environment=<EDF toml>`` hands the
whole toml (image, mounts, env, annotations) to the site plugin. ``pyxis`` is
stock NVIDIA pyxis, as found on sites without the CSCS plugin (e.g. KISTI):
there is no EDF, so the same toml is read here and translated into
``--container-image`` / ``--container-mounts`` / ``--container-workdir`` flags,
with its ``[env]`` table exported on the batch node (srun propagates the job
environment into the container). ``annotations`` are CSCS hooks and are ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ContainerRuntime = Literal["edf", "pyxis"]
CONTAINER_RUNTIME_EDF: ContainerRuntime = "edf"
CONTAINER_RUNTIME_PYXIS: ContainerRuntime = "pyxis"


class PyxisEnvironment(BaseModel):
    image: str
    mounts: list[str] = Field(default_factory=list)
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str) -> PyxisEnvironment:
        try:
            data = tomllib.loads(Path(path).read_text())
        except FileNotFoundError as e:
            # The FirecREST launcher passes a remote path; pyxis needs the toml
            # readable where the script is rendered.
            raise FileNotFoundError(
                f"The pyxis container runtime reads the environment toml locally, but {path} does not exist."
            ) from e
        if "image" not in data:
            raise ValueError(f"Environment toml {path} has no `image`.")
        return cls(
            image=data["image"],
            mounts=data.get("mounts", []),
            workdir=data.get("workdir"),
            env={k: str(v) for k, v in data.get("env", {}).items()},
        )

    def container_mounts(self) -> list[str]:
        # EDF allows a bare "/path" meaning "/path:/path"; spell it out for pyxis.
        return [m if ":" in m else f"{m}:{m}" for m in self.mounts]


def shell_double_quote(value: str) -> str:
    """Quote for bash, keeping ``$VAR`` expansion (e.g. ``/scratch/$USER``)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'
