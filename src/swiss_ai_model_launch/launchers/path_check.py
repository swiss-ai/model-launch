"""Check that the paths SML hands to a framework still hold what they claim to.

Catalog entries and example recipes name weights on shared storage, either by HF
repo id under the model registry or as an absolute path. Those directories get
moved, emptied by scratch cleanup, or never staged in the first place, and
nothing in SML notices: the launch is accepted, queues, and only fails on the
compute node minutes later. These checks ask the launcher to look at each path
directly — cheap filesystem calls, no SLURM job — so a dead path surfaces in CI
instead of in someone's launch.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from swiss_ai_model_launch.launchers.launcher import Launcher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry

# A loadable model directory carries one of these next to its weights: config.json
# for the HF layout, params.json for Mistral's native (consolidated) one. An
# emptied checkpoint directory still exists, so presence alone proves nothing.
MODEL_MARKERS = ("config.json", "params.json")

# A `--tokenizer` override points at a tokenizer directory, which has no model
# config of its own.
TOKENIZER_MARKERS = ("tokenizer.json", "tokenizer_config.json")

# Each check is one or two stat calls; a handful in flight keeps a full sweep
# quick without hammering the FirecREST gateway.
DEFAULT_CONCURRENCY = 8


class PathStatus(str, Enum):
    OK = "ok"
    # The directory is absent, or the launcher's account cannot read it — both
    # break a launch the same way, and the launcher can't tell them apart.
    MISSING = "missing"
    # The directory is there but holds none of the expected marker files: a
    # leftover, a partial download, or a checkpoint whose contents were cleaned up.
    NO_MARKER = "no-marker"


@dataclass(frozen=True)
class PathCheck:
    # What refers to this path — a catalog entry or an example script and flag.
    label: str
    path: str
    markers: tuple[str, ...]
    status: PathStatus

    @property
    def ok(self) -> bool:
        return self.status is PathStatus.OK

    def describe(self) -> str:
        """One line explaining the outcome, suitable for a CI failure message."""
        if self.status is PathStatus.OK:
            return f"{self.label}: OK — {self.path}"
        if self.status is PathStatus.MISSING:
            return f"{self.label}: path does not exist or is not readable — {self.path}"
        return f"{self.label}: directory holds none of {', '.join(self.markers)} — {self.path}"


async def check_path(
    launcher: Launcher,
    path: str,
    *,
    label: str,
    markers: tuple[str, ...] = MODEL_MARKERS,
) -> PathCheck:
    """Confirm ``path`` on the launcher's cluster holds one of ``markers``.

    A single ``stat`` of ``<path>/<marker>`` in the common case. Listing the
    directory instead would stat every weight shard, and on a few hundred of
    them under Lustre load that outlasts FirecREST's command timeout. Only when
    no marker turns up does a second ``stat`` of the directory itself tell an
    emptied checkpoint apart from a missing one.
    """
    directory = path.rstrip("/") or path
    for marker in markers:
        if await launcher.path_exists(f"{directory}/{marker}"):
            status = PathStatus.OK
            break
    else:
        status = PathStatus.NO_MARKER if await launcher.path_exists(directory) else PathStatus.MISSING
    return PathCheck(label=label, path=path, markers=markers, status=status)


async def check_model_path(launcher: Launcher, entry: ModelCatalogEntry) -> PathCheck:
    """Resolve ``entry``'s weights path on the launcher's cluster and inspect it."""
    return await check_path(
        launcher,
        launcher.model_path_of(entry),
        label=f"{entry.model} ({entry.framework})",
    )


async def check_catalog_paths(
    launcher: Launcher,
    entries: list[ModelCatalogEntry] | None = None,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[PathCheck]:
    """Check every catalog entry (or ``entries``), in catalog order."""
    entries = entries if entries is not None else await launcher.get_preconfigured_models()
    limit = asyncio.Semaphore(concurrency)

    async def check(entry: ModelCatalogEntry) -> PathCheck:
        async with limit:
            return await check_model_path(launcher, entry)

    return list(await asyncio.gather(*(check(entry) for entry in entries)))
