import asyncio
from pathlib import Path
from typing import Any, cast

import firecrest as f7t
import httpx
import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.slurm_launcher import SlurmLauncher


def _unexpected_status(status: int) -> f7t.UnexpectedStatusException:
    """What pyfirecrest raises when FirecREST answers ``stat`` with ``status``."""
    request = httpx.Request("GET", "https://firecrest.test/filesystem/test-system/ops/stat")
    response = httpx.Response(status, json={"message": f"status {status}"}, request=request)
    return f7t.UnexpectedStatusException([response], 200)  # type: ignore[no-untyped-call]


class FakeFirecrestClient:
    """Answers ``stat`` for known paths; others fail with the given status (404 like FirecREST, by default)."""

    def __init__(self, paths: set[str], failure_status: int = 404) -> None:
        self.paths = paths
        self.failure_status = failure_status
        self.dereferenced: list[bool] = []
        self.calls = 0

    async def stat(self, system_name: str, path: str, dereference: bool = False) -> dict[str, object]:
        self.calls += 1
        self.dereferenced.append(dereference)
        if path not in self.paths:
            raise _unexpected_status(self.failure_status)
        return {"mode": 33188, "size": 896}


def _firecrest_launcher(paths: set[str], failure_status: int = 404) -> tuple[FirecRESTLauncher, FakeFirecrestClient]:
    client = FakeFirecrestClient(paths, failure_status)
    launcher = FirecRESTLauncher(
        cast(Any, client),
        "test-system",
        "test-user",
        "test-account",
        "test-partition",
    )
    return launcher, client


async def test_firecrest_path_exists_stats_the_path() -> None:
    launcher, client = _firecrest_launcher({"/models/vendor/model/config.json"})

    assert await launcher.path_exists("/models/vendor/model/config.json")
    # Model directories are frequently symlinks; the stat must follow them.
    assert client.dereferenced == [True]


# FirecREST maps stat's "No such file or directory" to 404 and "Permission
# denied" to 403; both are verdicts on the path.
@pytest.mark.parametrize("status", [404, 403])  # type: ignore[misc]
async def test_firecrest_path_exists_is_false_for_unreachable_path(status: int) -> None:
    launcher, _ = _firecrest_launcher(set(), failure_status=status)

    assert not await launcher.path_exists("/models/vendor/gone")


async def test_firecrest_path_exists_propagates_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx that outlives the retries is an error, not a missing path.

    Reporting it as "missing" is how CI flagged perfectly good checkpoints
    as stale whenever FirecREST hiccupped.
    """

    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)
    launcher, client = _firecrest_launcher(set(), failure_status=500)

    with pytest.raises(f7t.UnexpectedStatusException):
        await launcher.path_exists("/models/vendor/model/config.json")
    assert client.calls > 1  # it was retried before giving up


async def test_firecrest_path_exists_propagates_other_client_errors() -> None:
    launcher, client = _firecrest_launcher(set(), failure_status=400)

    with pytest.raises(f7t.UnexpectedStatusException):
        await launcher.path_exists("/models/vendor/model/config.json")
    assert client.calls == 1  # 4xx isn't retried


async def test_slurm_path_exists_reads_the_local_filesystem(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    launcher = SlurmLauncher("test-system", "test-user", "test-account", "test-partition")

    assert await launcher.path_exists(str(tmp_path / "config.json"))
    assert not await launcher.path_exists(str(tmp_path / "missing"))


@pytest.mark.parametrize("model_path", [None, "/elsewhere/weights"])  # type: ignore[misc]
def test_model_path_of_follows_the_registry_unless_overridden(model_path: str | None) -> None:
    launcher = SlurmLauncher(
        "test-system",
        "test-user",
        "test-account",
        "test-partition",
        model_registry=Path("/registry"),
    )
    entry = ModelCatalogEntry(model="vendor/model", framework="vllm", model_path=model_path)

    assert launcher.model_path_of(entry) == (model_path or "/registry/vendor/model")
