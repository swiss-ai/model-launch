from pathlib import Path
from typing import Any, cast

import firecrest as f7t
import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.slurm_launcher import SlurmLauncher


class FakeFirecrestClient:
    """Serves ``ls`` from a dict; unknown paths fail the way FirecREST does."""

    def __init__(self, listings: dict[str, list[str]]) -> None:
        self.listings = listings
        self.dereferenced: list[bool] = []

    async def list_files(self, system_name: str, path: str, dereference: bool = False, **kwargs: object) -> list[dict]:
        self.dereferenced.append(dereference)
        if path not in self.listings:
            raise f7t.FirecrestException([])
        return [{"name": name, "type": "-"} for name in self.listings[path]]


def _firecrest_launcher(listings: dict[str, list[str]]) -> tuple[FirecRESTLauncher, FakeFirecrestClient]:
    client = FakeFirecrestClient(listings)
    launcher = FirecRESTLauncher(
        cast(Any, client),
        "test-system",
        "test-user",
        "test-account",
        "test-partition",
    )
    return launcher, client


async def test_firecrest_list_dir_returns_names() -> None:
    launcher, client = _firecrest_launcher({"/models/vendor/model": ["config.json", "model.safetensors"]})

    assert await launcher.list_dir("/models/vendor/model") == ["config.json", "model.safetensors"]
    # Model directories are frequently symlinks; the listing must follow them.
    assert client.dereferenced == [True]


async def test_firecrest_list_dir_returns_none_for_unreachable_path() -> None:
    launcher, _ = _firecrest_launcher({})

    assert await launcher.list_dir("/models/vendor/gone") is None


async def test_slurm_list_dir_reads_the_local_filesystem(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    launcher = SlurmLauncher("test-system", "test-user", "test-account", "test-partition")

    assert await launcher.list_dir(str(tmp_path)) == ["config.json"]
    assert await launcher.list_dir(str(tmp_path / "missing")) is None


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
