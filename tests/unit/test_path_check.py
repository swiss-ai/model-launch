from pathlib import Path

from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import Launcher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.path_check import (
    PathStatus,
    check_catalog_paths,
    check_model_path,
    check_path,
)

_REGISTRY = Path("/registry")


class FakeLauncher(Launcher):
    """A launcher whose filesystem is a dict of path -> directory listing."""

    def __init__(self, listings: dict[str, list[str]], entries: list[ModelCatalogEntry] | None = None) -> None:
        super().__init__(
            system_name="test-system",
            username="test-user",
            account="test-account",
            partition="test-partition",
            model_registry=_REGISTRY,
        )
        self.listings = listings
        self.entries = entries or []
        self.listed: list[str] = []

    async def list_dir(self, path: str) -> list[str] | None:
        self.listed.append(path)
        return self.listings.get(path)

    async def get_preconfigured_models(self) -> list[ModelCatalogEntry]:
        return self.entries

    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]:
        raise NotImplementedError

    async def launch_with_args(self, launch_args: LaunchArgs) -> tuple[int, str]:
        raise NotImplementedError

    async def get_job_status(self, job_id: int) -> JobStatus:
        return JobStatus.UNKNOWN

    async def cancel_job(self, job_id: int) -> None:
        return None

    def get_tail_hint(self, job_id: int) -> str:
        return ""

    async def read_job_file(self, job_id: int, filename: str) -> str | None:
        return None


def _entry(model: str, model_path: str | None = None) -> ModelCatalogEntry:
    return ModelCatalogEntry(model=model, framework="vllm", model_path=model_path)


async def test_registry_model_with_config_is_ok() -> None:
    launcher = FakeLauncher({"/registry/vendor/model": ["config.json", "model.safetensors"]})

    check = await check_model_path(launcher, _entry("vendor/model"))

    assert check.status is PathStatus.OK
    assert check.ok
    assert launcher.listed == ["/registry/vendor/model"]


async def test_absent_directory_is_missing() -> None:
    check = await check_model_path(FakeLauncher({}), _entry("vendor/model"))

    assert check.status is PathStatus.MISSING
    assert not check.ok
    assert "/registry/vendor/model" in check.describe()


async def test_directory_without_a_marker_is_flagged() -> None:
    # An emptied checkpoint directory still exists — the listing is what tells
    # it apart from a model the framework can actually load.
    launcher = FakeLauncher({"/scratch/checkpoint": []})

    check = await check_model_path(launcher, _entry("vendor/model", model_path="/scratch/checkpoint"))

    assert check.status is PathStatus.NO_MARKER
    assert "config.json" in check.describe()


async def test_mistral_native_layout_counts_as_a_model() -> None:
    # Mistral ships consolidated weights with params.json and no config.json.
    launcher = FakeLauncher({"/registry/vendor/model": ["params.json", "consolidated.safetensors"]})

    check = await check_model_path(launcher, _entry("vendor/model"))

    assert check.status is PathStatus.OK


async def test_check_path_takes_its_own_markers() -> None:
    # Tokenizer directories carry no model config, so the caller says what proves
    # the directory is the right kind of thing.
    launcher = FakeLauncher({"/store/tokenizers/vendor": ["tokenizer.json"]})

    tokenizer = await check_path(
        launcher, "/store/tokenizers/vendor", label="recipe.sh (--tokenizer)", markers=("tokenizer.json",)
    )
    as_model = await check_path(launcher, "/store/tokenizers/vendor", label="recipe.sh (--model)")

    assert tokenizer.status is PathStatus.OK
    assert tokenizer.describe().startswith("recipe.sh (--tokenizer): OK")
    assert as_model.status is PathStatus.NO_MARKER


async def test_model_path_override_bypasses_the_registry() -> None:
    launcher = FakeLauncher({"/elsewhere/weights": ["config.json"]})

    check = await check_model_path(launcher, _entry("vendor/model", model_path="/elsewhere/weights"))

    assert check.status is PathStatus.OK
    assert check.path == "/elsewhere/weights"


async def test_check_catalog_paths_covers_every_entry_in_order() -> None:
    entries = [_entry("vendor/good"), _entry("vendor/gone"), _entry("vendor/empty", model_path="/scratch/empty")]
    launcher = FakeLauncher(
        {"/registry/vendor/good": ["config.json"], "/scratch/empty": ["README.md"]},
        entries=entries,
    )

    checks = await check_catalog_paths(launcher)

    assert [c.label for c in checks] == [
        "vendor/good (vllm)",
        "vendor/gone (vllm)",
        "vendor/empty (vllm)",
    ]
    assert [c.status for c in checks] == [
        PathStatus.OK,
        PathStatus.MISSING,
        PathStatus.NO_MARKER,
    ]
