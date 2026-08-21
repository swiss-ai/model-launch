"""Static checks on the shipped catalog — everything provable without a cluster.

Whether the weights are still on shared storage needs FirecREST and lives in
tests/integration/test_model_catalog_paths.py.
"""

import importlib.resources
import json
from collections import Counter
from pathlib import Path

from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CATALOG = json.loads(importlib.resources.files("swiss_ai_model_launch.assets").joinpath("models.json").read_text())


def test_every_entry_parses() -> None:
    # ModelCatalogEntry forbids unknown keys, so this also catches a misspelt
    # field that would otherwise be dropped and silently defaulted.
    entries = [ModelCatalogEntry.model_validate(entry) for entry in _CATALOG]

    assert entries


def test_model_framework_pairs_are_unique() -> None:
    # A model may appear once per framework; two entries for the same pair would
    # make the picker ambiguous and duplicate the CI matrix.
    pairs = Counter((entry["model"], entry["framework"]) for entry in _CATALOG)
    duplicates = [pair for pair, count in pairs.items() if count > 1]

    assert not duplicates, f"duplicate catalog entries: {duplicates}"


def test_referenced_environment_files_exist() -> None:
    # `environment` is a repo-relative path to an env toml that gets uploaded at
    # launch time; a stale one fails only once someone tries to launch.
    missing = [
        entry["environment"]
        for entry in _CATALOG
        if entry.get("environment") and not (_REPO_ROOT / entry["environment"]).is_file()
    ]

    assert not missing, f"catalog references env files that don't exist: {missing}"


def test_model_paths_are_absolute() -> None:
    # A `model_path` override is passed straight to the framework on a compute
    # node, where the job's working directory is not the user's.
    relative = [entry["model_path"] for entry in _CATALOG if entry.get("model_path") and entry["model_path"][0] != "/"]

    assert not relative, f"catalog model_path overrides must be absolute: {relative}"
