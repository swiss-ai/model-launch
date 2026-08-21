"""Confirm over FirecREST that every catalog entry's weights are still on the cluster.

Nothing here submits a job: each case is a single directory listing, so the whole
catalog is checked in seconds. These run in every CI tier — a launch test only
covers the handful of models it launches, and a path that disappeared under any
of the others is exactly the failure this guards against.
"""

import importlib.resources
import json

import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.path_check import check_model_path

_MODEL_JSON = importlib.resources.files("swiss_ai_model_launch.assets").joinpath("models.json")
_CATALOG_ENTRIES = [ModelCatalogEntry.model_validate(entry) for entry in json.loads(_MODEL_JSON.read_text())]

_CATALOG_PATH_CASES = [
    pytest.param(
        entry,
        id=f"{entry.model}/{entry.framework}",
        marks=[
            pytest.mark.paths,
            pytest.mark.lightweight,
            pytest.mark.std,
            pytest.mark.comprehensive,
        ],
    )
    for entry in _CATALOG_ENTRIES
]


@pytest.mark.parametrize("entry", _CATALOG_PATH_CASES)  # type: ignore[misc]
async def test_catalog_model_path_is_valid(launcher: FirecRESTLauncher, entry: ModelCatalogEntry) -> None:
    check = await check_model_path(launcher, entry)

    if not check.ok:
        pytest.fail(check.describe(), pytrace=False)
