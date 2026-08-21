"""Confirm over FirecREST that the clariden examples still point at real weights.

Nothing here submits a job: each case is a single directory listing for one
distinct path, so all 80-odd recipes are covered in seconds. The comprehensive
tier does launch every example, but only that tier — and a recipe whose
checkpoint was cleaned off scratch is broken for everyone long before anyone
labels a PR to find out.
"""

import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.path_check import check_path
from tests.example_paths import ReferenceGroup, discover_references, group_references

_REFERENCE_GROUPS = group_references(discover_references())

_EXAMPLE_PATH_CASES = [
    pytest.param(
        group,
        id=group.value,
        marks=[
            pytest.mark.paths,
            pytest.mark.lightweight,
            pytest.mark.std,
            pytest.mark.comprehensive,
        ],
    )
    for group in _REFERENCE_GROUPS
    # A path built from a shell substitution CI can't expand isn't checkable;
    # tests/unit/test_example_paths.py is what keeps that set empty.
    if not group.reference.unresolved
]


@pytest.mark.parametrize("group", _EXAMPLE_PATH_CASES)  # type: ignore[misc]
async def test_example_path_is_valid(launcher: FirecRESTLauncher, group: ReferenceGroup) -> None:
    check = await check_path(
        launcher,
        group.resolve(launcher.model_registry),
        label=group.label,
        markers=group.markers,
    )

    if not check.ok:
        pytest.fail(check.describe(), pytrace=False)
