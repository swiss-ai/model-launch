import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import _primary_group_name


def test_primary_group_name_pre_2_6_0_shape() -> None:
    user_info = {
        "user": {"id": "1000", "name": "fireuser"},
        "group": {"id": "100", "name": "oldgroup"},
        "groups": [{"id": "100", "name": "oldgroup"}],
    }
    assert _primary_group_name(user_info) == "oldgroup"


def test_primary_group_name_2_6_0_shape_uses_default_flag() -> None:
    user_info = {
        "user": {"id": "1000", "name": "fireuser"},
        "groups": [
            {"id": "100", "name": "users", "default": False},
            {"id": "200", "name": "newgroup", "default": True},
        ],
    }
    assert _primary_group_name(user_info) == "newgroup"


def test_primary_group_name_2_6_0_shape_falls_back_to_first_group() -> None:
    user_info = {
        "user": {"id": "1000", "name": "fireuser"},
        "groups": [{"id": "300", "name": "fallbackgroup"}],
    }
    assert _primary_group_name(user_info) == "fallbackgroup"


def test_primary_group_name_raises_without_group_info() -> None:
    user_info = {"user": {"id": "1000", "name": "fireuser"}, "groups": []}
    with pytest.raises(RuntimeError, match="no group information"):
        _primary_group_name(user_info)
