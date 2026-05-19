from pathlib import Path

from swiss_ai_model_launch.launchers.utils import resolve_model_path


def test_resolve_model_path_uses_registry():
    registry = Path("/store/models")
    assert resolve_model_path("vendor/name", registry) == "/store/models/vendor/name"


def test_resolve_model_path_explicit_path_takes_precedence():
    registry = Path("/store/models")
    assert resolve_model_path("vendor/name", registry, "/custom/path") == "/custom/path"
