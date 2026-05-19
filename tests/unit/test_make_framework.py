import pytest

from swiss_ai_model_launch.launchers.framework import _make_framework


def test_make_framework_known():
    assert _make_framework("sglang").name == "sglang"
    assert _make_framework("vllm").name == "vllm"


def test_make_framework_unknown_raises():
    with pytest.raises(ValueError, match="Unknown framework"):
        _make_framework("nonexistent")
