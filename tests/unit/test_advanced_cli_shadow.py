import pytest
from pydantic import ValidationError

from swiss_ai_model_launch.cli.main import (
    _build_parser,
    build_launch_args_from_advanced,
)
from swiss_ai_model_launch.launchers.framework import render_rank_scripts

_BASE_ID = "SwissAI-Research/swiss-ai/Apertus-8B-Instruct-2509"


def _minimal_advanced_args(*extra: str):
    parser = _build_parser()
    tokens = [
        "advanced",
        "--system",
        "clariden",
        "--partition",
        "normal",
        "--framework",
        "sglang",
        "--environment",
        "/path/to/env.toml",
        "--framework-args",
        "--served-model-name vendor/model-abc",
        *extra,
    ]
    return parser.parse_args(tokens)


def test_advanced_default_is_not_a_shadow():
    args = _minimal_advanced_args()
    la = build_launch_args_from_advanced(args, username="alice", account="proj01", partition="normal")
    assert la.shadow_of is None


def test_advanced_shadow_of_is_wired_through():
    args = _minimal_advanced_args("--shadow-of", _BASE_ID)
    la = build_launch_args_from_advanced(args, username="alice", account="proj01", partition="normal")
    assert la.shadow_of == _BASE_ID
    # The shadow keeps its own namespaced served name; only its mesh service differs.
    assert la.served_model_name == "alice/vendor/model-abc"
    assert "    --service.name llm-shadow \\\n" in render_rank_scripts(la)["head.sh"]


def test_advanced_shadow_of_rejects_malformed_id():
    args = _minimal_advanced_args("--shadow-of", "swiss-ai/Apertus-8B")
    with pytest.raises(ValidationError, match="shadow_of must be a model id"):
        build_launch_args_from_advanced(args, username="alice", account="proj01", partition="normal")


def test_advanced_shadow_of_rejects_disabled_opentela():
    args = _minimal_advanced_args("--shadow-of", _BASE_ID, "--disable-opentela")
    with pytest.raises(ValidationError, match="cannot be combined with disable_opentela"):
        build_launch_args_from_advanced(args, username="alice", account="proj01", partition="normal")
