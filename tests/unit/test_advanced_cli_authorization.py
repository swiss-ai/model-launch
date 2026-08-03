import pytest

from swiss_ai_model_launch.cli.main import (
    _build_parser,
    build_launch_args_from_advanced,
)


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


def test_advanced_authorization_defaults_to_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SML_AUTHORIZATION", raising=False)
    args = _minimal_advanced_args()
    assert args.authorization == "private"


def test_advanced_authorization_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_AUTHORIZATION", "public")
    args = _minimal_advanced_args()
    assert args.authorization == "public"


def test_advanced_authorization_flag_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_AUTHORIZATION", "public")
    args = _minimal_advanced_args("--authorization", "user@epfl.ch")
    assert args.authorization == "user@epfl.ch"


def test_build_launch_args_default_emits_no_label() -> None:
    args = _minimal_advanced_args()
    la = build_launch_args_from_advanced(args, account="proj01", partition="normal")
    assert la.authorization == ""


def test_build_launch_args_threads_resolved_authorization() -> None:
    # The caller resolves first (private -> the launcher's email); this only
    # checks the already-resolved value is threaded through to LaunchArgs.
    args = _minimal_advanced_args()
    la = build_launch_args_from_advanced(
        args,
        account="proj01",
        partition="normal",
        authorization="user@epfl.ch",
    )
    assert la.authorization == "user@epfl.ch"


def test_preconfigured_authorization_is_parsed() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "preconfigured",
            "--system",
            "clariden",
            "--partition",
            "normal",
            "--model",
            "vendor/model-abc",
            "--framework",
            "sglang",
            "--replicas",
            "1",
            "--router",
            "opentela",
            "--time",
            "02:00:00",
            "--authorization",
            "public",
        ]
    )
    assert args.authorization == "public"


def test_loadtest_advanced_authorization_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SML_AUTHORIZATION", raising=False)
    parser = _build_parser()
    args = parser.parse_args(
        [
            "loadtest",
            "advanced",
            "--framework",
            "sglang",
            "--environment",
            "/path/to/env.toml",
        ]
    )
    assert args.authorization == "private"
