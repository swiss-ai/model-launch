import pytest

from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.served_name import (
    is_namespaced,
    namespace_of,
    namespace_served_model_name,
)


def test_vendor_model_gets_the_username_prepended():
    assert namespace_served_model_name("swiss-ai/Apertus-70B", "alice") == "alice/swiss-ai/Apertus-70B"


def test_bare_model_name_gets_the_username_prepended():
    assert namespace_served_model_name("Apertus-70B", "alice") == "alice/Apertus-70B"


def test_already_namespaced_under_own_username_passes_through():
    name = "alice/swiss-ai/Apertus-70B"
    assert namespace_served_model_name(name, "alice") == name


def test_namespaced_under_another_username_is_rejected():
    with pytest.raises(ValueError, match="namespaced under 'bob'"):
        namespace_served_model_name("bob/swiss-ai/Apertus-70B", "alice")


def test_username_must_be_usable_as_a_namespace():
    with pytest.raises(ValueError):
        namespace_served_model_name("swiss-ai/Apertus-70B", "")
    with pytest.raises(ValueError):
        namespace_served_model_name("swiss-ai/Apertus-70B", "a/b")
    with pytest.raises(ValueError):
        namespace_served_model_name("  ", "alice")


def test_namespace_predicates():
    assert not is_namespaced("swiss-ai/Apertus-70B")
    assert namespace_of("swiss-ai/Apertus-70B") is None
    assert is_namespaced("alice/swiss-ai/Apertus-70B")
    assert namespace_of("alice/swiss-ai/Apertus-70B") == "alice"


def _advanced_args(framework_args: str, *extra: str):
    parser = _build_parser()
    return parser.parse_args(
        [
            "advanced",
            "--partition",
            "normal",
            "--framework",
            "sglang",
            "--environment",
            "/path/to/env.toml",
            "--framework-args",
            framework_args,
            *extra,
        ]
    )


def test_advanced_namespaces_the_name_inside_framework_args():
    # The framework is what advertises the id to OpenTela, so the copy it is
    # handed has to be namespaced too — not just the LaunchArgs field the
    # labels are built from.
    args = _advanced_args("--model-path /models/Apertus --served-model-name swiss-ai/Apertus-v1.5-8B-sglang")
    la = build_launch_args_from_advanced(args, username="alice", account="proj01", partition="normal")

    assert la.served_model_name == "alice/swiss-ai/Apertus-v1.5-8B-sglang"
    assert "--served-model-name alice/swiss-ai/Apertus-v1.5-8B-sglang" in la.framework_args
    assert "--model-path /models/Apertus" in la.framework_args


def test_advanced_leaves_an_already_namespaced_name_alone():
    args = _advanced_args("--served-model-name alice/swiss-ai/Apertus-70B")
    la = build_launch_args_from_advanced(args, username="alice", account="proj01", partition="normal")

    assert la.served_model_name == "alice/swiss-ai/Apertus-70B"
    assert la.framework_args.count("--served-model-name") == 1


def test_advanced_rejects_someone_elses_namespace():
    args = _advanced_args("--served-model-name bob/swiss-ai/Apertus-70B")
    with pytest.raises(ValueError, match="namespaced under 'bob'"):
        build_launch_args_from_advanced(args, username="alice", account="proj01", partition="normal")
