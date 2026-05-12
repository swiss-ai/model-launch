import pytest

from swiss_ai_model_launch.cli.main import (
    _build_parser,
    _make_cluster_loadtest_config,
    _make_loadtest_config,
)
from swiss_ai_model_launch.loadtest.setup import DEFAULT_CLUSTER_CONTAINER_IMAGE


def test_loadtest_parser_does_not_register_batch() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["loadtest", "batch"])


def test_loadtest_run_has_health_wait_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--no-wait-until-healthy"])

    assert args.wait_until_healthy is False


def test_loadtest_run_help_excludes_removed_scenario_owned_flags(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["loadtest", "run", "--help"])

    help_text = capsys.readouterr().out
    assert "--loadtest-scenario" in help_text
    assert "--loadtest-prompts-file" in help_text
    assert "--loadtest-prompt-seed" in help_text
    assert "--loadtest-max-tokens" in help_text
    assert "--loadtest-ignore-eos" in help_text
    assert "--wait-until-healthy" in help_text
    assert "--loadtest-k6-script" in help_text
    assert "--loadtest-metrics-remote-write" in help_text
    assert "--job-id" not in help_text
    assert "--loadtest-chat-mode" not in help_text
    assert "--loadtest-cpus-per-task" not in help_text
    assert "--loadtest-job-time" not in help_text
    assert "--loadtest-ready-poll-interval" not in help_text
    assert "--loadtest-think-time" not in help_text
    assert "--loadtest-request-timeout" not in help_text
    assert "--loadtest-prompt-labels" not in help_text


def test_loadtest_ignore_eos_defaults_to_scenario() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-scenario", "open_loop"])

    assert args.loadtest_ignore_eos is None
    assert _make_loadtest_config(args).ignore_eos is None


def test_loadtest_ignore_eos_can_be_forced_on() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-ignore-eos"])

    assert _make_loadtest_config(args).ignore_eos is True


def test_loadtest_ignore_eos_can_be_forced_off() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--no-loadtest-ignore-eos"])

    assert _make_loadtest_config(args).ignore_eos is False


def test_loadtest_max_tokens_can_use_prompt_values() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-max-tokens", "prompt"])

    assert _make_loadtest_config(args).max_tokens is None


def test_loadtest_max_tokens_numeric_override_wins() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-max-tokens", "123"])

    assert _make_loadtest_config(args).max_tokens == "123"


def test_loadtest_prompt_seed_defaults_to_one() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run"])

    assert _make_loadtest_config(args).prompt_seed == 1


def test_loadtest_prompt_seed_can_be_overridden() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-prompt-seed", "17"])

    assert _make_loadtest_config(args).prompt_seed == 17


def test_loadtest_run_help_uses_single_model_name(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["loadtest", "run", "--help"])

    help_text = capsys.readouterr().out
    assert "--loadtest-model" in help_text
    assert "--served-model-name" not in help_text


def test_loadtest_run_rejects_served_model_name() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["loadtest", "run", "--served-model-name", "health-name"])


def test_loadtest_parser_does_not_expose_container_image_override() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "loadtest",
                "run",
                "--loadtest-container-image",
                "/cluster/images/k6.sqsh",
            ]
        )


def test_loadtest_parser_does_not_expose_api_key_override() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["loadtest", "run", "--loadtest-api-key", "secret"])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--loadtest-chat-mode", None),
        ("--no-loadtest-chat-mode", None),
        ("--job-id", "123"),
        ("--loadtest-cpus-per-task", "8"),
        ("--loadtest-job-time", "01:00:00"),
        ("--loadtest-ready-poll-interval", "30"),
        ("--loadtest-think-time", "0"),
        ("--loadtest-request-timeout", "120s"),
        ("--loadtest-prompt-labels", "short,medium"),
    ],
)
def test_loadtest_parser_rejects_scenario_owned_overrides(flag: str, value: str | None) -> None:
    parser = _build_parser()
    args = ["loadtest", "run", flag]
    if value is not None:
        args.append(value)

    with pytest.raises(SystemExit):
        parser.parse_args(args)


def test_loadtest_uses_packaged_container_image() -> None:
    assert str(DEFAULT_CLUSTER_CONTAINER_IMAGE).endswith("/container-images/ci/k6.sqsh")


def test_loadtest_metrics_remote_write_enabled_by_default() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run"])

    assert _make_cluster_loadtest_config(args).metrics_remote_write_url == (
        "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
    )


def test_loadtest_metrics_remote_write_can_be_disabled() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--no-loadtest-metrics-remote-write"])

    assert _make_cluster_loadtest_config(args).metrics_remote_write_url is None
