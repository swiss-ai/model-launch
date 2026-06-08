import pytest

from swiss_ai_model_launch.cli.loadtest import (
    _prompt_loadtest_scenario,
    make_cluster_loadtest_config,
    make_loadtest_config,
    make_loadtest_config_from_values,
)
from swiss_ai_model_launch.cli.main import _build_parser
from swiss_ai_model_launch.loadtest.setup import DEFAULT_CLUSTER_CONTAINER_IMAGE


def test_loadtest_parser_does_not_register_batch() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["loadtest", "batch"])


def test_loadtest_run_has_health_wait_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--no-wait-until-healthy"])

    assert args.wait_until_healthy is False


def test_loadtest_scenario_parser_default_is_unset() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run"])

    assert args.loadtest_scenario is None


def test_loadtest_scenario_config_default_is_throughput() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run"])

    assert make_loadtest_config(args).scenario == "throughput"


def test_loadtest_config_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="Unknown loadtest scenario 'missing'"):
        make_loadtest_config_from_values(
            scenario="missing",
            max_tokens=None,
            ignore_eos=None,
        )


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
    assert "--loadtest-metrics-remote-write" in help_text
    assert "--loadtest-job-time" in help_text
    assert "--job-id" not in help_text
    assert "--loadtest-chat-mode" not in help_text
    assert "--loadtest-cpus-per-task" not in help_text
    assert "--loadtest-k6-script" not in help_text
    assert "--loadtest-ready-poll-interval" not in help_text
    assert "--loadtest-think-time" not in help_text
    assert "--loadtest-request-timeout" not in help_text
    assert "--loadtest-prompt-labels" not in help_text


def test_loadtest_ignore_eos_defaults_to_scenario() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-scenario", "open_loop"])

    assert args.loadtest_ignore_eos is None
    assert make_loadtest_config(args).ignore_eos is None


def test_loadtest_ignore_eos_can_be_forced_on() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-ignore-eos"])

    assert make_loadtest_config(args).ignore_eos is True


def test_loadtest_ignore_eos_can_be_forced_off() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--no-loadtest-ignore-eos"])

    assert make_loadtest_config(args).ignore_eos is False


def test_loadtest_max_tokens_can_use_prompt_values() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-max-tokens", "prompt"])

    assert make_loadtest_config(args).max_tokens is None


def test_loadtest_max_tokens_numeric_override_wins() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-max-tokens", "123"])

    assert make_loadtest_config(args).max_tokens == "123"


def test_loadtest_prompt_seed_defaults_to_one() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run"])

    assert make_loadtest_config(args).prompt_seed == 1


def test_loadtest_prompt_seed_can_be_overridden() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-prompt-seed", "17"])

    assert make_loadtest_config(args).prompt_seed == 17


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
        ("--loadtest-k6-script", "script.js"),
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

    assert make_cluster_loadtest_config(args).metrics_remote_write_url == (
        "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
    )


def test_loadtest_job_time_defaults_to_cluster_default() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run"])

    assert make_cluster_loadtest_config(args).time == "02:00:00"


def test_loadtest_job_time_can_be_overridden() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-job-time", "01:00:00"])

    assert make_cluster_loadtest_config(args).time == "01:00:00"


def test_loadtest_job_time_rejects_invalid_format() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-job-time", "1h"])

    with pytest.raises(ValueError, match="--loadtest-job-time"):
        make_cluster_loadtest_config(args)


def test_loadtest_ready_timeout_rejects_non_positive_value() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-ready-timeout", "0"])

    with pytest.raises(ValueError, match="--loadtest-ready-timeout"):
        make_cluster_loadtest_config(args)


def test_loadtest_cancel_requires_wait() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "loadtest",
            "advanced",
            "--serving-framework",
            "sglang",
            "--slurm-environment",
            "/path/to/env.toml",
            "--cancel-after-loadtest",
            "--no-wait-for-loadtest",
        ]
    )

    with pytest.raises(ValueError, match="--cancel-after-loadtest"):
        make_cluster_loadtest_config(args)


def test_loadtest_metrics_remote_write_can_be_disabled() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--no-loadtest-metrics-remote-write"])

    assert make_cluster_loadtest_config(args).metrics_remote_write_url is None


async def test_prompt_loadtest_scenario_uses_existing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "preconfigured", "--loadtest-scenario", "open_loop"])

    def fail_select(*args: object, **kwargs: object) -> None:
        raise AssertionError("scenario prompt should not run")

    monkeypatch.setattr("swiss_ai_model_launch.cli.loadtest.questionary.select", fail_select)

    await _prompt_loadtest_scenario(args)

    assert args.loadtest_scenario == "open_loop"


async def test_prompt_loadtest_scenario_sets_selected_value(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "preconfigured"])

    class FakeQuestion:
        async def ask_async(self) -> str:
            return "open_loop"

    def fake_select(*args: object, **kwargs: object) -> FakeQuestion:
        return FakeQuestion()

    monkeypatch.setattr("swiss_ai_model_launch.cli.loadtest.questionary.select", fake_select)

    await _prompt_loadtest_scenario(args)

    assert args.loadtest_scenario == "open_loop"
