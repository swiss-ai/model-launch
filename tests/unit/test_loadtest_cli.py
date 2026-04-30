import pytest

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


def test_loadtest_uses_packaged_container_image() -> None:
    assert str(DEFAULT_CLUSTER_CONTAINER_IMAGE).endswith("/container-images/ci/k6.sqsh")
