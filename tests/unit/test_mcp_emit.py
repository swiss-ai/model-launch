from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Client
from mcp.types import LoggingMessageNotificationParams

import swiss_ai_model_launch.mcp.server as mcp_server
from swiss_ai_model_launch.cli.healthcheck import ModelHealth
from swiss_ai_model_launch.launchers.launcher import JobStatus
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry


def _make_launcher(statuses, log_pairs):
    launcher = MagicMock()
    launcher.get_preconfigured_models = AsyncMock(
        return_value=[ModelCatalogEntry(model="test/model", framework="vllm")]
    )
    launcher.launch_model = AsyncMock(return_value=(42, "test/model-abcd"))
    launcher.get_job_status = AsyncMock(side_effect=statuses)
    launcher.get_job_logs = AsyncMock(side_effect=log_pairs)
    return launcher


async def test_emit_writes_to_stderr(capsys):
    launcher = _make_launcher(
        statuses=[JobStatus.PENDING, JobStatus.RUNNING, JobStatus.RUNNING],
        log_pairs=[
            ("loading weights\n", ""),
            ("loading weights\nweights loaded\n", "some warning\n"),
            ("loading weights\nweights loaded\n", "some warning\n"),
        ],
    )

    with (
        patch.object(mcp_server, "_launcher", launcher),
        patch.object(mcp_server, "_POLL_INTERVAL_SECONDS", 0),
        patch("swiss_ai_model_launch.mcp.server.InitConfig") as mock_cfg,
        patch("swiss_ai_model_launch.mcp.server.LaunchRequest") as mock_lr,
        patch(
            "swiss_ai_model_launch.mcp.server.check_model_health_fn",
            AsyncMock(side_effect=[ModelHealth.NOT_DEPLOYED, ModelHealth.HEALTHY]),
        ),
    ):
        mock_cfg.exists.return_value = True
        mock_cfg.load.return_value.get_value.return_value = "fake-api-key"
        mock_lr.from_catalog_entry.return_value = MagicMock()

        async with Client(mcp_server.mcp) as client:
            await client.call_tool(
                "launch_preconfigured_model",
                {"model": "test/model", "framework": "vllm"},
            )

    err = capsys.readouterr().err
    assert "[stdout] loading weights" in err
    assert "[stdout] weights loaded" in err
    assert "[stderr] some warning" in err
    assert "[status]" in err


async def test_emit_sends_mcp_notifications(capsys):
    notifications: list[str] = []

    async def log_handler(msg: LoggingMessageNotificationParams) -> None:
        notifications.append(str(msg.data))

    launcher = _make_launcher(
        statuses=[JobStatus.PENDING, JobStatus.RUNNING, JobStatus.RUNNING],
        log_pairs=[
            ("loading weights\n", ""),
            ("loading weights\nweights loaded\n", "some warning\n"),
            ("loading weights\nweights loaded\n", "some warning\n"),
        ],
    )

    with (
        patch.object(mcp_server, "_launcher", launcher),
        patch.object(mcp_server, "_POLL_INTERVAL_SECONDS", 0),
        patch("swiss_ai_model_launch.mcp.server.InitConfig") as mock_cfg,
        patch("swiss_ai_model_launch.mcp.server.LaunchRequest") as mock_lr,
        patch(
            "swiss_ai_model_launch.mcp.server.check_model_health_fn",
            AsyncMock(side_effect=[ModelHealth.NOT_DEPLOYED, ModelHealth.HEALTHY]),
        ),
    ):
        mock_cfg.exists.return_value = True
        mock_cfg.load.return_value.get_value.return_value = "fake-api-key"
        mock_lr.from_catalog_entry.return_value = MagicMock()

        async with Client(mcp_server.mcp, log_handler=log_handler) as client:
            await client.call_tool(
                "launch_preconfigured_model",
                {"model": "test/model", "framework": "vllm"},
            )

    assert any("[stdout] loading weights" in n for n in notifications)
    assert any("[stdout] weights loaded" in n for n in notifications)
    assert any("[stderr] some warning" in n for n in notifications)
    assert any("health=" in n for n in notifications)


async def test_emit_log_appears_in_return_value(capsys):
    launcher = _make_launcher(
        statuses=[JobStatus.PENDING, JobStatus.RUNNING, JobStatus.RUNNING],
        log_pairs=[
            ("loading weights\n", ""),
            ("loading weights\nweights loaded\n", "some warning\n"),
            ("loading weights\nweights loaded\n", "some warning\n"),
        ],
    )

    with (
        patch.object(mcp_server, "_launcher", launcher),
        patch.object(mcp_server, "_POLL_INTERVAL_SECONDS", 0),
        patch("swiss_ai_model_launch.mcp.server.InitConfig") as mock_cfg,
        patch("swiss_ai_model_launch.mcp.server.LaunchRequest") as mock_lr,
        patch(
            "swiss_ai_model_launch.mcp.server.check_model_health_fn",
            AsyncMock(side_effect=[ModelHealth.NOT_DEPLOYED, ModelHealth.HEALTHY]),
        ),
    ):
        mock_cfg.exists.return_value = True
        mock_cfg.load.return_value.get_value.return_value = "fake-api-key"
        mock_lr.from_catalog_entry.return_value = MagicMock()

        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool(
                "launch_preconfigured_model",
                {"model": "test/model", "framework": "vllm"},
            )

    text = result.content[0].text
    assert "is healthy" in text
    assert "Log:" in text
    assert "[stdout] weights loaded" in text
    assert "[stderr] some warning" in text
    assert "[status]" in text
