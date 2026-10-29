import asyncio
import importlib

import pytest

from swiss_ai_model_launch.cli.display.live import _MODEL_HEALTH_STYLE
from swiss_ai_model_launch.cli.healthcheck import ModelHealth
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launcher import ScheduledJob

# The package re-exports a `main` function that shadows the module of that name.
main = importlib.import_module("swiss_ai_model_launch.cli.main")
_REAL_SLEEP = asyncio.sleep


class _FakeLauncher:
    system_name = "clariden"
    partition = "normal"
    terminal_command = None

    async def get_job_status(self, job_id):
        return JobStatus.RUNNING

    async def get_job_times(self, job_id):
        return None, None

    async def get_replica_health(self, job_id, served, expected_replicas):
        return None

    async def read_job_file(self, job_id, filename):
        return ""

    async def cancel_job(self, job_id):  # pragma: no cover - the fake display never kills
        raise AssertionError("unexpected cancel")


async def _run_monitor_briefly(monkeypatch, *, shadow: bool) -> tuple[list[str], list]:
    probes: list[str] = []
    states: list = []

    async def fake_probe(model_name, api_key):
        probes.append(model_name)
        return ModelHealth.HEALTHY

    class FakeDisplay:
        def __init__(self, state):
            states.append(state)

        async def run(self, coro):
            # Drive the monitor for a moment instead of a Textual app, then stop.
            try:
                await asyncio.wait_for(coro, timeout=0.2)
            except TimeoutError:
                pass
            return False

    async def fast_sleep(_seconds):
        await _REAL_SLEEP(0.001)

    async def submit():
        return [ScheduledJob(job_id=1, served_model_name="alice/vendor/model", begin=None)]

    monkeypatch.setattr(main, "check_model_health", fake_probe)
    monkeypatch.setattr(main, "LiveDisplay", FakeDisplay)
    monkeypatch.setattr(main.asyncio, "sleep", fast_sleep)
    await main._run_monitor(_FakeLauncher(), submit(), "key", expected_replicas=1, shadow=shadow)
    return probes, states


async def test_monitor_probes_gateway_for_regular_launch(monkeypatch):
    # Control: proves the harness would notice a gateway probe.
    probes, states = await _run_monitor_briefly(monkeypatch, shadow=False)
    assert probes
    assert states[0].model_health == ModelHealth.HEALTHY


async def test_monitor_never_probes_gateway_for_shadow(monkeypatch):
    # The gateway never routes to a shadow; probing would hit the baseline when
    # it shares the served name.
    probes, states = await _run_monitor_briefly(monkeypatch, shadow=True)
    assert probes == []
    assert states[0].model_health == ModelHealth.SHADOW


@pytest.mark.parametrize("health", list(ModelHealth))
def test_every_model_health_has_a_style(health):
    assert _MODEL_HEALTH_STYLE[health]
