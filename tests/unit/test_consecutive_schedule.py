from datetime import datetime

import pytest

from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs, plan_consecutive_offsets
from swiss_ai_model_launch.launchers.launcher import Launcher
from swiss_ai_model_launch.launchers.topology import Topology

HOUR = 3600


def _make_args(**overrides):
    defaults = dict(
        job_name="test_job",
        served_model_name="vendor/model-abc1",
        account="proj01",
        partition="normal",
        environment="/path/to/env.toml",
        framework="sglang",
        time="12:00:00",
    )
    return LaunchArgs(**{**defaults, **overrides})


# ── plan_consecutive_offsets ────────────────────────────────────────────────


def test_single_job_when_total_fits():
    assert plan_consecutive_offsets(10 * HOUR, 12 * HOUR, HOUR) == [0]


def test_total_equal_to_cap_is_single_job():
    assert plan_consecutive_offsets(12 * HOUR, 12 * HOUR, 2 * HOUR) == [0]


def test_two_jobs_spaced_by_interval():
    # 20h total, 12h cap, 2h handover -> interval 10h, coverage (1)*10+12=22h >= 20h.
    assert plan_consecutive_offsets(20 * HOUR, 12 * HOUR, 2 * HOUR) == [0, 10 * HOUR]


def test_chain_covers_requested_total():
    # 48h total, 12h cap, 2h handover -> interval 10h.
    offsets = plan_consecutive_offsets(48 * HOUR, 12 * HOUR, 2 * HOUR)
    interval = 10 * HOUR
    # Every successive job is exactly one interval after the previous.
    assert offsets == [i * interval for i in range(len(offsets))]
    # Continuous coverage reaches the requested total.
    assert (len(offsets) - 1) * interval + 12 * HOUR >= 48 * HOUR
    # ...and one fewer job would not.
    assert (len(offsets) - 2) * interval + 12 * HOUR < 48 * HOUR


def test_zero_handover_uses_full_job_length_as_interval():
    assert plan_consecutive_offsets(24 * HOUR, 12 * HOUR, 0) == [0, 12 * HOUR]


def test_handover_at_least_job_length_is_rejected():
    with pytest.raises(ValueError):
        plan_consecutive_offsets(24 * HOUR, 12 * HOUR, 12 * HOUR)
    with pytest.raises(ValueError):
        plan_consecutive_offsets(24 * HOUR, 12 * HOUR, 13 * HOUR)


# ── launch_consecutive_with_args ────────────────────────────────────────────


class _FakeLauncher(Launcher):
    """Records the LaunchArgs of every submitted job; ids are 100, 101, ..."""

    def __init__(self):
        super().__init__(system_name="local", username="u", account="proj01", partition="normal")
        self.submitted: list[LaunchArgs] = []
        self.prepared = 0

    async def _prepare_launch_args(self, launch_args: LaunchArgs) -> LaunchArgs:
        self.prepared += 1
        return launch_args.model_copy(update={"environment": "/resolved/env.toml"})

    async def _submit_one(self, launch_args: LaunchArgs) -> int:
        self.submitted.append(launch_args)
        return 100 + len(self.submitted) - 1

    # Unused abstract members for these tests.
    async def get_preconfigured_models(self):  # pragma: no cover
        return []

    async def launch_model(self, launch_request):  # pragma: no cover
        raise NotImplementedError

    async def launch_with_args(self, launch_args):  # pragma: no cover
        raise NotImplementedError

    async def get_job_status(self, job_id):  # pragma: no cover
        return JobStatus.UNKNOWN

    async def cancel_job(self, job_id):  # pragma: no cover
        pass

    def get_tail_hint(self, job_id):  # pragma: no cover
        return ""

    async def read_job_file(self, job_id, filename):  # pragma: no cover
        return None


async def test_chain_submits_one_job_for_short_total():
    launcher = _FakeLauncher()
    base = datetime(2026, 6, 19, 8, 0, 0)
    scheduled = await launcher.launch_consecutive_with_args(
        _make_args(time="12:00:00"), total_time="10:00:00", handover_time="01:00:00", now=base
    )
    assert len(scheduled) == 1
    # Even the head job carries an explicit absolute begin (anchored to base).
    assert scheduled[0].begin == "2026-06-19T08:00:00"
    # End = begin + the 12h per-job limit.
    assert scheduled[0].end == "2026-06-19T20:00:00"
    assert launcher.submitted[0].previous_job_id is None
    assert launcher.prepared == 1  # env prepared exactly once


async def test_chain_threads_previous_job_id_and_begin_times():
    launcher = _FakeLauncher()
    base = datetime(2026, 6, 19, 8, 0, 0)
    scheduled = await launcher.launch_consecutive_with_args(
        _make_args(time="12:00:00"),
        total_time="20:00:00",
        handover_time="02:00:00",
        now=base,
    )
    assert [s.job_id for s in scheduled] == [100, 101]
    assert all(s.served_model_name == "vendor/model-abc1" for s in scheduled)

    # First job is anchored to base; second begins one 10h interval later.
    assert scheduled[0].begin == "2026-06-19T08:00:00"
    assert scheduled[1].begin == "2026-06-19T18:00:00"
    # Each end is begin + 12h; the successor begins 2h (the handover) before it.
    assert scheduled[0].end == "2026-06-19T20:00:00"
    assert scheduled[1].end == "2026-06-20T06:00:00"

    # Each job carries its predecessor's id so it can cancel it once healthy.
    assert launcher.submitted[0].previous_job_id is None
    assert launcher.submitted[1].previous_job_id == 100

    # Env is staged once, not per job.
    assert launcher.prepared == 1
    assert all(a.environment == "/resolved/env.toml" for a in launcher.submitted)


async def test_chain_per_job_time_is_unchanged():
    launcher = _FakeLauncher()
    await launcher.launch_consecutive_with_args(
        _make_args(time="12:00:00", topology=Topology(replicas=2, nodes_per_replica=1)),
        total_time="30:00:00",
        handover_time="01:00:00",
    )
    assert all(a.time == "12:00:00" for a in launcher.submitted)
