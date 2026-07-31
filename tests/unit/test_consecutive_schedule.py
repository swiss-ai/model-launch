from datetime import datetime

import pytest

from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import (
    LaunchArgs,
    plan_consecutive_offsets,
    seconds_to_time_str,
)
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


def test_seconds_to_time_str():
    assert seconds_to_time_str(8 * HOUR) == "08:00:00"
    assert seconds_to_time_str(2 * HOUR + 30 * 60) == "02:30:00"
    assert seconds_to_time_str(12 * HOUR) == "12:00:00"
    # Never emits 00:00:00 — SLURM's minimum granularity is one minute.
    assert seconds_to_time_str(0) == "00:01:00"
    assert seconds_to_time_str(30) == "00:01:00"


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
    # The sole job is also the last, so it's trimmed to the requested total (10h),
    # not the 12h cap: begin at base, end 10h later.
    assert scheduled[0].begin == "2026-06-19T08:00:00"
    assert scheduled[0].end == "2026-06-19T18:00:00"
    assert launcher.submitted[0].time == "10:00:00"
    assert launcher.submitted[0].begin == "2026-06-19T08:00:00"
    assert launcher.submitted[0].dependency is None
    assert launcher.submitted[0].previous_job_id is None
    assert launcher.prepared == 1  # env prepared exactly once


async def test_chain_head_is_anchored_successors_use_dependencies():
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

    # Head: absolute begin/end anchor, full 12h cap, no dependency.
    assert scheduled[0].begin == "2026-06-19T08:00:00"
    assert scheduled[0].end == "2026-06-19T20:00:00"
    assert scheduled[0].after is None
    assert launcher.submitted[0].time == "12:00:00"
    assert launcher.submitted[0].begin == "2026-06-19T08:00:00"
    assert launcher.submitted[0].dependency is None

    # Successor: no wall-clock time; scheduled by dependency on the head's actual
    # start, a 10h interval (12h cap − 2h handover) later, expressed in minutes.
    # It's the last job, so trimmed to land on the 20h total: starts at +10h,
    # runs 10h.
    assert scheduled[1].begin is None
    assert scheduled[1].end is None
    assert scheduled[1].after == "10h after 100 starts"
    assert launcher.submitted[1].begin is None
    assert launcher.submitted[1].dependency == "after:100+600"
    assert launcher.submitted[1].time == "10:00:00"

    # Each job carries its predecessor's id so it can cancel it once healthy.
    assert launcher.submitted[0].previous_job_id is None
    assert launcher.submitted[1].previous_job_id == 100

    # Env is staged once, not per job.
    assert launcher.prepared == 1
    assert all(a.environment == "/resolved/env.toml" for a in launcher.submitted)


async def test_chain_dependency_delay_uses_whole_minutes():
    launcher = _FakeLauncher()
    # 12h cap, 90m handover -> interval 10h30m -> 630 minutes.
    scheduled = await launcher.launch_consecutive_with_args(
        _make_args(time="12:00:00"), total_time="40:00:00", handover_time="01:30:00"
    )
    assert len(scheduled) > 2  # a few successors to check
    for i in range(1, len(scheduled)):
        prev_id = scheduled[i - 1].job_id
        assert launcher.submitted[i].dependency == f"after:{prev_id}+630"
        assert scheduled[i].after == f"10h30m after {prev_id} starts"


async def test_chain_trims_only_the_last_job_to_hit_total():
    launcher = _FakeLauncher()
    # 30h total, 12h cap, 1h handover -> interval 11h -> 3 jobs at 0/11h/22h.
    # Earlier jobs run the full cap; the last is trimmed to 30h - 22h = 8h so the
    # chain ends exactly at the requested total instead of overshooting.
    scheduled = await launcher.launch_consecutive_with_args(
        _make_args(time="12:00:00", topology=Topology(replicas=2, nodes_per_replica=1)),
        total_time="30:00:00",
        handover_time="01:00:00",
    )
    assert len(scheduled) == 3
    assert [a.time for a in launcher.submitted] == ["12:00:00", "12:00:00", "08:00:00"]


async def test_chain_last_job_trim_never_exceeds_cap():
    launcher = _FakeLauncher()
    # An awkward total that isn't a clean multiple of the interval: the trimmed
    # last job must still be a positive duration no greater than the 12h cap.
    scheduled = await launcher.launch_consecutive_with_args(
        _make_args(time="12:00:00"), total_time="13:00:00", handover_time="01:00:00"
    )
    last = launcher.submitted[-1].time
    assert last == "02:00:00"  # 13h total - 11h last-offset
    assert all(a.time == "12:00:00" for a in launcher.submitted[:-1])
    assert scheduled  # sanity
