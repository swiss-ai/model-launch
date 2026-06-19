from swiss_ai_model_launch.cli.display.state import ChainJobView, DisplayState
from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher, _firecrest_time
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.slurm_launcher import _normalise_slurm_time, _slurm_iso_env


def test_slurm_iso_env_forces_standard_time_format(monkeypatch: object) -> None:
    # Even when the ambient env asks for relative stamps, our squeue env overrides
    # it to "standard" (ISO) so we never get "Tomorr 06:04"-style output.
    import os

    os.environ["SLURM_TIME_FORMAT"] = "relative"
    try:
        env = _slurm_iso_env()
        assert env["SLURM_TIME_FORMAT"] == "standard"
        assert "PATH" in env  # inherits the rest of the environment
    finally:
        del os.environ["SLURM_TIME_FORMAT"]


def test_normalise_slurm_time() -> None:
    assert _normalise_slurm_time("2026-06-19T08:00:00") == "2026-06-19T08:00:00"
    assert _normalise_slurm_time("  2026-06-19T08:00:00  ") == "2026-06-19T08:00:00"
    for unknown in ("", "N/A", "Unknown", "None", "(null)"):
        assert _normalise_slurm_time(unknown) is None


def test_firecrest_time_shapes() -> None:
    # SLURM's {"set", "infinite", "number"} wrapper.
    assert _firecrest_time({"set": True, "infinite": False, "number": 1750320000}) is not None
    assert _firecrest_time({"set": False, "infinite": False, "number": 0}) is None
    assert _firecrest_time({"set": True, "infinite": True, "number": 0}) is None
    # Bare epoch and the unset 0 sentinel.
    assert _firecrest_time(1750320000) is not None
    assert _firecrest_time(0) is None
    assert _firecrest_time(True) is None  # bool is an int subclass; must not format
    # Already-formatted strings pass through; placeholders don't.
    assert _firecrest_time("2026-06-19T08:00:00") == "2026-06-19T08:00:00"
    assert _firecrest_time("N/A") is None
    assert _firecrest_time(None) is None


class _FakeClient:
    def __init__(self, jobs: list[dict]) -> None:
        self._jobs = jobs

    async def job_info(self, **_kwargs: object) -> list[dict]:
        return self._jobs


def _launcher(jobs: list[dict]) -> FirecRESTLauncher:
    return FirecRESTLauncher(
        client=_FakeClient(jobs),  # type: ignore[arg-type]
        system_name="clariden",
        username="u",
        account="proj01",
        partition="normal",
    )


async def test_firecrest_get_job_times_reads_time_block() -> None:
    launcher = _launcher([{"status": {"state": "RUNNING"}, "time": {"start": 1750320000, "end": 1750356000}}])
    start, end = await launcher.get_job_times(123)
    assert start is not None and end is not None
    assert start != end


async def test_firecrest_get_job_times_missing_block_is_none() -> None:
    launcher = _launcher([{"status": {"state": "PENDING"}}])
    assert await launcher.get_job_times(123) == (None, None)


async def test_firecrest_get_job_times_empty_list_is_none() -> None:
    launcher = _launcher([])
    assert await launcher.get_job_times(123) == (None, None)


def test_set_chain_status_preserves_times_on_none() -> None:
    state = DisplayState()
    state.set_chain([ChainJobView(job_id=101)])

    # First real times arrive.
    state.set_chain_status(101, JobStatus.RUNNING, "2026-06-19T08:00:00", "2026-06-19T20:00:00")
    job = state.chain[0]
    assert (job.begin, job.end, job.status) == ("2026-06-19T08:00:00", "2026-06-19T20:00:00", JobStatus.RUNNING)

    # A later poll with unknown times must not wipe the known ones.
    state.set_chain_status(101, JobStatus.RUNNING, None, None)
    job = state.chain[0]
    assert (job.begin, job.end) == ("2026-06-19T08:00:00", "2026-06-19T20:00:00")
