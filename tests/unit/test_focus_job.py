from swiss_ai_model_launch.cli.main import _focus_job
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launcher import ScheduledJob


def _chain(n: int) -> list[ScheduledJob]:
    return [ScheduledJob(job_id=100 + i, served_model_name="m", begin=None) for i in range(n)]


def test_focus_newest_running_after_handover() -> None:
    jobs = _chain(3)
    statuses = {100: JobStatus.RUNNING, 101: JobStatus.RUNNING, 102: JobStatus.PENDING}
    # Two overlapping during handover: follow the newest live one.
    assert _focus_job(jobs, statuses).job_id == 101


def test_focus_next_pending_when_nothing_running() -> None:
    jobs = _chain(3)
    statuses = {100: JobStatus.UNKNOWN, 101: JobStatus.PENDING, 102: JobStatus.PENDING}
    assert _focus_job(jobs, statuses).job_id == 101


def test_focus_last_when_chain_finished() -> None:
    jobs = _chain(3)
    statuses = {100: JobStatus.UNKNOWN, 101: JobStatus.UNKNOWN, 102: JobStatus.UNKNOWN}
    assert _focus_job(jobs, statuses).job_id == 102


def test_focus_single_job() -> None:
    jobs = _chain(1)
    assert _focus_job(jobs, {100: JobStatus.RUNNING}).job_id == 100
