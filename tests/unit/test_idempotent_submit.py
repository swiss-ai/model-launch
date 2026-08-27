"""Job submission is idempotent: a named launch is never duplicated.

FirecREST can report an error (5xx, or 408 when its own SSH command to the
cluster times out) for an sbatch that actually went through. A retry that
doesn't check first allocates a second job. These tests drive
FirecRESTLauncher with a fake client that behaves like FirecREST on a bad day.
"""

import asyncio

import httpx
import pytest
from firecrest import UnexpectedStatusException

from swiss_ai_model_launch.launchers import FirecRESTLauncher, JobStatus, LaunchRequest
from swiss_ai_model_launch.launchers.utils import firecrest_status, is_firecrest_retryable


def _status_error(code: int) -> UnexpectedStatusException:
    resp = httpx.Response(code, request=httpx.Request("POST", "https://api.cscs.ch/jobs"))
    return UnexpectedStatusException([resp], 201)


class FakeClient:
    """FirecREST with a script: submit() answers each call from `submit_outcomes`
    (an int job id, or an exception to raise); job_info() lists whatever
    `jobs` holds -- the launcher matches names itself, since FirecREST's own
    name filter needs API >= 2.6 and CSCS runs 2.5."""

    def __init__(self, submit_outcomes, jobs=None):
        self.submit_outcomes = list(submit_outcomes)
        self.jobs = jobs or []
        self.submits = 0
        self.lookups = []

    async def submit(self, **kwargs):
        self.submits += 1
        outcome = self.submit_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"jobId": outcome}

    async def job_info(self, system_name, jobid=None, name=None, **kwargs):
        self.lookups.append(name)
        return [j for j in self.jobs if name is None or j["name"] == name]

    async def mkdir(self, **kwargs):
        return None

    async def upload(self, **kwargs):
        return None


def _launcher(client) -> FirecRESTLauncher:
    return FirecRESTLauncher(
        client=client, system_name="clariden", username="alice", account="infra01", partition="normal"
    )


def _request(job_name=None) -> LaunchRequest:
    return LaunchRequest(
        model="swiss-ai/Apertus-8B-Instruct-2509",
        framework="sglang",
        nodes_per_replica=1,
        replicas=1,
        time="01:00:00",
        job_name=job_name,
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


def test_408_and_429_count_as_transient() -> None:
    assert is_firecrest_retryable(_status_error(408))
    assert is_firecrest_retryable(_status_error(429))
    assert is_firecrest_retryable(_status_error(503))
    assert not is_firecrest_retryable(_status_error(400))
    assert not is_firecrest_retryable(ValueError("nope"))


def test_firecrest_status_reads_the_last_response() -> None:
    assert firecrest_status(_status_error(404)) == 404
    # Not a FirecREST status error at all, or one that carries no response.
    assert firecrest_status(httpx.ConnectError("down")) is None
    assert firecrest_status(UnexpectedStatusException([], 200)) is None


def test_named_launch_adopts_the_job_a_failed_submit_created() -> None:
    # FirecREST 408s, but the sbatch went through: the job exists under our name.
    client = FakeClient(
        submit_outcomes=[_status_error(408), 999],
        jobs=[{"jobId": 4242, "name": "evalsvc-abc123", "status": {"state": "PENDING"}}],
    )
    launcher = _launcher(client)

    job_id, served = asyncio.run(launcher.launch_model(_request(job_name="evalsvc-abc123")))

    assert job_id == 4242
    assert served == "alice/swiss-ai/Apertus-8B-Instruct-2509"
    assert client.submits == 1  # no second sbatch
    assert len(client.lookups) == 1


def test_named_launch_retries_when_no_job_exists_yet() -> None:
    client = FakeClient(submit_outcomes=[_status_error(503), _status_error(408), 777], jobs=[])
    launcher = _launcher(client)

    job_id, _ = asyncio.run(launcher.launch_model(_request(job_name="evalsvc-def456")))

    assert job_id == 777
    assert client.submits == 3
    assert len(client.lookups) == 2  # one per failed submit


def test_waits_before_looking_the_job_up(monkeypatch) -> None:
    """SLURM needs a moment to list a job FirecREST reported an error for."""
    events: list[str] = []

    async def recording_sleep(seconds):
        events.append(f"sleep {seconds:g}")

    monkeypatch.setattr(asyncio, "sleep", recording_sleep)
    client = FakeClient(
        submit_outcomes=[_status_error(408)],
        jobs=[{"jobId": 7, "name": "evalsvc-slow", "status": {"state": "PENDING"}}],
    )
    original = client.job_info

    async def recording_lookup(**kwargs):
        events.append("lookup")
        return await original(**kwargs)

    client.job_info = recording_lookup
    job_id, _ = asyncio.run(_launcher(client).launch_model(_request(job_name="evalsvc-slow")))
    assert job_id == 7
    assert events == ["sleep 10", "lookup"]


def test_the_job_is_picked_out_of_the_full_list() -> None:
    client = FakeClient(
        submit_outcomes=[_status_error(503)],
        jobs=[
            {"jobId": 1, "name": "someone-else", "status": {"state": "RUNNING"}},
            {"jobId": 2, "name": "evalsvc-mine", "status": {"state": "PENDING"}},
        ],
    )
    job_id, _ = asyncio.run(_launcher(client).launch_model(_request(job_name="evalsvc-mine")))
    assert job_id == 2
    assert client.lookups == [None]  # no server-side name filter (API 2.5)


def test_finished_job_with_our_name_is_not_adopted() -> None:
    client = FakeClient(
        submit_outcomes=[_status_error(503), 555],
        jobs=[{"jobId": 1, "name": "evalsvc-old", "status": {"state": "CANCELLED by 0"}}],
    )
    job_id, _ = asyncio.run(_launcher(client).launch_model(_request(job_name="evalsvc-old")))
    assert job_id == 555
    assert client.submits == 2


def test_non_transient_error_is_not_retried() -> None:
    client = FakeClient(submit_outcomes=[_status_error(400), 1])
    with pytest.raises(UnexpectedStatusException):
        asyncio.run(_launcher(client).launch_model(_request(job_name="evalsvc-bad")))
    assert client.submits == 1


def test_unnamed_launch_gets_a_unique_name_and_still_checks_by_it() -> None:
    client = FakeClient(submit_outcomes=[_status_error(503), 31337], jobs=[])
    launcher = _launcher(client)
    job_id, _ = asyncio.run(launcher.launch_model(_request()))
    assert job_id == 31337
    assert len(client.lookups) == 1


def test_find_job_reports_live_jobs_only() -> None:
    client = FakeClient(
        submit_outcomes=[],
        jobs=[
            {"jobId": 1, "name": "x", "status": {"state": "COMPLETED"}},
            {"jobId": 2, "name": "x", "status": {"state": "RUNNING"}},
        ],
    )
    assert asyncio.run(_launcher(client).find_job("x")) == (2, JobStatus.RUNNING)
    assert asyncio.run(_launcher(client).find_job("y")) is None


def test_lookup_failing_too_falls_back_to_a_plain_retry() -> None:
    """If FirecREST is down for the lookup as well, adoption is impossible; the
    launcher must still retry the submission rather than give up or crash."""
    client = FakeClient(submit_outcomes=[_status_error(503), 8], jobs=[])

    async def broken_lookup(**kwargs):
        raise _status_error(503)

    client.job_info = broken_lookup
    job_id, _ = asyncio.run(_launcher(client).launch_model(_request(job_name="evalsvc-x")))
    assert job_id == 8
    assert client.submits == 2


def test_exhausted_attempts_raise_the_last_error() -> None:
    client = FakeClient(submit_outcomes=[_status_error(503)] * 6, jobs=[])
    with pytest.raises(UnexpectedStatusException):
        asyncio.run(_launcher(client).launch_model(_request(job_name="evalsvc-never")))
    assert client.submits == 6  # 1 + 5 retries


def test_slurm_find_job_parses_squeue(monkeypatch) -> None:
    from swiss_ai_model_launch.launchers import SlurmLauncher

    calls = []

    class FakeProc:
        def __init__(self, out: bytes):
            self._out = out

        async def communicate(self):
            return self._out, b""

    outputs = {
        "live": b"123 PENDING\n124 RUNNING\n",
        "done": b"125 COMPLETED\n",
        "none": b"",
        "junk": b"not squeue output\n",
    }

    async def fake_exec(*argv, **kwargs):
        calls.append(argv)
        name = argv[argv.index("--name") + 1]
        return FakeProc(outputs[name])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    launcher = SlurmLauncher(system_name="local", username="alice", account="infra01", partition="normal")

    assert asyncio.run(launcher.find_job("live")) == (123, JobStatus.PENDING)
    assert asyncio.run(launcher.find_job("done")) is None
    assert asyncio.run(launcher.find_job("none")) is None
    assert asyncio.run(launcher.find_job("junk")) is None
    assert calls[0][:4] == ("squeue", "--me", "--name", "live")


def test_slurm_launch_honours_job_name(tmp_path) -> None:
    from swiss_ai_model_launch.launchers import SlurmLauncher

    launcher = SlurmLauncher(system_name="local", username="alice", account="infra01", partition="normal")
    env = str(tmp_path / "env.toml")
    req = _request(job_name="evalsvc-slurm").model_copy(update={"environment": env})
    args = launcher._get_launch_args_from_request(req)
    assert args.job_name == "evalsvc-slurm"
    unnamed = launcher._get_launch_args_from_request(_request().model_copy(update={"environment": env}))
    assert unnamed.job_name.startswith("swiss-ai_Apertus-8B-Instruct-2509_alice_")
