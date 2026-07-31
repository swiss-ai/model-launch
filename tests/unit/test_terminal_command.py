"""The per-launcher command that opens an interactive shell on a replica's node."""

from swiss_ai_model_launch.launchers import FirecRESTLauncher, SlurmLauncher
from swiss_ai_model_launch.launchers.launcher import Launcher

_SRUN = "srun --overlap --jobid=7 --nodes=1 --ntasks=1 --nodelist=nid9 --pty bash -l"


def _slurm() -> SlurmLauncher:
    return SlurmLauncher(system_name="local", username="u", account="a", partition="normal")


def _firecrest(ssh_host: str | None) -> FirecRESTLauncher:
    # terminal_command never touches the client, so a placeholder stands in for it.
    return FirecRESTLauncher(
        client=object(),  # type: ignore[arg-type]
        system_name="clariden",
        username="u",
        account="a",
        partition="normal",
        ssh_host=ssh_host,
    )


def test_slurm_attaches_via_srun_overlap() -> None:
    cmd = _slurm().terminal_command(7, "nid9")
    assert cmd.available
    assert cmd.argv == [
        "srun",
        "--overlap",
        "--jobid=7",
        "--nodes=1",
        "--ntasks=1",
        "--nodelist=nid9",
        "--pty",
        "bash",
        "-l",
    ]
    assert cmd.display == _SRUN  # copy/paste-safe rendering of the same command


def test_firecrest_wraps_srun_in_ssh_when_host_known() -> None:
    cmd = _firecrest("clariden").terminal_command(7, "nid9")
    assert cmd.available
    # ssh -t forces a remote PTY; the srun shell is passed as a single argument.
    assert cmd.argv == ["ssh", "-t", "clariden", _SRUN]
    assert cmd.display == f"ssh -t clariden '{_SRUN}'"


def test_firecrest_unavailable_without_ssh_host_but_still_shows_command() -> None:
    cmd = _firecrest(None).terminal_command(7, "nid9")
    assert not cmd.available
    assert cmd.argv == []
    assert cmd.display == _SRUN  # the command to run manually after SSHing in
    assert cmd.reason is not None and "SSH host" in cmd.reason


def test_base_launcher_reports_terminal_unsupported() -> None:
    class _Bare(Launcher):
        async def get_preconfigured_models(self):  # type: ignore[no-untyped-def]
            return []

        async def launch_model(self, launch_request):  # type: ignore[no-untyped-def]
            return 0, ""

        async def launch_with_args(self, launch_args):  # type: ignore[no-untyped-def]
            return 0, ""

        async def get_job_status(self, job_id):  # type: ignore[no-untyped-def]
            ...

        async def cancel_job(self, job_id):  # type: ignore[no-untyped-def]
            ...

        def get_tail_hint(self, job_id):  # type: ignore[no-untyped-def]
            return ""

        async def read_job_file(self, job_id, filename):  # type: ignore[no-untyped-def]
            return None

    cmd = _Bare("s", "u", "a", "p").terminal_command(1, "n")
    assert not cmd.available
    assert cmd.argv == []
