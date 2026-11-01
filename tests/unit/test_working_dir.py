"""A launcher's working directory: ~/.sml by default, or a configured one.

Jobs run from it and keep their env files, rank scripts and logs there. A
service launching for many people (evals-svc) puts it off the FirecREST
user's home directory, whose small quota filled up.
"""

from pathlib import Path
from typing import Any

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.slurm_launcher import SlurmLauncher


class _FakeClient:
    def __init__(self) -> None:
        self.mkdirs: list[str] = []
        self.uploads: list[str] = []
        self.submits: list[str] = []

    async def mkdir(self, system_name: str, path: str, create_parents: bool = False) -> None:
        self.mkdirs.append(path)

    async def upload(self, system_name: str, local_file: str, directory: str, filename: str, **_: Any) -> None:
        self.uploads.append(directory)

    async def submit(self, system_name: str, working_dir: str, script_str: str, **_: Any) -> dict[str, Any]:
        self.submits.append(working_dir)
        return {"jobId": 7}


def _launcher(client: _FakeClient, working_dir: str | None = None) -> FirecRESTLauncher:
    return FirecRESTLauncher(
        client=client,  # type: ignore[arg-type]
        system_name="clariden",
        username="u",
        account="proj01",
        partition="normal",
        working_dir=working_dir,
    )


def test_firecrest_defaults_to_the_users_sml() -> None:
    launcher = _launcher(_FakeClient())
    assert launcher._get_working_dir() == "/users/u/.sml"
    assert launcher.get_log_dir(7) == "/users/u/.sml/logs/7"


async def test_firecrest_uses_the_configured_working_dir(tmp_path: Path) -> None:
    client = _FakeClient()
    launcher = _launcher(client, "/capstor/store/x/model_launch/")
    assert launcher.get_log_dir(7) == "/capstor/store/x/model_launch/logs/7"
    assert "/capstor/store/x/model_launch/logs/7/log.out" in launcher.get_tail_hint(7)

    env = tmp_path / "env.toml"
    env.write_text("")
    remote = await launcher._upload_env_file(str(env), "vllm")
    assert remote.startswith("/capstor/store/x/model_launch/env_vllm_")
    assert await launcher._submit_script("#!/bin/bash\n") == 7
    assert client.mkdirs == ["/capstor/store/x/model_launch/logs"]
    assert client.uploads == client.submits == ["/capstor/store/x/model_launch"]


def test_slurm_uses_the_configured_working_dir() -> None:
    kwargs = {"system_name": "c", "username": "u", "account": "a", "partition": "p"}
    assert SlurmLauncher(**kwargs)._get_working_dir() == Path.home() / ".sml"
    launcher = SlurmLauncher(**kwargs, working_dir="/scratch/u/sml")
    assert launcher._get_working_dir() == Path("/scratch/u/sml")
    assert launcher.get_log_dir(7) == "/scratch/u/sml/logs/7"
