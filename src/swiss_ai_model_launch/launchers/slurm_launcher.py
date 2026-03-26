import asyncio
import json
import shutil
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import JobStatus, Launcher
from swiss_ai_model_launch.launchers.utils import create_salt, render_job_script

_REMOTE_MODEL_REGISTRY = Path("/capstor/store/cscs/swissai/infra01/hf_models/models/")

_SGLANG_ENVIRONMENT = files("swiss_ai_model_launch.assets.envs").joinpath("sglang.toml")
_VLLM_ENVIRONMENT = files("swiss_ai_model_launch.assets.envs").joinpath("vllm.toml")

_PRECONFIGURED_MODELS = files("swiss_ai_model_launch.assets").joinpath("models.json")

_APP_WORKING_DIRECTORY = ".sml"

_SLURM_STATE_MAP: dict[str, JobStatus] = {
    "PENDING": JobStatus.PENDING,
    "CONFIGURING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    "TIMEOUT": JobStatus.TIMEOUT,
}


class SlurmLauncher(Launcher):
    def __init__(
        self,
        system_name: str,
        username: str,
        account: str,
        partition: str,
        model_registry: Path = _REMOTE_MODEL_REGISTRY,
        telemetry_endpoint: str | None = None,
    ):
        super().__init__(
            system_name=system_name,
            username=username,
            account=account,
            partition=partition,
            telemetry_endpoint=telemetry_endpoint,
        )
        self.model_registry = model_registry

    def _get_working_dir(self) -> Path:
        return Path.home() / _APP_WORKING_DIRECTORY

    def _get_launch_args_from_request(
        self, launch_request: LaunchRequest
    ) -> LaunchArgs:
        vendor = launch_request.vendor
        model_name = launch_request.model_name
        job_name = f"{vendor}_{model_name}_{self.username}_{create_salt(8)}"
        served_model_name = (
            launch_request.served_model_name
            or f"{vendor}/{model_name}-{create_salt(4)}"
        )
        return LaunchArgs(
            job_name=job_name,
            account=self.account,
            partition=self.partition,
            workers=launch_request.workers,
            nodes_per_worker=launch_request.nodes_per_worker,
            time=launch_request.time,
            environment=launch_request.environment,
            framework=launch_request.framework,
            served_model_name=served_model_name,
            framework_args=(
                f"--model {str(self.model_registry / vendor / model_name)} "
                f"--served-model-name {served_model_name} "
                "--host 0.0.0.0 "
                "--port 8080 "
                + (
                    launch_request.framework_args
                    if launch_request.framework_args
                    else ""
                )
            ),
            pre_launch_cmds=launch_request.pre_launch_cmds or "",
            telemetry_endpoint=self.telemetry_endpoint,
            use_router=launch_request.use_router,
        )

    def _get_local_env_file_path(self, launch_request: LaunchRequest) -> str:
        if launch_request.environment is not None:
            return launch_request.environment
        elif launch_request.framework == "sglang":
            return str(_SGLANG_ENVIRONMENT)
        elif launch_request.framework == "vllm":
            return str(_VLLM_ENVIRONMENT)
        else:
            raise ValueError(
                "`environment` is not provided in the launch request, "
                "and no default environment is available for the specified framework."
            )

    def _create_env_file_path(self, launch_request: LaunchRequest) -> str:
        working_dir = self._get_working_dir()
        working_dir.mkdir(parents=True, exist_ok=True)

        local_env_path = self._get_local_env_file_path(launch_request)
        env_filename = "env_{}_{}_{}.toml".format(
            launch_request.framework,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            create_salt(8),
        )

        shutil.copy(local_env_path, working_dir / env_filename)
        return str(working_dir / env_filename)

    async def _sbatch(self, launch_args: LaunchArgs) -> int:
        script_str = render_job_script(launch_args)
        working_dir = self._get_working_dir()
        working_dir.mkdir(parents=True, exist_ok=True)

        script_path = working_dir / f"job_{launch_args.job_name}.sh"
        script_path.write_text(script_str)

        proc = await asyncio.create_subprocess_exec(
            "sbatch",
            "--chdir",
            str(working_dir),
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (exit {proc.returncode}): {stderr.decode().strip()}"
            )

        # sbatch prints: "Submitted batch job 12345"
        return int(stdout.decode().strip().split()[-1])

    async def get_preconfigured_models(self) -> list[LaunchRequest]:
        return [
            LaunchRequest(**item)
            for item in json.loads(_PRECONFIGURED_MODELS.read_text())
        ]

    async def launch_with_args(self, launch_args: LaunchArgs) -> tuple[int, str]:
        job_id = await self._sbatch(launch_args)
        return job_id, launch_args.served_model_name

    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]:
        env_path = self._create_env_file_path(launch_request)

        launch_args = self._get_launch_args_from_request(
            LaunchRequest.model_copy(
                launch_request,
                update={"environment": env_path},
            )
        )

        job_id = await self._sbatch(launch_args)
        return job_id, launch_args.served_model_name

    async def get_job_status(self, job_id: int) -> JobStatus:
        proc = await asyncio.create_subprocess_exec(
            "squeue",
            "-j",
            str(job_id),
            "-h",
            "-o",
            "%T",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        state = stdout.decode().strip()

        if state:
            return _SLURM_STATE_MAP.get(state, JobStatus.UNKNOWN)

        # Job not in squeue — check sacct for terminal state
        proc = await asyncio.create_subprocess_exec(
            "sacct",
            "-j",
            str(job_id),
            "-n",
            "-o",
            "State",
            "--parsable2",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        lines = [line.strip() for line in stdout.decode().splitlines() if line.strip()]
        if lines:
            return _SLURM_STATE_MAP.get(lines[0].split()[0], JobStatus.UNKNOWN)

        return JobStatus.UNKNOWN

    async def get_job_logs(self, job_id: int) -> tuple[str, str]:
        log_dir = self._get_working_dir() / "logs" / str(job_id)

        try:
            out_log = (log_dir / "log.out").read_text()
        except FileNotFoundError:
            out_log = ""

        try:
            err_log = (log_dir / "log.err").read_text()
        except FileNotFoundError:
            err_log = ""

        return out_log, err_log

    async def cancel_job(self, job_id: int) -> None:
        proc = await asyncio.create_subprocess_exec(
            "scancel",
            str(job_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"scancel failed (exit {proc.returncode}): {stderr.decode().strip()}"
            )
