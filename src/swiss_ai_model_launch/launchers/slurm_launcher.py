import asyncio
import json
from importlib.resources import files
from pathlib import Path

from swiss_ai_model_launch.launchers.framework import render_master
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs, Topology
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import JobStatus, Launcher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.utils import (
    create_salt,
    decode_log,
    resolve_model_path,
)

_REMOTE_MODEL_REGISTRY = Path("/capstor/store/cscs/swissai/infra01/hf_models/models/")

_SGLANG_ENVIRONMENT = files("swiss_ai_model_launch.assets.envs").joinpath("sglang.toml")
_VLLM_ENVIRONMENT = files("swiss_ai_model_launch.assets.envs").joinpath("vllm.toml")

_PRECONFIGURED_MODELS = files("swiss_ai_model_launch.assets").joinpath("models.json")

_APP_WORKING_DIRECTORY = ".sml"


class SlurmLauncher(Launcher):
    def __init__(
        self,
        system_name: str,
        username: str,
        account: str,
        partition: str,
        reservation: str | None = None,
        model_registry: Path = _REMOTE_MODEL_REGISTRY,
        telemetry_endpoint: str | None = None,
    ):
        super().__init__(
            system_name=system_name,
            username=username,
            account=account,
            partition=partition,
            reservation=reservation,
            telemetry_endpoint=telemetry_endpoint,
        )
        self.model_registry = model_registry

    def _get_working_dir(self) -> Path:
        return Path.home() / _APP_WORKING_DIRECTORY

    def _get_launch_args_from_request(self, launch_request: LaunchRequest) -> LaunchArgs:
        model = launch_request.model
        job_name = f"{model.replace('/', '_')}_{self.username}_{create_salt(8)}"
        served_model_name = launch_request.served_model_name or f"{model}-{create_salt(4)}"
        return LaunchArgs(
            job_name=job_name,
            account=self.account,
            partition=self.partition,
            topology=Topology(
                replicas=launch_request.replicas,
                nodes_per_replica=launch_request.nodes_per_replica,
            ),
            time=launch_request.time,
            reservation=self.reservation,
            environment=launch_request.environment,
            framework=launch_request.framework,
            served_model_name=served_model_name,
            framework_args=(
                f"--model {resolve_model_path(model, self.model_registry, launch_request.model_path)} "
                f"--served-model-name {served_model_name} "
                "--host 0.0.0.0 " + (launch_request.framework_args if launch_request.framework_args else "")
            ),
            pre_launch_cmds=launch_request.pre_launch_cmds or "",
            telemetry_endpoint=self.telemetry_endpoint,
            use_router=launch_request.use_router,
        )

    def _get_local_env_file_path(self, launch_request: LaunchRequest) -> str:
        if launch_request.environment is not None:
            return str(Path(launch_request.environment).resolve())
        elif launch_request.framework == "sglang":
            return str(_SGLANG_ENVIRONMENT)
        elif launch_request.framework == "vllm":
            return str(_VLLM_ENVIRONMENT)
        else:
            raise ValueError(
                "`environment` is not provided in the launch request, "
                "and no default environment is available for the specified framework."
            )

    async def _sbatch(self, launch_args: LaunchArgs) -> int:
        working_dir = self._get_working_dir()
        working_dir.mkdir(parents=True, exist_ok=True)

        # Master.sh self-extracts its rank scripts at job start time
        # (under $HOME/.sml/job-${SLURM_JOB_ID}/). We only write master
        # locally so sbatch has something to submit.
        script_path = working_dir / f"job_{launch_args.job_name}.sh"
        script_path.write_text("#!/bin/bash\n" + render_master(launch_args))
        script_path.chmod(0o755)

        proc = await asyncio.create_subprocess_exec(
            "sbatch",
            "--chdir",
            str(working_dir),
            *launch_args.to_sbatch_args(),
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed (exit {proc.returncode}): {stderr.decode().strip()}")

        # sbatch prints: "Submitted batch job 12345"
        return int(stdout.decode().strip().split()[-1])

    async def get_preconfigured_models(self) -> list[ModelCatalogEntry]:
        return [ModelCatalogEntry(**item) for item in json.loads(_PRECONFIGURED_MODELS.read_text())]

    async def launch_with_args(self, launch_args: LaunchArgs) -> tuple[int, str]:
        launch_args = launch_args.model_copy(
            update={
                "reservation": self.reservation,
                "environment": str(Path(launch_args.environment).resolve()),
            }
        )
        job_id = await self._sbatch(launch_args)
        return job_id, launch_args.served_model_name

    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]:
        env_path = self._get_local_env_file_path(launch_request)

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
            return JobStatus.from_str(state)

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
            return JobStatus.from_str(lines[0].split()[0])

        return JobStatus.UNKNOWN

    async def get_job_logs(self, job_id: int) -> tuple[str, str]:
        log_dir = self._get_working_dir() / "logs" / str(job_id)

        try:
            out_log = decode_log((log_dir / "log.out").read_bytes())
        except FileNotFoundError:
            out_log = ""

        try:
            err_log = decode_log((log_dir / "log.err").read_bytes())
        except FileNotFoundError:
            err_log = ""

        return out_log, err_log

    def get_log_dir(self, job_id: int) -> str:
        return str(self._get_working_dir() / "logs" / str(job_id))

    async def cancel_job(self, job_id: int) -> None:
        proc = await asyncio.create_subprocess_exec(
            "scancel",
            str(job_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"scancel failed (exit {proc.returncode}): {stderr.decode().strip()}")
