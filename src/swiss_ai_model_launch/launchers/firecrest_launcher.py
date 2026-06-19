import json
import tempfile
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import firecrest as f7t

from swiss_ai_model_launch.launchers.framework import render_master
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import Launcher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.topology import Topology
from swiss_ai_model_launch.launchers.utils import (
    call_with_firecrest_retry,
    create_salt,
    decode_log,
    render_sbatch_header,
    resolve_model_path,
)

_REMOTE_MODEL_REGISTRY = Path("/capstor/store/cscs/swissai/infra01/hf_models/models/")

_SGLANG_ENVIRONMENT = files("swiss_ai_model_launch.assets.envs").joinpath("sglang.toml")
_VLLM_ENVIRONMENT = files("swiss_ai_model_launch.assets.envs").joinpath("vllm.toml")

_PRECONFIGURED_MODELS = files("swiss_ai_model_launch.assets").joinpath("models.json")

_APP_WORKING_DIRECTORY = ".sml"

_FIRECREST_UNKNOWN_TIMES = frozenset({"", "N/A", "Unknown", "None"})


def _firecrest_time(value: object) -> str | None:
    """Normalise a FirecREST/SLURM job time field to a display string or None.

    Handles the shapes seen across API versions: an epoch int/float, SLURM's
    ``{"set", "infinite", "number"}`` wrapper, or an already-formatted string.
    """
    if isinstance(value, dict):
        if not value.get("set", True) or value.get("infinite"):
            return None
        value = value.get("number")
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%dT%H:%M:%S") if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped in _FIRECREST_UNKNOWN_TIMES else stripped
    return None


class FirecRESTLauncher(Launcher):
    def __init__(
        self,
        client: f7t.v2.AsyncFirecrest,
        system_name: str,
        username: str,
        account: str,
        partition: str,
        reservation: str | None = None,
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
        self.client = client

    @classmethod
    async def from_client(
        cls,
        client: f7t.v2.AsyncFirecrest,
        system_name: str,
        partition: str,
        reservation: str | None = None,
        account: str | None = None,
        telemetry_endpoint: str | None = None,
    ) -> "FirecRESTLauncher":
        user_info = await call_with_firecrest_retry(lambda: client.userinfo(system_name))
        return cls(
            client=client,
            system_name=system_name,
            username=user_info["user"]["name"],
            account=account or user_info["group"]["name"],
            partition=partition,
            reservation=reservation,
            telemetry_endpoint=telemetry_endpoint,
        )

    def _get_user_dir(self) -> str:
        return f"/users/{self.username}"

    def _get_working_dir(self) -> str:
        return str(Path(self._get_user_dir()) / _APP_WORKING_DIRECTORY)

    def _get_launch_args_from_request(
        self,
        launch_request: LaunchRequest,
    ) -> LaunchArgs:
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
            environment=launch_request.environment,
            framework=launch_request.framework,
            served_model_name=served_model_name,
            framework_args=(
                f"--model {resolve_model_path(model, _REMOTE_MODEL_REGISTRY, launch_request.model_path)} "
                f"--served-model-name {served_model_name} "
                "--host 0.0.0.0 " + (launch_request.framework_args if launch_request.framework_args else "")
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
                "`envionment` is not provided in the launch request, "
                "and no default environment is available for the specified framework."
            )

    async def _upload_env_file(self, local_env_path: str, framework: str) -> str:
        working_dir = self._get_working_dir()
        await call_with_firecrest_retry(
            lambda: self.client.mkdir(
                system_name=self.system_name,
                path=working_dir,
                create_parents=True,
            )
        )
        remote_env_filename = "env_{}_{}_{}.toml".format(
            framework,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            create_salt(8),
        )
        await call_with_firecrest_retry(
            lambda: self.client.upload(
                system_name=self.system_name,
                local_file=local_env_path,
                directory=working_dir,
                filename=remote_env_filename,
                account=self.account,
                blocking=True,
            )
        )
        return str(Path(working_dir) / remote_env_filename)

    async def _create_remote_env_file_path(self, launch_request: LaunchRequest) -> str:
        return await self._upload_env_file(
            self._get_local_env_file_path(launch_request),
            launch_request.framework,
        )

    async def _prepare_launch_args(self, launch_args: LaunchArgs) -> LaunchArgs:
        remote_env_path = await self._upload_env_file(launch_args.environment, launch_args.framework)
        return launch_args.model_copy(update={"environment": remote_env_path})

    async def _submit_one(self, launch_args: LaunchArgs) -> int:
        script_str = render_sbatch_header(launch_args, reservation=self.reservation) + render_master(launch_args)
        job_submission_report = await call_with_firecrest_retry(
            lambda: self.client.submit(
                system_name=self.system_name,
                working_dir=self._get_working_dir(),
                script_str=script_str,
                account=self.account,
            )
        )
        return int(job_submission_report["jobId"])

    async def launch_with_args(self, launch_args: LaunchArgs) -> tuple[int, str]:
        prepared = await self._prepare_launch_args(launch_args)
        job_id = await self._submit_one(prepared)
        return job_id, prepared.served_model_name

    async def read_job_file(self, job_id: int, filename: str) -> str | None:
        remote_path = Path(self._get_working_dir()) / "logs" / str(job_id) / filename
        with tempfile.TemporaryDirectory(prefix=f"sml_logs_{job_id}_") as target_dir:
            target_path = Path(target_dir) / Path(filename).name
            try:
                await call_with_firecrest_retry(
                    lambda: self.client.download(
                        system_name=self.system_name,
                        source_path=str(remote_path),
                        target_path=target_path,
                        account=self.account,
                        blocking=True,
                    )
                )
                return decode_log(target_path.read_bytes())
            except (FileNotFoundError, f7t.FirecrestException):
                return None

    async def get_preconfigured_models(self) -> list[ModelCatalogEntry]:
        return [ModelCatalogEntry(**item) for item in json.loads(_PRECONFIGURED_MODELS.read_text())]

    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]:
        remote_env_path = await self._create_remote_env_file_path(launch_request)

        launch_args = self._get_launch_args_from_request(
            LaunchRequest.model_copy(
                launch_request,
                update={"environment": remote_env_path},
            )
        )

        script_str = render_sbatch_header(launch_args, reservation=self.reservation) + render_master(launch_args)
        job_submission_report = await call_with_firecrest_retry(
            lambda: self.client.submit(
                system_name=self.system_name,
                working_dir=self._get_working_dir(),
                script_str=script_str,
                account=self.account,
            )
        )

        return int(job_submission_report["jobId"]), launch_args.served_model_name

    async def get_job_status(self, job_id: int) -> JobStatus:
        job_info = await call_with_firecrest_retry(
            lambda: self.client.job_info(
                system_name=self.system_name,
                jobid=str(job_id),
            )
        )
        return JobStatus.from_str(str(job_info[0]["status"]["state"]))

    async def get_job_times(self, job_id: int) -> tuple[str | None, str | None]:
        job_info = await call_with_firecrest_retry(
            lambda: self.client.job_info(
                system_name=self.system_name,
                jobid=str(job_id),
            )
        )
        # The v2 job object carries SLURM's `time` block (start/end). Parse
        # defensively — the field shape varies across API versions — and fall
        # back to (None, None) so the chain panel just shows its dependency hint.
        try:
            time_info = job_info[0].get("time") or {}
            return _firecrest_time(time_info.get("start")), _firecrest_time(time_info.get("end"))
        except (AttributeError, TypeError, IndexError, KeyError):
            return None, None

    async def get_job_logs(self, job_id: int) -> tuple[str, str]:
        log_dir = Path(self._get_working_dir()) / "logs" / str(job_id)

        with tempfile.TemporaryDirectory(prefix=f"sml_logs_{job_id}_") as target_dir:
            target_dir_path = Path(target_dir)

            try:
                await call_with_firecrest_retry(
                    lambda: self.client.download(
                        system_name=self.system_name,
                        source_path=str(log_dir / "log.out"),
                        target_path=target_dir_path / "log.out",
                        account=self.account,
                        blocking=True,
                    )
                )
                with open(target_dir_path / "log.out", "rb") as out_f:
                    out_log = decode_log(out_f.read())
            except (FileNotFoundError, f7t.FirecrestException):
                out_log = ""

            try:
                await call_with_firecrest_retry(
                    lambda: self.client.download(
                        system_name=self.system_name,
                        source_path=str(log_dir / "log.err"),
                        target_path=target_dir_path / "log.err",
                        account=self.account,
                        blocking=True,
                    )
                )
                with open(target_dir_path / "log.err", "rb") as err_f:
                    err_log = decode_log(err_f.read())
            except (FileNotFoundError, f7t.FirecrestException):
                err_log = ""

            return out_log, err_log

    def get_tail_hint(self, job_id: int) -> str:
        return f"ssh <host> tail -f ~/.sml/logs/{job_id}/log.out\n  (replace <host> with your cluster SSH alias)"

    async def cancel_job(self, job_id: int) -> None:
        try:
            await call_with_firecrest_retry(
                lambda: self.client.cancel_job(
                    system_name=self.system_name,
                    jobid=str(job_id),
                )
            )
        except f7t.UnexpectedStatusException as exc:
            # FirecREST returns 500 when scancel succeeds (exit_status:0) but
            # the job already ended on its own — Slurm prints
            # "Job/step already completing or completed" to stderr and the
            # gateway treats any stderr as failure. The job is gone, which is
            # what cancel asked for, so treat this as success.
            try:
                message = exc.responses[-1].json().get("message", "")
            except (AttributeError, IndexError, ValueError):
                raise exc from None
            if "already completing or completed" not in message:
                raise
