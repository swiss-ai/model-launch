import json
import tempfile
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import firecrest as f7t

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import JobStatus, Launcher
from swiss_ai_model_launch.launchers.utils import (
    create_salt,
    decode_log,
    render_job_script,
)

_REMOTE_MODEL_REGISTRY = Path("/capstor/store/cscs/swissai/infra01/hf_models/models/")

_SGLANG_ENVIRONMENT = files("swiss_ai_model_launch.assets.envs").joinpath("sglang.toml")
_VLLM_ENVIRONMENT = files("swiss_ai_model_launch.assets.envs").joinpath("vllm.toml")

_PRECONFIGURED_MODELS = files("swiss_ai_model_launch.assets").joinpath("models.json")

_APP_WORKING_DIRECTORY = ".sml"


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
        telemetry_endpoint: str | None = None,
    ) -> "FirecRESTLauncher":
        user_info = await client.userinfo(system_name)
        return cls(
            client=client,
            system_name=system_name,
            username=user_info["user"]["name"],
            account=user_info["group"]["name"],
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
            reservation=self.reservation,
            environment=launch_request.environment,
            framework=launch_request.framework,
            served_model_name=served_model_name,
            framework_args=(
                f"--model {str(_REMOTE_MODEL_REGISTRY / vendor / model_name)} "
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
                "`envionment` is not provided in the launch request, "
                "and no default environment is available for the specified framework."
            )

    async def _upload_env_file(self, local_env_path: str, framework: str) -> str:
        working_dir = self._get_working_dir()
        await self.client.mkdir(
            system_name=self.system_name,
            path=working_dir,
            create_parents=True,
        )
        remote_env_filename = "env_{}_{}_{}.toml".format(
            framework,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            create_salt(8),
        )
        await self.client.upload(
            system_name=self.system_name,
            local_file=local_env_path,
            directory=working_dir,
            filename=remote_env_filename,
            account=self.account,
            blocking=True,
        )
        return str(Path(working_dir) / remote_env_filename)

    async def _create_remote_env_file_path(self, launch_request: LaunchRequest) -> str:
        return await self._upload_env_file(
            self._get_local_env_file_path(launch_request),
            launch_request.framework,
        )

    async def launch_with_args(self, launch_args: LaunchArgs) -> tuple[int, str]:
        remote_env_path = await self._upload_env_file(
            launch_args.environment, launch_args.framework
        )
        launch_args = launch_args.model_copy(
            update={"environment": remote_env_path, "reservation": self.reservation}
        )
        script_str = render_job_script(launch_args)
        job_submission_report = await self.client.submit(
            system_name=self.system_name,
            working_dir=self._get_working_dir(),
            script_str=script_str,
            account=self.account,
        )
        return int(job_submission_report["jobId"]), launch_args.served_model_name

    async def get_preconfigured_models(self) -> list[LaunchRequest]:
        return [
            LaunchRequest(**item)
            for item in json.loads(_PRECONFIGURED_MODELS.read_text())
        ]

    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]:
        remote_env_path = await self._create_remote_env_file_path(launch_request)

        launch_args = self._get_launch_args_from_request(
            LaunchRequest.model_copy(
                launch_request,
                update={"environment": remote_env_path},
            )
        )

        script_str = render_job_script(launch_args)
        job_submission_report = await self.client.submit(
            system_name=self.system_name,
            working_dir=self._get_working_dir(),
            script_str=script_str,
            account=self.account,
        )

        return int(job_submission_report["jobId"]), launch_args.served_model_name

    async def get_job_status(self, job_id: int) -> JobStatus:
        job_info = await self.client.job_info(
            system_name=self.system_name,
            jobid=str(job_id),
            # account=self.account,  # TODO
        )
        return JobStatus.from_str(str(job_info[0]["status"]["state"]))

    async def get_job_logs(self, job_id: int) -> tuple[str, str]:
        log_dir = Path(self._get_working_dir()) / "logs" / str(job_id)

        with tempfile.TemporaryDirectory(prefix=f"sml_logs_{job_id}_") as target_dir:
            target_dir_path = Path(target_dir)

            try:
                await self.client.download(
                    system_name=self.system_name,
                    source_path=str(log_dir / "log.out"),
                    target_path=target_dir_path / "log.out",
                    account=self.account,
                    blocking=True,
                )
                with open(target_dir_path / "log.out", "rb") as out_f:
                    out_log = decode_log(out_f.read())
            except FileNotFoundError:
                out_log = ""

            try:
                await self.client.download(
                    system_name=self.system_name,
                    source_path=str(log_dir / "log.err"),
                    target_path=target_dir_path / "log.err",
                    account=self.account,
                    blocking=True,
                )
                with open(target_dir_path / "log.err", "rb") as err_f:
                    err_log = decode_log(err_f.read())
            except FileNotFoundError:
                err_log = ""

            return out_log, err_log

    async def cancel_job(self, job_id: int) -> None:
        await self.client.cancel_job(
            system_name=self.system_name,
            jobid=str(job_id),
        )
