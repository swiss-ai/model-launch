from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import firecrest as f7t

from swiss_ai_model_launch.launchers import FirecRESTLauncher, Launcher, SlurmLauncher
from swiss_ai_model_launch.launchers.utils import create_salt

from .core import build_run_config
from .models import LoadtestConfig, ServerConfig


@dataclass
class ClusterLoadtestConfig:
    container_image: str
    time: str = "00:30:00"
    cpus_per_task: int = 4
    wait: bool = True
    reservation: str | None = None


def build_cluster_loadtest_script(
    *,
    bench: LoadtestConfig,
    cluster: ClusterLoadtestConfig,
    account: str,
    partition: str,
    reservation: str | None,
    run_label: str,
    prompts_path: str,
    container_mounts: str,
) -> str:
    job_name = f"sml_loadtest_{bench.scenario}_{create_salt(6)}"
    reservation_line = f"#SBATCH --reservation={reservation}\n" if reservation else ""

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --time={cluster.time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cluster.cpus_per_task}
#SBATCH --partition={partition}
{reservation_line}#SBATCH --output=logs/%j/loadtest.out
#SBATCH --error=logs/%j/loadtest.out

set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST SLURM_CPU_BIND_VERBOSE

RUN_DIR="${{PWD}}/{run_label}"
mkdir -p "$RUN_DIR" logs/"${{SLURM_JOB_ID}}"

echo "Starting cluster k6 loadtest in $RUN_DIR"
echo "Container image: {cluster.container_image}"

srun --nodes=1 --ntasks=1 \\
    --cpus-per-task={cluster.cpus_per_task} \\
    --container-image="{cluster.container_image}" \\
    --container-mounts="{container_mounts}" \\
    --container-workdir=/work \\
    sh -lc 'k6 run \\
        --env RUN_CONFIG_JSON="$(cat /work/run_config.json)" \\
        --env PROMPTS_FILE="{prompts_path}" \\
        --summary-export /work/summary.json \\
        /work/script.js'

echo "Summary written to $RUN_DIR/summary.json"
"""


async def _run_checked(*cmd: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}): {stderr.decode().strip()}")
    return stdout.decode()


async def _wait_for_local_slurm_job(
    job_id: int,
    *,
    log_path: Path,
    poll_seconds: int = 10,
) -> None:
    while True:
        state = (await _run_checked("squeue", "-j", str(job_id), "-h", "-o", "%T")).strip()
        if not state:
            break
        await asyncio.sleep(poll_seconds)

    state_report = (await _run_checked("sacct", "-j", str(job_id), "-n", "-o", "State,ExitCode", "--parsable2")).strip()
    first_line = state_report.splitlines()[0] if state_report else ""
    if first_line and not first_line.startswith("COMPLETED|0:0"):
        raise RuntimeError(
            f"Cluster loadtest job {job_id} did not complete successfully: {first_line}. Log: {log_path}"
        )


_FIRECREST_ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
}


async def _wait_for_firecrest_job(
    launcher: FirecRESTLauncher,
    job_id: int,
    *,
    log_path: str,
    poll_seconds: int = 10,
) -> None:
    while True:
        job_info = await launcher.client.job_info(
            system_name=launcher.system_name,
            jobid=str(job_id),
        )
        raw_state = str(job_info[0]["status"]["state"]).split()[0].upper()
        if raw_state not in _FIRECREST_ACTIVE_STATES:
            if raw_state != "COMPLETED":
                raise RuntimeError(f"Cluster loadtest job {job_id} ended with state {raw_state}. Log: {log_path}")
            return
        await asyncio.sleep(poll_seconds)


def _write_local_run_files(
    *,
    run_dir: Path,
    server: ServerConfig,
    bench: LoadtestConfig,
    k6_script: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(k6_script, run_dir / "script.js")
    (run_dir / "run_config.json").write_text(json.dumps(build_run_config(server, bench), separators=(",", ":")))


def _container_mounts_for_external_prompts(run_label: str, prompts_file: Path) -> tuple[str, str]:
    if not prompts_file.is_absolute():
        raise ValueError(
            "Loadtest prompts must be an absolute path visible on the cluster. "
            "Use --loadtest-prompts-file or SML_LOADTEST_PROMPTS_FILE."
        )
    mounts = [f"${{PWD}}/{run_label}:/work"]
    top_level = prompts_file.parts[1] if len(prompts_file.parts) > 1 else ""
    if top_level:
        mounts.append(f"/{top_level}:/{top_level}")
    return str(prompts_file), ",".join(mounts)


async def submit_cluster_loadtest(
    *,
    launcher: Launcher,
    server: ServerConfig,
    bench: LoadtestConfig,
    k6_script: Path,
    prompts_file: Path,
    summary_path: Path,
    cluster: ClusterLoadtestConfig,
) -> int:
    run_label = f"loadtest_{bench.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{create_salt(6)}"
    prompts_path, container_mounts = _container_mounts_for_external_prompts(run_label, prompts_file)
    script = build_cluster_loadtest_script(
        bench=bench,
        cluster=cluster,
        account=launcher.account,
        partition=launcher.partition,
        reservation=cluster.reservation or launcher.reservation,
        run_label=run_label,
        prompts_path=prompts_path,
        container_mounts=container_mounts,
    )

    if isinstance(launcher, SlurmLauncher):
        working_dir = Path.home() / ".sml" / "loadtest_runs"
        run_dir = working_dir / run_label
        _write_local_run_files(
            run_dir=run_dir,
            server=server,
            bench=bench,
            k6_script=k6_script,
        )
        script_path = working_dir / f"{run_label}.sh"
        script_path.write_text(script)
        stdout = await _run_checked("sbatch", "--chdir", str(working_dir), str(script_path))
        job_id = int(stdout.strip().split()[-1])
        if cluster.wait:
            await _wait_for_local_slurm_job(
                job_id,
                log_path=working_dir / "logs" / str(job_id) / "loadtest.out",
            )
            remote_summary = working_dir / run_label / "summary.json"
            if remote_summary.exists():
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(remote_summary, summary_path)
        return job_id

    if isinstance(launcher, FirecRESTLauncher):
        firecrest_working_dir = str(launcher._get_working_dir())
        remote_run_dir = str(Path(firecrest_working_dir) / run_label)
        await launcher.client.mkdir(
            system_name=launcher.system_name,
            path=remote_run_dir,
            create_parents=True,
        )
        with tempfile.TemporaryDirectory(prefix="sml_loadtest_") as tmp_dir:
            tmp_run_dir = Path(tmp_dir)
            _write_local_run_files(
                run_dir=tmp_run_dir,
                server=server,
                bench=bench,
                k6_script=k6_script,
            )
            for filename in ("script.js", "run_config.json"):
                await launcher.client.upload(
                    system_name=launcher.system_name,
                    local_file=tmp_run_dir / filename,
                    directory=remote_run_dir,
                    filename=filename,
                    account=launcher.account,
                    blocking=True,
                )

        report = await launcher.client.submit(
            system_name=launcher.system_name,
            working_dir=firecrest_working_dir,
            script_str=script,
            account=launcher.account,
        )
        job_id = int(report["jobId"])
        if cluster.wait:
            await _wait_for_firecrest_job(
                launcher,
                job_id,
                log_path=str(Path(firecrest_working_dir) / "logs" / str(job_id) / "loadtest.out"),
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                await launcher.client.download(
                    system_name=launcher.system_name,
                    source_path=str(Path(firecrest_working_dir) / run_label / "summary.json"),
                    target_path=summary_path,
                    account=launcher.account,
                    blocking=True,
                )
            except f7t.FirecrestException as e:
                raise RuntimeError(f"Could not download cluster loadtest summary for job {job_id}: {e}") from e
        return job_id

    raise TypeError(f"Cluster loadtests are not supported for launcher type {type(launcher).__name__}")
