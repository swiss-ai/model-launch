# Python API

The `swiss_ai_model_launch` package exposes a Python API so you can submit, monitor, and cancel jobs from your own scripts — no interactive prompts, no TUI. This is the right choice when:

- You need to launch multiple models concurrently.
- You're orchestrating jobs from CI, a notebook, or another Python program.
- You want fine-grained control over job lifecycle in code.

For the interactive flow, see [`sml`](usage-sml.md). For a fully declarative one-shot CLI invocation, see [`sml advanced`](usage-advanced.md).

## Installation

The Python API is part of the same package as the CLI. No extra dependencies are needed — install once and both surfaces are available:

```bash
pip install swiss-ai-model-launch
```

## Choosing a launcher

There are two launcher implementations that share a common interface:

| Launcher | When to use |
| --- | --- |
| `SlurmLauncher` | You are running on a node that already has `sbatch` / `squeue` / `sacct` in `$PATH` (i.e. you are on the cluster or on a login node). |
| `FirecRESTLauncher` | You are running off-cluster and submitting jobs to a remote HPC system through the [FirecREST](https://firecrest.readthedocs.io/) API. |

Both expose the same async methods, so switching between them requires only changing the constructor call.

## `SlurmLauncher`

Use this when your script runs directly on the cluster.

```python
import asyncio
import getpass
import grp
import os

from swiss_ai_model_launch import SlurmLauncher

async def main() -> None:
    username = getpass.getuser()
    account = grp.getgrgid(os.getgid()).gr_name

    launcher = SlurmLauncher(
        system_name="local",   # identifier used in logs; "local" is conventional
        username=username,
        account=account,
        partition="normal",
        reservation=None,      # optional SLURM reservation
    )

asyncio.run(main())
```

## `FirecRESTLauncher`

Use this when your script runs off-cluster and connects to a remote HPC system via FirecREST. Authentication is handled by a `firecrest.v2.AsyncFirecrest` client; how you obtain the client depends on your site's auth setup (client-credentials, device-code, etc.).

```python
import asyncio
import firecrest as f7t

from swiss_ai_model_launch import FirecRESTLauncher

async def main() -> None:
    auth = f7t.ClientCredentialsAuth(
        client_id="my-client-id",
        client_secret="my-client-secret",
        token_uri="https://auth.example.com/token",
    )
    client = f7t.v2.AsyncFirecrest(
        firecrest_url="https://firecrest.example.com",
        authorization=auth,
    )

    launcher = await FirecRESTLauncher.from_client(
        client=client,
        system_name="clariden",
        partition="normal",
        reservation=None,
    )

asyncio.run(main())
```

`FirecRESTLauncher.from_client` calls the API to resolve your username and primary group, so `username` and `account` are set automatically.

## Submitting a job

There are two ways to submit: using the model **catalog** or specifying every argument explicitly.

### Option 1 — catalog-based launch (`launch_model`)

`launch_model` takes a `LaunchRequest` built from the catalog and fills in model paths, environment files, and framework flags automatically.

```python
from swiss_ai_model_launch import LaunchRequest

launch_request = LaunchRequest(
    model="swiss-ai/Apertus-8B-Instruct-2509",
    framework="sglang",
    workers=1,
    nodes_per_worker=1,
    time="02:00:00",
)

job_id, served_model_name = await launcher.launch_model(launch_request)
print(f"job {job_id} — serving as {served_model_name}")
```

`launch_model` resolves the model path from the on-disk registry and picks the right environment `.toml` for the chosen framework. Prefer this path when you're launching a model that is in the SML catalog.

You can also build a `LaunchRequest` from a catalog entry directly:

```python
catalog = await launcher.get_preconfigured_models()
entry = next(e for e in catalog if e.model == "swiss-ai/Apertus-8B-Instruct-2509")

launch_request = LaunchRequest.from_catalog_entry(
    entry,
    workers=1,
    time="02:00:00",
)

job_id, served_model_name = await launcher.launch_model(launch_request)
```

### Option 2 — explicit launch (`launch_with_args`)

`launch_with_args` takes a `LaunchArgs` object where every SLURM and framework parameter is set by you. Use this when:

- The model isn't in the catalog.
- You need non-default framework flags (`--tp-size`, `--mem-fraction-static`, quantization, etc.).
- You want a fully declarative, reproducible script with no implicit defaults.

```python
from swiss_ai_model_launch import LaunchArgs

args = LaunchArgs(
    job_name=f"my_apertus_8b_{username}",
    served_model_name=f"swiss-ai/Apertus-8B-Instruct-2509-{username}",
    account=account,
    partition="normal",
    environment="src/swiss_ai_model_launch/assets/envs/sglang.toml",
    framework="sglang",
    framework_args=(
        "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 "
        f"--served-model-name swiss-ai/Apertus-8B-Instruct-2509-{username} "
        "--host 0.0.0.0 "
        "--port 8080 "
        "--tp-size 1"
    ),
    time="02:00:00",
    workers=1,
    nodes_per_worker=1,
)

job_id, served_model_name = await launcher.launch_with_args(args)
print(f"job {job_id} — serving as {served_model_name}")
```

Both methods return `(job_id: int, served_model_name: str)`. The `served_model_name` is the value you pass as the `model` field when sending inference requests to the cluster API endpoint.

## `LaunchArgs` reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `job_name` | `str` | required | SLURM job name |
| `served_model_name` | `str` | required | Name under which the model is served |
| `account` | `str` | required | SLURM account (billing group) |
| `partition` | `str` | required | SLURM partition |
| `environment` | `str` | required | Path to the environment `.toml` file |
| `framework` | `str` | required | Inference framework (`sglang` or `vllm`) |
| `framework_args` | `str` | `""` | Arguments forwarded verbatim to the framework |
| `workers` | `int` | `1` | Number of replicas |
| `nodes_per_worker` | `int` | `1` | Nodes per replica |
| `nodes` | `int \| None` | `workers × nodes_per_worker` | Total node count (computed if not set) |
| `time` | `str` | `"00:05:00"` | Job time limit (`HH:MM:SS`) |
| `reservation` | `str \| None` | `None` | SLURM reservation |
| `worker_port` | `int` | `5000` | Port used by worker replicas |
| `use_router` | `bool` | `False` | Enable load-balancing router across replicas |
| `router_args` | `str` | `""` | Arguments forwarded to the router |
| `pre_launch_cmds` | `str` | `""` | Shell commands to run before the framework starts |
| `disable_ocf` | `bool` | `False` | Disable OpenTela/OCF registration |
| `disable_metrics` | `bool` | `False` | Disable Prometheus metrics export |
| `disable_dcgm_exporter` | `bool` | `False` | Disable DCGM GPU metrics exporter |
| `telemetry_endpoint` | `str \| None` | `None` | Custom telemetry endpoint |

## Monitoring and managing jobs

All methods are async coroutines.

### Check job status

```python
from swiss_ai_model_launch import JobStatus

status = await launcher.get_job_status(job_id)

if status == JobStatus.RUNNING:
    print("model is up")
elif status == JobStatus.PENDING:
    print("still queued")
elif status == JobStatus.TIMEOUT:
    print("job timed out")
else:
    print("unknown state")
```

`JobStatus` is an enum with values: `PENDING`, `RUNNING`, `TIMEOUT`, `UNKNOWN`.

### Poll until running

```python
import asyncio

while True:
    status = await launcher.get_job_status(job_id)
    if status not in (JobStatus.PENDING, JobStatus.RUNNING):
        break
    if status == JobStatus.RUNNING:
        print("job is running")
        break
    await asyncio.sleep(10)
```

### Fetch logs

```python
stdout, stderr = await launcher.get_job_logs(job_id)
print(stdout)
```

### Cancel a job

```python
await launcher.cancel_job(job_id)
```

## Launching multiple models concurrently

The async API composes naturally with `asyncio.gather` to submit a batch of jobs in parallel without blocking:

```python
import asyncio
from swiss_ai_model_launch import LaunchArgs, SlurmLauncher

MODELS = [
    {"name": "Apertus-8B-Instruct-2509",  "extra_args": "--tp-size 1"},
    {"name": "Apertus-70B-Instruct-2509", "extra_args": "--tp-size 8", "nodes": 4},
]

async def submit(launcher: SlurmLauncher, model: dict) -> tuple[str, int]:
    name = model["name"]
    served = f"swiss-ai/{name}-{launcher.username}"
    args = LaunchArgs(
        job_name=f"sml_{name}_{launcher.username}",
        served_model_name=served,
        account=launcher.account,
        partition=launcher.partition,
        environment="src/swiss_ai_model_launch/assets/envs/sglang.toml",
        framework="sglang",
        framework_args=(
            f"--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/{name} "
            f"--served-model-name {served} "
            "--host 0.0.0.0 --port 8080 " + model.get("extra_args", "")
        ),
        nodes=model.get("nodes", 1),
        time="02:00:00",
    )
    job_id, served_name = await launcher.launch_with_args(args)
    return served_name, job_id

async def main() -> None:
    import getpass, grp, os
    username = getpass.getuser()
    account = grp.getgrgid(os.getgid()).gr_name
    launcher = SlurmLauncher(
        system_name="local", username=username, account=account, partition="normal"
    )
    results = await asyncio.gather(*(submit(launcher, m) for m in MODELS))
    for served_name, job_id in results:
        print(f"  {served_name} -> job {job_id}")

asyncio.run(main())
```

See [`examples/clariden/python/launch_multiple.py`](https://github.com/swiss-ai/model-launch/tree/main/examples/clariden/python/launch_multiple.py) for the full runnable version.

## More examples

Ready-to-run scripts for each cluster are in the [`examples/`](https://github.com/swiss-ai/model-launch/tree/main/examples) directory, organized by cluster and then by `cli/` or `python/`.

## Next

- [How to size a model](sizing.md) — picking replica count, nodes-per-replica, GPU type
- [Benchmarking](benchmarking.md) — measuring throughput once the model is up
- [Architecture](architecture.md) — how the serving stack fits together
