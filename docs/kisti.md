# Running on KISTI

[KISTI-6](https://www.ksc.re.kr/) has 4× GH200 (aarch64) per node, like Clariden, but it isn't a CSCS system. It has no FirecREST, no CSCS container plugin (EDF), no `/capstor`, and no route to the OpenTela mesh. sml runs there with the **slurm launcher** from a login node, **stock pyxis** for containers, and **OpenTela and metrics disabled**.

> Only `sml advanced` is KISTI-aware for now. The interactive `sml` flow still assumes CSCS (model paths, OpenTela, no `--comment`) and gets refused.

## Site rules

KISTI's submit filter (`cli_filter`) refuses jobs that break either of these rules:

| Rule                                                                   | How sml handles it                             |
| ---------------------------------------------------------------------- | ---------------------------------------------- |
| Jobs must run from `/scratch` or `/appsdata`, not `$HOME`              | `SML_WORKING_DIR=/scratch/$USER/sml`           |
| Every job carries `--comment="field=<field>;appl=<appl>"` (`showappl`) | `--sbatch-arg="--comment=field=nlp;appl=vllm"` |

Two more rules don't trigger a refusal, but the job won't work without them:

- **GPUs are only visible if requested:** the example passes `--sbatch-arg=--gres=gpu:4`.
- **The driver is 570 (CUDA ≤ 12.8):** use `*-cu129` images (they run via minor-version compatibility), not the default CUDA 13 builds.

## One-time setup

On a login node (`ssh <user>@hangang.ksc.re.kr`). The system `python3` is 3.6, so use `cray-python`:

```bash
cd /scratch/$USER
git clone https://github.com/swiss-ai/model-launch && cd model-launch
module load cray-python/3.12.12
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
sml init   # choose the "slurm" launcher; the API key is only used by the TUI health check

# Import the vLLM image once (~4 min). The tag must match vllm_kisti.toml.
mkdir -p /scratch/$USER/sml/images
enroot import -o /scratch/$USER/sml/images/vllm-v0.32.0-cu129-arm64.sqsh \
  docker://vllm/vllm-openai:v0.32.0-cu129
```

## Launching

```bash
cd /scratch/$USER/model-launch && module load cray-python/3.12.12 && source .venv/bin/activate
export SML_ACCOUNT=<your project>               # e.g. in0138
export SML_WORKING_DIR=/scratch/$USER/sml
./examples/kisti/cli/qwen/Qwen3-0.6B-vllm.sh    # partition intTest; override with SML_PARTITION=normal
```

The example is an ordinary `sml advanced` call. To serve another model, copy it and change `--framework-args`. Models are pulled from Hugging Face at job start (compute nodes have internet) into `HF_HOME=/scratch/$USER/sml/hf`. The flags that make it KISTI-specific:

| Flag                                                           | Why                                                                        |
| -------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `--environment …/vllm_kisti.toml`                              | Image on `/scratch`, `/scratch` mounted, `HF_HOME` set                     |
| `--container-runtime pyxis`                                    | `srun --container-image` instead of CSCS `--environment`                   |
| `--sbatch-arg="--comment=field=nlp;appl=vllm"`                 | Required by `cli_filter`                                                   |
| `--sbatch-arg=--gres=gpu:4`                                    | GPUs are only visible when requested                                       |
| `--disable-opentela --disable-metrics`                         | Their binaries and endpoints live on CSCS                                  |
| `--pre-launch-cmds "pip uninstall -y -q torchcodec \|\| true"` | `vllm-openai:*-cu129` ships a CUDA 13 `torchcodec` that crashes the import |

`SML_SBATCH_ARGS` and `SML_CONTAINER_RUNTIME` can replace the two `--sbatch-arg`s and `--container-runtime` if you'd rather set them once in your shell.

## Reaching the model

Without OpenTela there is no gateway, so you talk to the node directly on port **8080**:

```bash
squeue --me                                   # NODELIST, e.g. node1895
curl http://node1895:8080/v1/models
curl http://node1895:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<user>/Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hi"}]}'
```

The served name is namespaced with your username (see [served model names](usage-advanced.md#arguments)). From your laptop, tunnel through the login node:

```bash
ssh -N -L 8080:node1895:8080 <user>@hangang.ksc.re.kr   # then use localhost:8080
```

## Logs

Everything lives under `$SML_WORKING_DIR/logs/<jobid>/`:

- `log.out` / `log.err`: the batch script (node list, replica URLs)
- `replica_<r>.out` / `.err`: the framework itself
- `replica_health.json`: the in-job health checker's last report

Cancel with `scancel <jobid>`.

## How it works

`--container-runtime pyxis` makes sml read the env toml where the job is rendered (so it must be a local path) and translate it for stock pyxis:

| EDF toml key  | pyxis                                                          |
| ------------- | -------------------------------------------------------------- |
| `image`       | `--container-image` (`{arch}` resolved on the node)            |
| `mounts`      | `--container-mounts` (bare `/path` becomes `/path:/path`)      |
| `workdir`     | `--container-workdir`                                          |
| `[env]`       | exported on the batch node; srun carries it into the container |
| `annotations` | ignored (CSCS hooks)                                           |

Values are expanded by bash on the batch node, so `$USER` works. The same mechanism should carry over to other clusters with stock pyxis (e.g. JSC).
