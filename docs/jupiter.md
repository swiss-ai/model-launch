# Running on JUPITER (JSC)

[JUPITER](https://www.fz-juelich.de/en/ias/jsc/jupiter) at the Jülich Supercomputing Centre has 4× GH200 (aarch64) per node, like Clariden, but it isn't a CSCS system. It has no FirecREST, no CSCS container plugin (EDF), no stock pyxis, no `/capstor`, and **compute nodes have no internet access**. sml runs there with the **slurm launcher** from a login node, the **apptainer** container runtime, and **OpenTela and metrics disabled**.

> Only `sml advanced` is JUPITER-aware for now. Verified end to end with the Qwen3-0.6B example on one node (vLLM served `/v1/chat/completions` on a GH200 node). Multi-node runs and OpenTela are not covered yet.

## Site rules

| Rule                                                                 | How sml handles it                                              |
| -------------------------------------------------------------------- | --------------------------------------------------------------- |
| Compute nodes have no internet                                       | Weights and image must be on disk first; `HF_HUB_OFFLINE=1`     |
| `/p/...` is not visible from compute nodes; use `/e/...`             | `vllm_jupiter.toml` mounts `/e`                                 |
| Containers need the JuDoor `container` group (apptainer)             | Sign the Container Runtime Engine agreement in JuDoor           |
| Project file-count limits block everyone when crossed                | Keep the image and weights as few large files                   |
| Needs a budget account and `--gres=gpu:4`                            | `export SML_ACCOUNT=<account>`, `--sbatch-arg=--gres=gpu:4`     |

## One-time setup

On a login node (`ssh <user>@login01.jupiter.fz-juelich.de -4`, 2FA required):

```bash
git clone https://github.com/swiss-ai/model-launch && cd model-launch
# The system Python is 3.9; sml needs >= 3.12. Get one with uv, for example:
python3 -m venv ~/uvboot && ~/uvboot/bin/pip install uv
~/uvboot/bin/uv venv --python 3.13 .venv && source .venv/bin/activate
~/uvboot/bin/uv pip install -e . huggingface_hub
sml init   # choose the "slurm" launcher

# Image (needs the `container` group; login nodes have internet)
mkdir -p /e/home/jusers/$USER/jupiter/sml/images
apptainer pull /e/home/jusers/$USER/jupiter/sml/images/vllm-v0.32.0-arm64.sif docker://vllm/vllm-openai:v0.32.0

# Weights (compute nodes cannot download them)
hf download Qwen/Qwen3-0.6B \
  --local-dir /e/home/jusers/$USER/jupiter/sml/models/Qwen/Qwen3-0.6B
```

## Launching

```bash
export SML_ACCOUNT=<budget account>   # see `jutil user projects -u $USER`
export SML_PARTITION=booster
./examples/jupiter/cli/qwen/Qwen3-0.6B-vllm.sh
```

| Flag                                   | Why                                                       |
| -------------------------------------- | --------------------------------------------------------- |
| `--environment …/vllm_jupiter.toml`    | Image on `/e`, `/e` mounted, offline Hub                  |
| `--container-runtime apptainer`        | `srun … apptainer exec --nv` instead of `--environment`   |
| `--sbatch-arg=--gres=gpu:4`            | GPUs are only visible when requested                      |
| `--disable-opentela --disable-metrics` | Their binaries and endpoints live on CSCS, unreachable    |

## Reaching the model

Without OpenTela there is no gateway; talk to the node directly on port **8080**, from a login node or through a tunnel:

```bash
squeue --me
curl http://<node>:8080/v1/models
ssh -N -L 8080:<node>:8080 <user>@login01.jupiter.fz-juelich.de -4
```

## How it works

`--container-runtime apptainer` reads the same env toml as `pyxis`, and puts the container in the command instead of the srun flags:

| EDF toml key | apptainer                                                       |
| ------------ | --------------------------------------------------------------- |
| `image`      | `apptainer exec … "$SML_IMAGE"` (`{arch}` resolved on the node) |
| `mounts`     | `--bind` (bare `/path` becomes `/path:/path`)                   |
| `workdir`    | `--pwd`                                                         |
| `[env]`      | exported on the batch node; apptainer passes the host env in    |
