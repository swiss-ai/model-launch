# Porting a Raw sglang / vLLM Command

If you already have a `python3 -m sglang.launch_server ...` or `vllm serve ...` command that works, porting it to SML is mechanical: SML owns the SLURM job, the container, and the multi-node wiring — everything model-specific moves into `--framework-args` unchanged.

## The split

| Concern | Raw command | `sml advanced` |
| --- | --- | --- |
| Which framework | the entrypoint you type | `--framework sglang` / `vllm` |
| Container | `docker run`, `srun --environment` | `--environment <toml>` |
| Where it runs | `sbatch` / `srun` flags | `--system`, `--partition`, `--account`, `--reservation`, `--time` |
| Multi-node wiring | `--nnodes` / `--node-rank` / `--dist-init-addr`, or `ray start` | `--nodes-per-replica` |
| More copies | several jobs | `--replicas` (+ `--router sglang`) |
| Everything else | model path, `--tp-size`, parsers, … | `--framework-args "…"`, verbatim |

## 1. Move the framework flags in, unchanged

`--framework-args` is a single quoted string forwarded to the framework as typed. SML applies exactly two edits:

- prepends `--port 8080` (the fixed framework HTTP port), and
- rewrites `--served-model-name` to add your namespace.

## 2. Drop the flags SML sets

| Drop | Why |
| --- | --- |
| `--port` | Always injected as `8080` |
| `--dist-init-addr`, `--nnodes`, `--node-rank` (sglang, multi-node) | Rendered per rank from `--nodes-per-replica` |
| `ray start …`, `--distributed-executor-backend ray` (vLLM, multi-node) | The head bootstraps Ray and waits for all workers |

Keep `--host 0.0.0.0`. Genuine per-rank setup goes in `--pre-launch-cmds`, not in a wrapper shell script.

## 3. Translate the node layout

A node has **4 GPUs**, so a replica has `nodes-per-replica × 4`. Your parallelism must add up to that: `--tp-size` (sglang) or `--tensor-parallel-size` (vLLM). A 4-node replica means TP 16.

Need more throughput rather than more room? That's `--replicas N`, not N jobs. See [How to size a model](sizing.md).

## 4. Environment variables

`export`s you set before the raw command belong in the env toml's `[env]` block — they then apply inside the container on every rank. NCCL/libfabric tuning is already there in the shipped tomls; don't re-add it.

## 5. Name the model

Pass `--served-model-name` **inside `--framework-args`** so SML and the framework advertise the same id. SML prepends your cluster username (`<username>/swiss-ai/Apertus-8B-Instruct-2509`), and that namespaced id is what clients send in the `model` field. Write the name without a namespace and let SML add it — a name under someone else's namespace is rejected before submission.

## Example: sglang, one node

```bash
# Before
python3 -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
  --served-model-name swiss-ai/Apertus-8B-Instruct-2509 \
  --tp-size 4 --host 0.0.0.0 --port 30000 --enable-metrics
```

```bash
# After
sml advanced \
  --system clariden \
  --partition normal \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509 \
    --tp-size 4 \
    --host 0.0.0.0 \
    --enable-metrics"
```

Only `--port` disappeared; the rest moved across as-is.

## Example: vLLM, four nodes

```bash
# Before — head node
ray start --head --port=6379 --num-gpus=4
# ... on each of the other three nodes
ray start --address=<head-ip>:6379 --num-gpus=4
# ... then, back on the head
vllm serve --model /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 \
  --served-model-name deepseek-ai/DeepSeek-V3.1 \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --host 0.0.0.0 --port 8000
```

```bash
# After
sml advanced \
  --system clariden \
  --partition normal \
  --nodes-per-replica 4 \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 \
    --served-model-name deepseek-ai/DeepSeek-V3.1 \
    --tensor-parallel-size 16 \
    --host 0.0.0.0"
```

The whole Ray dance is gone — that's what `--nodes-per-replica 4` buys.

## Check it before submitting

Render the scripts without submitting and read the launch line:

```bash
sml advanced ... --output-script /tmp/check
grep -A3 launch_server /tmp/check/head.sh   # or: grep -A3 'vllm serve' /tmp/check/head.sh
```

The command in `head.sh` should be your original, plus `--port 8080` and the multi-node flags SML added. See [`--output-script`](usage-advanced.md#inspecting-what-would-be-submitted-output-script-dir).

## Checklist

- [ ] `--port` removed
- [ ] Multi-node flags / `ray start` removed
- [ ] TP × PP equals `nodes-per-replica × 4`
- [ ] `--served-model-name` inside `--framework-args`, un-namespaced
- [ ] `--environment` points at a toml whose image has your framework version
- [ ] `--time` covers cold start plus the workload
