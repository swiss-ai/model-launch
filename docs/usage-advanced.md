# Advanced Usage

`sml advanced` bypasses the model catalog and the interactive menu. You specify every launch parameter on the command line. Use it when:

- The model you want isn't in the curated catalog.
- You need to pass framework-specific flags (custom `--tp-size`, attention backend, quant config, …).
- You're scripting from CI and want a fully declarative invocation.

For the guided flow with a curated catalog, use [`sml`](usage-sml.md).

## Arguments

| Argument                    | Environment Variable    | Description                                                       |
| --------------------------- | ----------------------- | ----------------------------------------------------------------- |
| `--firecrest-system`        | `SML_FIRECREST_SYSTEM`  | Target HPC system                                                 |
| `--partition`               | `SML_PARTITION`         | SLURM partition                                                   |
| `--slurm-reservation`       | `SML_RESERVATION`       | SLURM reservation (optional)                                      |
| `--serving-framework`       |                         | Inference framework (`sglang`, `vllm`) — **required**             |
| `--slurm-environment`       |                         | Local path to the environment `.toml` file — **required**         |
| `--framework-args`          |                         | Arguments forwarded to the inference framework                    |
| `--slurm-nodes`             |                         | Total nodes (default: `replicas × nodes-per-replica`)             |
| `--slurm-replicas`          |                         | Number of replicas (default: `1`)                                 |
| `--slurm-nodes-per-replica` |                         | Nodes per replica (default: `1`)                                  |
| `--slurm-time`              |                         | Job time limit `HH:MM:SS` (default: `00:05:00`)                   |
| `--served-model-name`       |                         | Name under which the model is served (auto-generated if omitted)  |
| `--replica-port`            |                         | Port used by replicas (default: `5000`)                           |
| `--use-router`              |                         | Enable router to load-balance across replicas                     |
| `--router-args`             |                         | Arguments forwarded to the router                                 |
| `--disable-opentela`             |                         | Disable OCF/OpenTela wrapper                                               |
| `--pre-launch-cmds`         |                         | Shell commands to run before the framework starts                 |

## Example: Apertus 8B on Clariden with sglang

```bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-replicas 1 \
  --slurm-nodes-per-replica 1 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami) \
    --host 0.0.0.0 \
    --port 8080"
```

> **Note:** A model named `swiss-ai/Apertus-8B-Instruct-2509` is usually already running. The `--served-model-name` suffix avoids name collisions with shared deployments.

For more ready-to-run scripts per cluster and vendor, see [`examples/`](https://github.com/swiss-ai/model-launch/tree/main/examples).

## When to disable OCF/OpenTela

> "OCF" and "OpenTela" refer to the same thing — `OCF` is the on-disk binary name from the [OpenTela project](https://github.com/swiss-ai/opentela). The flag is `--disable-opentela`.

By default, every replica joins the OpenTela p2p mesh at startup. That registration is what makes the model resolvable through the public gateway at [serving.swissai.svc.cscs.ch](https://serving.swissai.svc.cscs.ch/). See [Architecture](architecture.md#disabling-opentela-registration-disable-opentela) for the longer story.

Pass `--disable-opentela` when:

- **You're benchmarking max throughput.** OpenTela adds a hop on the request path; disabling it gives you the framework's raw numbers. See [Benchmarking](benchmarking.md).
- **You want the model kept private.** With OpenTela disabled, the replica never registers with the mesh — so serving-api can't find it and it isn't reachable from outside the cluster. Useful for private fine-tunes or in-flight experiments.
- **You're running at scale and the mesh is in the way.** If you've stood up your own routing in front of N replicas (or you're driving load directly from another cluster job), OpenTela registration is just overhead.

If you disable it, you're responsible for reaching the model yourself — usually directly via its host:port from another job on the same cluster.

## Notes on flag style

- `sml advanced` takes system and partition as **arguments**, not env vars. This keeps each script reproducible without depending on shell state. (The interactive `sml` flow is different — see the [env-var tip](usage-sml.md#tip-env-vars-for-things-that-rarely-change) there.)
- `--framework-args` is a single quoted string forwarded verbatim to the framework. Keep it explicit; SML doesn't massage it.

## Next

- [How to size a model](sizing.md) — picking the right replica/node layout
- [Benchmarking](benchmarking.md) — throughput and latency measurement
- [Architecture](architecture.md) — how `sml advanced` fits with the serving stack
