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
| `--slurm-replicas`          |                         | Number of replicas (default: `1`)                                 |
| `--slurm-nodes-per-replica` |                         | Nodes per replica (default: `1`)                                  |
| `--slurm-time`              |                         | Job time limit `HH:MM:SS` (default: `00:05:00`)                   |
| `--served-model-name`       |                         | Name under which the model is served (auto-generated if omitted)  |
| `--use-router`              |                         | Load-balance across replicas (needs `replicas > 1`)               |
| `--router-args`             |                         | Arguments forwarded to the router                                 |
| `--disable-ocf`             |                         | Disable OCF wrapper                                               |
| `--disable-metrics`         |                         | Disable vmagent metrics push                                      |
| `--disable-dcgm-exporter`   |                         | Disable DCGM GPU metrics exporter                                 |
| `--pre-launch-cmds`         |                         | Shell commands to run before the framework starts                 |
| `--output-script DIR`       |                         | Render master.sh + rank scripts into DIR and exit (no submit)     |

> Total nodes is automatically `--slurm-replicas × --slurm-nodes-per-replica`; there is no separate `--slurm-nodes` flag. The framework HTTP port is hardcoded to **8080** across every job — no `--worker-port`/`--replica-port` knob.

## Example: Apertus 8B on Clariden with sglang

```bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami) \
    --host 0.0.0.0 \
    --enable-metrics"
```

> **Note:** A model named `swiss-ai/Apertus-8B-Instruct-2509` is usually already running. The `--served-model-name` suffix avoids name collisions with shared deployments.

For more ready-to-run scripts per cluster and vendor, see [`examples/`](https://github.com/swiss-ai/model-launch/tree/main/examples).

## Inspecting what would be submitted (`--output-script DIR`)

`--output-script DIR` writes the rendered submission scripts into the given directory and exits without submitting:

```bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/.../Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami) \
    --host 0.0.0.0 --enable-metrics" \
  --output-script /tmp/debug
```

Produces something like:

```text
Wrote 2 file(s) to /tmp/debug:
  master.sh
  head.sh
```

For a multi-node / router config the directory also gets `follower.sh` and/or `router.sh`. The layout is **byte-identical** to what a live submission writes to `~/.sml/job-${SLURM_JOB_ID}/` at job start — so each rank shape is its own bash file:

```bash
shellcheck /tmp/debug/*.sh        # lint each independently
cat /tmp/debug/head.sh            # inspect just the head-rank logic
sbatch /tmp/debug/master.sh       # submit manually if you want
diff /tmp/debug/head.sh /tmp/older-debug/head.sh   # compare runs
```

Useful for:

- **Debugging a launch failure:** see exactly what `--framework-args` your invocation translated to (the `--port 8080` auto-injection, etc.), which `srun` calls would run, and what each rank shape does.
- **Reviewing changes during SML development:** render against a known invocation before and after a code change, diff the rank scripts.
- **Starting point for a hand-tuned job:** edit any of the rank scripts, then `sbatch master.sh` directly.

After a real (non-`--output-script`) submission, the same rank scripts also land on disk at `~/.sml/job-${SLURM_JOB_ID}/` for post-mortem inspection.

> **`master.sh` is self-contained.** It carries the rank scripts as embedded `cat`-heredocs and re-extracts them under `~/.sml/job-${SLURM_JOB_ID}/` at job start — that's why `sbatch master.sh` works regardless of where the directory lives. The standalone `head.sh` / `follower.sh` / `router.sh` files in the output dir are byte-equal duplicates for inspection; the heredoc bodies inside `master.sh` are what actually run.

## When to disable OCF

> "OCF" and "OpenTela" refer to the same thing — `OCF` is the on-disk binary name from the [OpenTela project](https://github.com/swiss-ai/opentela). The flag is `--disable-ocf` for historical reasons.

By default, every replica joins the OpenTela p2p mesh at startup. That registration is what makes the model resolvable through the public gateway at [serving.swissai.svc.cscs.ch](https://serving.swissai.svc.cscs.ch/). See [Architecture](architecture.md#disabling-opentela-registration-disable-ocf) for the longer story.

Pass `--disable-ocf` when:

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
