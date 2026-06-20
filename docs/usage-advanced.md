# Advanced Usage

`sml advanced` bypasses the model catalog and the interactive menu. You specify every launch parameter on the command line. Use it when:

- The model you want isn't in the curated catalog.
- You need to pass framework-specific flags (custom `--tp-size`, attention backend, quant config, …).
- You're scripting from CI and want a fully declarative invocation.

For the guided flow with a curated catalog, use [`sml`](usage-sml.md).

## Arguments

| Argument                    | Environment Variable   | Description                                                                                                                    |
| --------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--firecrest-system`        | `SML_FIRECREST_SYSTEM` | Target HPC system                                                                                                              |
| `--partition`               | `SML_PARTITION`        | SLURM partition                                                                                                                |
| `--slurm-account`           | `SML_ACCOUNT`          | SLURM account used for job submission                                                                                          |
| `--slurm-reservation`       | `SML_RESERVATION`      | SLURM reservation (optional)                                                                                                   |
| `--serving-framework`       |                        | Inference framework (`sglang`, `vllm`) — **required**                                                                          |
| `--slurm-environment`       |                        | Local path to the environment `.toml` file — **required**                                                                      |
| `--framework-args`          |                        | Arguments forwarded to the inference framework                                                                                 |
| `--slurm-replicas`          |                        | Number of replicas (default: `1`)                                                                                              |
| `--slurm-nodes-per-replica` |                        | Nodes per replica (default: `1`)                                                                                               |
| `--time`                    |                        | Total uptime `HH:MM:SS` (default: `02:00:00`)                                                                                  |
| `--consecutive`             |                        | Serve a `--time` longer than the per-job cap with a chain of jobs                                                              |
| `--handover-time`           |                        | Overlap before the previous job ends (default: `03:00:00`)                                                                     |
| `--max-job-time`            |                        | Per-job cap for chains `HH:MM:SS` (default: `12:00:00`)                                                                        |
| `--served-model-name`       |                        | Required: pass it here, or include `--served-model-name <name>` inside `--framework-args`. Omitting both aborts with an error. |
| `--router`                  |                        | Routing: `OCF` (default) or `SGL` (in-job router, replicas > 1)                                                                |
| `--router-args`             |                        | Arguments forwarded to the router (`--router SGL`)                                                                             |
| `--disable-ocf`             |                        | Disable OCF wrapper                                                                                                            |
| `--otela-bootstrap-addr`    |                        | Override the OCF/OpenTela bootstrap peer (full multiaddr)                                                                      |
| `--dev`                     |                        | Shorthand for the dev OCF/OpenTela bootstrap peer                                                                              |
| `--disable-metrics`         |                        | Disable vmagent metrics push                                                                                                   |
| `--disable-dcgm-exporter`   |                        | Disable DCGM GPU metrics exporter                                                                                              |
| `--pre-launch-cmds`         |                        | Shell commands to run before the framework starts                                                                              |
| `--output-script DIR`       |                        | Render master.sh + rank scripts into DIR and exit (no submit)                                                                  |

> Total nodes is `--slurm-replicas × --slurm-nodes-per-replica`. The framework HTTP port is **8080**.

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

## Running past the 12 h cap (`--consecutive`)

SLURM caps a single job at 12 h. `--time` is the **total** time you want the
model up; when it exceeds the per-job cap (`--max-job-time`, default `12:00:00`),
pass `--consecutive` to serve it with a pre-scheduled **chain of jobs**:

```bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --time 36:00:00 \
  --consecutive \
  --framework-args "--model-path /capstor/.../Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami) \
    --host 0.0.0.0 --enable-metrics"
```

How it works:

- **All jobs are submitted up front.** The job count is the minimum whose
  continuous coverage reaches `--time`, spaced `(--max-job-time − --handover-time)`
  apart. With the defaults a `36:00:00` request becomes **4 jobs** spaced 9 h apart.
- **Anchored to actual start, not a guessed clock.** The head job gets a single
  absolute SLURM `--begin` (≈ now); every successor is submitted with a SLURM
  `--dependency=after:<prev>+<interval>` so it starts that interval after its
  predecessor *actually begins running*. If the head (or any job) sits in the
  queue waiting for resources, the rest of the chain slides with it — the
  handover overlap stays correct instead of firing against stale wall-clock times.
- **Handover overlap.** A job starts `--handover-time` (default `03:00:00`)
  before its predecessor's limit, giving the fresh job time to load weights and
  become healthy before the old one expires.
- **Self-cancelling.** Each job is stamped with its predecessor's id and cancels
  it from inside the job once all of its own replicas are healthy — so the old
  allocation is released and no resources are wasted. This runs on the batch
  node, so it works even after you've detached the CLI.
- **One endpoint.** Every job in the chain shares the same `--served-model-name`,
  so clients see a single continuous model across handovers.
- **Exact total.** Every job but the last runs the full `--max-job-time`; the
  last job's time limit is trimmed so the chain ends exactly at `--time` instead
  of overshooting by up to a whole job. The `36:00:00` example above runs
  `12 h + 12 h + 12 h + 9 h`, ending right on 36 h rather than at 39 h.

In the TUI, a **Consecutive Job Chain** panel lists every job with a ▶ on the one
currently serving, its live status (`PENDING` → `RUNNING` → `CANCELLED` once
handed over), and its **scheduled start/end** fetched from the scheduler — actual
once running, the queue's estimate while pending. A job not yet scheduled shows
its dependency instead (e.g. `15m after 2569107 starts`). During an overlapping
handover the replica-health panel shows **each running job's** replicas in its
own labelled section; once a job stops reporting (ended or cancelled) its
replicas are marked `STALE` rather than left showing a frozen `HEALTHY`.

> If `--time` is within the per-job cap, `--consecutive` is a no-op (single job).
> For a quick end-to-end test, shrink `--max-job-time` (e.g.
> `--max-job-time 00:10:00 --handover-time 00:03:00`) so handovers fire in
> minutes.
>
> Dependency anchoring keeps the chain correct under queue delay, but it can't
> conjure capacity: if a successor still hasn't been allocated by the time its
> predecessor hits the 12 h wall, the model is briefly down until it starts.
> Raise `--handover-time` to widen the overlap if your partition is contended.

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

> **`master.sh` is self-contained.** Rank scripts are embedded as `cat`-heredocs and extracted at job start to `$HOME/.sml/job-${SLURM_JOB_ID}/` — shared FS, so every compute node `srun` reaches can read them. The sibling `head.sh` / `follower.sh` / `router.sh` from `--output-script` are inspection-only and never read at runtime; to hand-tune, edit the heredoc bodies inside `master.sh`.

## When to disable OCF

> "OCF" and "OpenTela" refer to the same thing — `OCF` is the legacy name for the [OpenTela project](https://github.com/swiss-ai/opentela)'s client binary (shipped on-disk as `otela-<arch>`, referenced via `OCF_BIN`). The flag is `--disable-ocf` for historical reasons.

By default, every replica joins the OpenTela p2p mesh at startup. That registration is what makes the model resolvable through the public gateway at [serving.swissai.svc.cscs.ch](https://serving.swissai.svc.cscs.ch/). See [Architecture](architecture.md#disabling-opentela-registration-disable-ocf) for the longer story.

Pass `--disable-ocf` when:

- **You're benchmarking max throughput.** OpenTela adds a hop on the request path; disabling it gives you the framework's raw numbers. See [Benchmarking](benchmarking.md).
- **You want the model kept private.** With OpenTela disabled, the replica never registers with the mesh — so serving-api can't find it and it isn't reachable from outside the cluster. Useful for private fine-tunes or in-flight experiments.
- **You're running at scale and the mesh is in the way.** If you've stood up your own routing in front of N replicas (or you're driving load directly from another cluster job), OpenTela registration is just overhead.

If you disable it, you're responsible for reaching the model yourself — usually directly via its host:port from another job on the same cluster.

## Pointing at a different OCF bootstrap peer

The bootstrap multiaddr the replica uses to join the mesh is baked in — it's the prod peer by default. Two flags override it:

- `--dev` — switch to the dev-datacenter peer. Shorthand for the most common alternate environment.
- `--otela-bootstrap-addr <multiaddr>` — point at an arbitrary peer, e.g. an OCF instance running in another datacenter or on a custom IP. Takes precedence over `--dev` if both are passed (with a warning).

Example:

```bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "..." \
  --dev
```

Or for a custom peer:

```bash
sml advanced \
  ... \
  --otela-bootstrap-addr /ip4/10.0.0.42/tcp/43905/p2p/QmYourPeerId...
```

The chosen multiaddr is recorded under `ocf_bootstrap_addr` in the telemetry payload, so launches against different environments are distinguishable downstream.

## Notes on flag style

- For reproducibility we recommend passing system and partition as explicit **arguments** in `sml advanced` scripts rather than relying on shell state, though `SML_FIRECREST_SYSTEM` and `SML_PARTITION` are still honoured if the corresponding flag is omitted. (The interactive `sml` flow leans on these env vars more — see the [env-var tip](usage-sml.md#tip-env-vars-for-things-that-rarely-change) there.)
- `--framework-args` is a single quoted string forwarded verbatim to the framework. Keep it explicit; SML doesn't massage it.

## Next

- [How to size a model](sizing.md) — picking the right replica/node layout
- [Benchmarking](benchmarking.md) — throughput and latency measurement
- [Architecture](architecture.md) — how `sml advanced` fits with the serving stack
