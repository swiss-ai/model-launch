# FAQ

## I want to keep a model running 24/7 — can SML do that?

No. SML submits SLURM jobs, which are bounded by the partition's time limit. For an always-on serving deployment, the right home is Kubernetes — get in touch with the SwissAI infrastructure team to be onboarded.

If your need is "running for several hours unattended", that's fine — pick the time limit accordingly with `--time` (interactive `sml`) or `--slurm-time` (`sml advanced`).
There is an [open issue](https://github.com/swiss-ai/model-launch/issues/63) to allow sml to keep starting a job continuously.

## Should I use FirecREST or SLURM?

See the [decision table in Initialization](initialization.md#firecrest-or-slurm). Short version: FirecREST if you're launching from your laptop, SLURM if you're already SSH'd into the cluster.

## How many replicas / nodes-per-replica should I pick?

See [How to size a model](sizing.md). The short answer is "enough VRAM to fit weights + KV cache + headroom" — `sizing.md` walks through the math.

## Does SML do load balancing?

Yes — by default OCF/OpenTela does random assignment among peers. There is a PR in progress which can change this to different assignment modes: <https://github.com/swiss-ai/OpenTela/pull/4>

In SML, you can pass `--use-router` to put a router in front of N replicas. Without it, you get N independent endpoints with no traffic shaping; this works both with and without OpenTela. OpenTela gives you external access via <https://serving.swissai.svc.cscs.ch> and the API. Without OpenTela (using `--disable-ocf`), the model will not appear there and must be accessed directly from the cluster.

## My job is stuck in `PENDING`

Almost always a SLURM scheduling issue, not an SML one. Common causes:

- Partition is full or reserved.
- Time limit exceeds the partition's max.
- Reservation name is wrong (the job will silently sit pending).

Check via `squeue` on the cluster, or use the MCP tool / TUI status panel.

## Can I bring my own model that isn't in the catalog?

Yes — use [`sml advanced`](usage-advanced.md) and pass the model's path on the cluster filesystem via `--framework-args "--model-path /capstor/store/.../my-model"`. Or use the HF <org>/<model-name> to make framework download it on start. If the model is gated you will need a key.

## How do I keep a model private (not publicly routable)?

Pass `--disable-ocf` to `sml advanced`. By default each replica registers itself on the OpenTela p2p mesh — that registration is what the public gateway at [serving.swissai.svc.cscs.ch](https://serving.swissai.svc.cscs.ch/) routes through. Disabling it means the replica never joins the mesh, so the model is only reachable from inside the cluster. See [When to disable OCF](usage-advanced.md#when-to-disable-ocf).

> "OCF" and "OpenTela" are the same thing — `OCF` is the on-disk binary name; `OpenTela` is the project. The flag is named `--disable-ocf` for historical reasons.

## How do I see metrics?

Aggregated metrics land in Grafana — see [Benchmarking](benchmarking.md) for dashboard pointers.
Additional metrics are available from several sources:

- Framework metrics from vLLM/SGLang can be gathered with `--enable-metrics`. This is enabled by default for vLLM; for SGLang, it must be enabled with the flag.
- We use an agent called vmagent to gather these metrics and send them to Prometheus, where they can be displayed in a Grafana dashboard.
- Hardware-counter metrics can also be collected with NVIDIA's DCGM.

## How do I contribute a new model recipe?

Add an entry under `examples/<system>/cli/<vendor>/`. See [Development](development.md) for the contribution flow and the [adding-new-model issue template](https://github.com/swiss-ai/model-launch/blob/main/.github/ISSUE_TEMPLATE/adding-new-model-to-sml.md).

## Where do I report bugs?

[GitHub Issues](https://github.com/swiss-ai/model-launch/issues). Use the bug-report template and include the failing command + the trailing chunk of the TUI logs.

## What's the difference between `sml` and `sml advanced`?

`sml` is the curated/interactive entry point — pick from a catalog of vetted model+framework combos. `sml advanced` is the all-flags entry point — point at any model, pass any framework args. The two share the same SLURM machinery underneath.
