# Architecture

SML is a thin orchestrator. It doesn't serve models itself — it submits SLURM jobs that bring up an inference framework (sglang or vLLM) on cluster nodes, optionally fronted by a router for load balancing.

## Components

```
┌──────────┐    ┌──────────────┐    ┌─────────────────────┐
│  user    │ ─► │  sml CLI     │ ─► │  FirecREST / SLURM  │
│  / MCP   │    │  (this repo) │    │  job submission     │
└──────────┘    └──────────────┘    └──────────┬──────────┘
                                               │
                                  ┌────────────▼─────────────┐
                                  │   SLURM job (per launch) │
                                  │  ┌──────────────────────┐│
                                  │  │ router (optional)    ││
                                  │  └─────────┬────────────┘│
                                  │  ┌─────────▼────────────┐│
                                  │  │ N replicas           ││
                                  │  │  ┌─────┐ ┌──────────┐││
                                  │  │  │ OCF │─│ sglang / │││
                                  │  │  │     │ │ vLLM     │││
                                  │  │  └──┬──┘ └──────────┘││
                                  │  └─────┼────────────────┘│
                                  │  ┌─────┼────────────────┐│
                                  │  │ DCGM + vmagent       ││
                                  │  └──┬──┼────────────────┘│
                                  └─────┼──┼─────────────────┘
                                        │  │
                                        │  └──► OpenTela p2p mesh ◄── serving-api
                                        │                              (public gateway)
                                        │
                                        └──► telemetry endpoint ──► Grafana
```

Two independent planes leave the job:

- **Request plane** (right): each replica's OCF client registers it on the **OpenTela p2p mesh**. The serving-api gateway resolves model names through OpenTela and forwards requests to a registered peer.
- **Metrics plane** (bottom): DCGM and vmagent scrape per-GPU and per-process metrics and push them to the telemetry endpoint, which Grafana reads from. Separate system; not OpenTela.

## Repos in the serving stack

SML is one piece of a larger system. The siblings:

- **[swiss-ai/model-launch](https://github.com/swiss-ai/model-launch)** — this repo. The CLI and MCP server.
- **[swiss-ai/serving-api](https://github.com/swiss-ai/serving-api)** — the public-facing inference gateway at [serving.swissai.svc.cscs.ch](https://serving.swissai.svc.cscs.ch/). Resolves model names against OpenTela and forwards requests to a registered peer.
- **[swiss-ai/opentela](https://github.com/swiss-ai/opentela)** — the **p2p service mesh** that connects models regardless of where they live (SLURM job, Kubernetes pod, any network or location). Each model registers itself with OpenTela through a small client called **OCF** (the wrapper around the inference framework). By default OpenTela does **random assignment among peers** registered under the same model name — that's the load-balancing primitive. OpenTela is what makes a model launched here on Clariden interchangeable, from the gateway's perspective, with the same model running in a k8s deployment elsewhere.

## Request path (typical SML deployment)

1. User runs `sml advanced ...` (or interactive `sml`).
2. SML serializes launch args, builds an `sbatch` script, submits via FirecREST or directly via SLURM.
3. SLURM allocates nodes; the job script starts the inference framework on each replica, **wrapped in OCF**.
4. OCF announces each replica to the OpenTela p2p mesh under the served model name.
5. (Optional) `--use-router` puts a framework router (e.g. sglang-router) in front of the replicas inside the job. This is orthogonal to OpenTela — the router shapes traffic *within* the job; OpenTela picks *which* job/peer a request lands on.
6. DCGM exporter and vmagent start in sidecar fashion on each replica node, pushing metrics to the telemetry endpoint.
7. A user request hits serving-api → serving-api looks up the model name in OpenTela → OpenTela picks a registered peer → request flows to that replica's OCF, then to the framework process.

## OCF: the OpenTela client on each replica

OCF is the small wrapper that makes a replica part of the OpenTela mesh. Concretely it's a binary at `/ocfbin/ocf-{arm,amd64}` started on rank 0 of each replica, which contacts an OpenTela bootstrap peer, registers the local framework process under the served model name, and proxies inbound requests to it.

Pass `--disable-ocf` to skip OCF. The framework still runs and serves on its replica port inside the cluster, but it never registers with OpenTela — so:

- It is **not reachable through [serving-api](https://github.com/swiss-ai/serving-api)** at [serving.swissai.svc.cscs.ch](https://serving.swissai.svc.cscs.ch/).
- It is only reachable directly via host:port from another job on the same cluster.

Use `--disable-ocf` for private models, raw-throughput benchmarks (no OpenTela hop), or when you've stood up your own routing in front of the replicas. See [usage-advanced.md](usage-advanced.md#when-to-disable-ocf).

## Where SML's responsibility ends

SML's job is "get the framework process running on the right nodes with the right args, and stream you the logs until it's healthy." It does not:

- Persist the deployment past the SLURM time limit (use k8s for that — see [FAQ](faq.md#i-want-to-keep-a-model-running-247--can-sml-do-that)).
- Route public traffic (that's serving-api + OpenTela).

This separation keeps SML small enough that a single user can read the whole codebase in an afternoon.

## Next

- [How to size a model](sizing.md) — picking the layout the architecture above will materialize
- [MCP](mcp.md) — driving the same orchestrator from an LLM client
