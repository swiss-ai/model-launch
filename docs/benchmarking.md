# Benchmarking & Performance

This page is a living reference for measuring SML deployments. Contributions of methodology, scripts, or write-ups are welcome — see [Contributing a benchmark](#contributing-a-benchmark) at the bottom.

## What to measure

Always report at least:

| Metric                   | Why it matters                                         |
| ------------------------ | ------------------------------------------------------ |
| **TTFT** (time-to-first-token) | User-visible latency; dominated by prefill.       |
| **Tokens / sec / replica**     | Throughput ceiling per replica.                   |
| **Tokens / sec / GPU**         | Hardware efficiency; lets you compare layouts.    |
| **P50 / P95 / P99 latency**    | Tail behavior under load.                         |
| **Concurrent requests**        | What input rate were these numbers measured at?   |

A throughput number without the concurrency it was measured at is meaningless — always pair them.

## Best practices

- **Warm up first.** Discard the first ~30 seconds of measurements: weights cache, NCCL channels, and the KV cache all need to settle.
- **Use a realistic workload.** Synthetic 100-token-in / 100-token-out benchmarks rarely match production. Capture or replay a real prompt distribution.
- **Vary one thing at a time.** Replicas × precision × batch size × context length is a 4D space; sweep one axis with the others fixed.
- **Pin the framework version.** Both sglang and vLLM iterate fast — record exact image tag / git SHA in your write-up.
- **Match the partition's nodes.** Performance on `normal` vs. a debug partition can differ; benchmarks should target the partition users will use.
- **Disable OCF for raw numbers.** Pass `--disable-ocf` to skip the OpenTela mesh registration on each replica. You then drive load directly to the framework's host:port — no OpenTela hop, no serving-api routing — which gives you the framework's true throughput. Re-enable OCF for end-to-end numbers that include the mesh + gateway hop. See [When to disable OCF](usage-advanced.md#when-to-disable-ocf).

## Observability

- **Grafana** — aggregated dashboards for SML jobs are wired through the telemetry endpoint set at `sml init` time. Ask the SwissAI infra team for the dashboard URL.
- **DCGM exporter** — per-GPU metrics (SM utilization, memory bandwidth, NVLink, power). DCGM runs alongside the inference framework on each replica node; metrics are scraped to the same Grafana stack. Disable via `--disable-dcgm-exporter` if needed.

## Pre-canned methodology

> _More methodologies welcome — open a PR adding a section here._

(Placeholder — add a "How we measured X" subsection per benchmark you publish, with the exact `sml advanced` invocation, the load generator, and the resulting numbers.)

## Contributing a benchmark

If you've run a serious benchmark and want it preserved here:

1. Open a PR adding a new `## ...` section to this file (or, for big write-ups, a sibling file like `docs/benchmarks/<topic>.md` linked from here).
2. Include: model, framework version, GPU layout, exact `sml` invocation, load generator + workload, raw numbers, brief discussion.
3. If a chart helps, drop the source PNG / SVG under `docs/assets/` and link it.

The goal is a small, browseable library of "we tried X, got Y" so the next person doesn't redo the same experiment.

## Next

- [How to size a model](sizing.md) — pick the layout you're benchmarking
- [Architecture](architecture.md) — what's actually in the request path
