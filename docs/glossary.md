# Glossary

One-line definitions for terms that show up in SML and the surrounding serving stack. Pages elsewhere link directly to the anchors here (e.g. `glossary.md#opentela`).

## Beverin

A CSCS HPC system; one of the [systems](#system) SML can target.

## Bristen

A CSCS HPC system, GPU-equipped; one of the [systems](#system) SML can target.

## Clariden

A CSCS HPC system, primarily GPU; one of the [systems](#system) SML can target. Most examples in this repo target Clariden.

## CSCS

The [Swiss National Supercomputing Centre](https://www.cscs.ch/), which operates the HPC clusters SML launches jobs on.

## DCGM

NVIDIA's [Data Center GPU Manager](https://developer.nvidia.com/dcgm). The DCGM exporter runs on each replica node and surfaces per-GPU metrics (SM utilization, memory bandwidth, NVLink, power) to the [telemetry endpoint](#telemetry-endpoint).

## FirecREST

A [REST API](https://eth-cscs.github.io/firecrest/) in front of SLURM, maintained by CSCS. Lets you submit and manage jobs without an interactive SSH session — SML uses it as one of two [launchers](#launcher).

## Framework

The inference engine that actually serves the model: [sglang](https://github.com/sgl-project/sglang) or [vLLM](https://github.com/vllm-project/vllm). Selected via `--serving-framework` in `sml advanced`. SML brings the framework up; the framework owns the request/response loop.

## Launcher

How SML submits jobs: `firecrest` (REST API, works from a laptop) or `slurm` (direct `sbatch`, works on a cluster login node). See [Initialization](initialization.md#firecrest-or-slurm).

## MCP

[Model Context Protocol](https://modelcontextprotocol.io/) — a standard for letting an LLM client (Claude Desktop, Cursor, …) call external tools. SML ships an MCP server so a client can list, launch, monitor, and cancel SML jobs as native tools. See [MCP Server](mcp.md).

## OCF

The OpenTela client wrapper that runs alongside the [framework](#framework) on each replica. Registers the replica with the [OpenTela](#opentela) p2p mesh under the served model name; pass `--disable-ocf` to skip it. See [Architecture → OCF](architecture.md#ocf-the-opentela-client-on-each-replica).

## OpenTela

The [p2p service mesh](https://github.com/swiss-ai/opentela) that connects models regardless of where they live (SLURM job, k8s pod, anywhere). The public gateway resolves model names through OpenTela and forwards requests to a registered peer. Default load-balancing across peers is random assignment.

## Partition

A SLURM concept — a named subset of cluster nodes with its own queue, time limit, and access policy. Set via `--partition`. Common values on Clariden: `normal`, `debug`.

## Replica

One independent copy of the model (a [DP](sizing.md#parallelism-dp--tp--pp--ep--and-why-dp-is-replicas) unit). Set via `--slurm-replicas`. More replicas = more throughput. Distinct from `--slurm-nodes-per-replica`, which sets how many nodes one replica spans.

## Reservation

A SLURM concept — a slot of nodes pre-allocated to a user/group, bypassing the normal queue. Set via `--slurm-reservation` (advanced) or `--reservation` (interactive). Optional.

## Router

A framework-side load balancer (e.g. `sglang-router`) inserted in front of N replicas inside one SLURM job. Enabled via `--use-router`. Orthogonal to OpenTela: the router shapes traffic *within* the job; OpenTela picks *which* job/peer a request lands on.

## Served-model name

The name a client uses to request the model from the public gateway (e.g. `swiss-ai/Apertus-8B-Instruct-2509-rosmith`). Set via `--served-model-name`. Auto-generated if omitted; the `-<user>` suffix avoids collisions with shared deployments.

## serving-api

[swiss-ai/serving-api](https://github.com/swiss-ai/serving-api) — the public-facing inference gateway at <https://serving.swissai.svc.cscs.ch/>. Resolves model names against [OpenTela](#opentela) and forwards requests to a registered peer.

## SLURM

The job scheduler used on most CSCS systems. SML serializes its launch into an `sbatch` script and submits it via either FirecREST or direct `sbatch`.

## sml

This CLI. Three subcommands: `init` (one-time credential setup), and two ways to launch — interactive (`sml`) or fully-flagged (`sml advanced`). See [Using SML](usage-sml.md).

## sml advanced

The all-flags entry point — point at any model, pass any framework args. Use for non-catalog models, custom framework config, or scripted CI launches. See [Advanced Usage](usage-advanced.md).

## System

The CSCS cluster a job targets — `clariden`, `beverin`, `bristen`, etc. Set via `--firecrest-system` or the `SML_FIRECREST_SYSTEM` env var.

## Telemetry endpoint

The URL metrics get pushed to (configured at `sml init` time). [DCGM exporter](#dcgm) and vmagent on each replica node push metrics here; Grafana reads from the same stack. Separate plane from [OpenTela](#opentela) (which carries request traffic, not metrics).

## TUI

The terminal UI SML opens after job submission — shows job state and live logs until the model is healthy.

## vmagent

A [VictoriaMetrics agent](https://docs.victoriametrics.com/vmagent.html) that scrapes Prometheus-format metrics (from the [framework](#framework) and from [DCGM](#dcgm)) and pushes them to the [telemetry endpoint](#telemetry-endpoint).
