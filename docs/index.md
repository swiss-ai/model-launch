# Swiss AI Model Launch

<p align="center"><img src="assets/logo-wide.png" alt="SML Logo" width="220"></p>

<p align="center"><strong>Make it easy to launch models 🚀</strong></p>

A CLI for launching LLMs on HPC clusters via SLURM or FirecREST. Public serving endpoint: <https://serving.swissai.svc.cscs.ch/>.

## Quickstart

```bash
pip install git+https://github.com/swiss-ai/model-launch.git
sml init
sml
```

That's it — the second command `sml init` sets up credentials, the third launches a model interactively.

## Where to start

- New here? → [Getting Started](getting-started.md)
- Setting up credentials? → [Initialization](initialization.md)
- Just want a script to run? → browse [`examples/`](https://github.com/swiss-ai/model-launch/tree/main/examples) on GitHub
- Sizing questions? → [How to size a model](sizing.md)
- Hooking up Claude Desktop? → [MCP Server](mcp.md)
- Always-on hosting / general questions? → [FAQ](faq.md)

## What SML is and isn't

SML is a thin orchestrator that submits SLURM jobs to bring up sglang or vLLM with the right model and arguments. It hands you back a live TUI of logs and job state until the model is healthy.

It is **not** a model server itself, **not** a long-running deployment manager (use Kubernetes for always-on serving — see [FAQ](faq.md#i-want-to-keep-a-model-running-247--can-sml-do-that)), and **not** a public traffic gateway (that's [serving-api](https://github.com/swiss-ai/serving-api)).

See [Architecture](architecture.md) for how the pieces fit together.
