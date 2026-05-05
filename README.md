# `sml`: Swiss AI Model Launch

<p align="center"><img src="docs/assets/logo-wide.png" alt="SML Logo" width="220"></p>

<p align="center">
  <a href="https://github.com/swiss-ai/model-launch/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/swiss-ai/model-launch/ci.yml?branch=main&label=CI"></a>
  <a href="https://swiss-ai.github.io/model-launch/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/swiss-ai/model-launch/docs.yml?branch=main&label=docs"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=swiss-ai_model-launch"><img alt="Quality Gate" src="https://sonarcloud.io/api/project_badges/measure?project=swiss-ai_model-launch&metric=alert_status"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=swiss-ai_model-launch"><img alt="Coverage" src="https://sonarcloud.io/api/project_badges/measure?project=swiss-ai_model-launch&metric=coverage"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.10-blue">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
</p>

<p align="center"><strong>Easy to launch LLM models 🚀</strong></p>

A CLI for launching LLMs on HPC clusters via SLURM directly or through FirecREST, and making them accessible via [OpenTela](https://github.com/eth-easl/OpenTela). Public serving endpoint: <https://serving.swissai.svc.cscs.ch/>.

## Quickstart

```bash
pip install git+https://github.com/swiss-ai/model-launch.git
sml init
sml
```

That's it — the second command `sml init` sets up credentials, the third launches a model interactively.

Prefer a script you can copy? Browse [`examples/`](examples/) and run any of them after `pip install`.

## Documentation

| Topic                                              | When to read                                     |
| -------------------------------------------------- | ------------------------------------------------ |
| [Getting Started](docs/getting-started.md)         | First time here                                  |
| [Initialization](docs/initialization.md)           | Setting up credentials, FirecREST vs SLURM       |
| [Using SML](docs/usage-sml.md)                     | Day-to-day launches via the interactive CLI      |
| [Advanced Usage](docs/usage-advanced.md)           | Full SLURM/framework control                     |
| [How to Size a Model](docs/sizing.md)              | Picking replica/node layout for a given model    |
| [Benchmarking](docs/benchmarking.md)               | Measuring throughput and latency                 |
| [MCP Server](docs/mcp.md)                          | Driving SML from Claude Desktop / Cursor         |
| [Architecture](docs/architecture.md)               | How SML fits with serving-api and [OpenTela](https://github.com/eth-easl/opentela)       |
| [Development](docs/development.md)                 | Contributing to SML itself                       |
| [CI/CD](docs/ci-cd.md)                             | Pipeline structure                               |
| [FAQ](docs/faq.md)                                 | Always-on hosting, common gotchas                |

A rendered docs site is built from the same files via MkDocs — run `make docs` for a local preview, or browse the published site at <https://swiss-ai.github.io/model-launch/>.

<p align="center"><img src="docs/assets/launch-apertus.gif" alt="Launching Apertus-8B with sml" width="800"></p>

<!-- Source: tapes/launch-apertus.tape — regenerate with `make demo`. -->

## License

[Apache 2.0](LICENSE).
