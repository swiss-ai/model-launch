# Getting Started

This page is a router. Pick the goal that matches what you're trying to do — each row points at the page that explains the *how*.

## Install first

Requires Python 3.10 through 3.14.

```bash
pip install git+https://github.com/swiss-ai/model-launch.git
sml --version
```

Then pick your goal below. (Contributing to SML itself? Skip the install above and see [Development](development.md) for the editable-install flow.)

## What do you want to do?

| Goal | Where to go |
| --- | --- |
| **Try a model — vibe-check responses, see what it sounds like** | Run an [example script](https://github.com/swiss-ai/model-launch/tree/main/examples) for a 1-shot launch, or use [`sml`](usage-sml.md) for the interactive menu. Both give you a live model in one command. |
| **Run a model with low latency** (chat, interactive demos) | [Sizing → Latency tuning](sizing.md#latency-tuning). Short version: smaller model, FP8/INT4 if quality allows, batch-1, no router. |
| **Run a model at high throughput** (batch eval, dataset processing) | [Sizing → Throughput tuning](sizing.md#throughput-tuning) for the layout, [Benchmarking](benchmarking.md) for measuring it. |
| **Keep the model private — only I can reach it** | Pass `--disable-opentela` so the replica never registers with the public gateway. See [When to disable OpenTela](usage-advanced.md#when-to-disable-opentela). |
| **Run a model that isn't in the catalog** | Use [`sml advanced`](usage-advanced.md) and point at the model's path on the cluster filesystem. **Try this yourself first** — see [Adding a new model recipe](development.md#adding-a-new-model-recipe). The SML team can't take a custom request for every model. |
| **Keep a model running 24/7** | SML can't — SLURM jobs are time-limited. You want Kubernetes. See the [24/7 hosting answer](faq.md#i-want-to-keep-a-model-running-247-can-sml-do-that) for who to contact. |
| **Drive SML from Claude Desktop / Cursor** | [MCP Server](mcp.md) — wire up the JSON config snippet and you get launch/monitor/cancel as native tools. |
| **Set up credentials for the first time** | [Initialization](initialization.md). Pick FirecREST (laptop) or SLURM (already on the cluster). |

## Got a question, not a goal?

If you have a specific operational question — *"why is my job stuck pending?"*, *"where do metrics live?"*, *"what's the difference between `sml` and `sml advanced`?"* — start with the [FAQ](faq.md). Unfamiliar word? See the [Glossary](glossary.md).
