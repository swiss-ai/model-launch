# Adding a New Model

There are two levels of support, and they're independent:

| Level | What it means | Where it lives |
| --- | --- | --- |
| **Recipe** | A ready-to-run script anyone can launch with [`sml advanced`](usage-advanced.md) | `examples/<system>/cli/<vendor>/` |
| **Catalog entry** | The model appears in the interactive [`sml`](usage-sml.md) picker and the [MCP server](mcp.md) | `src/swiss_ai_model_launch/assets/models.json` |

A recipe is the lowest-friction contribution and the prerequisite for the other — get the model serving first, then decide whether it belongs in the catalog.

## Before you start

- **Weights on the cluster.** The model must be under `/capstor/store/cscs/swissai/infra01/hf_models/models/<vendor>/<model>`.
- **A layout that fits.** Work out GPUs, TP, and nodes per replica — see [How to size a model](sizing.md).
- **A framework that supports the architecture.** Check the framework version in the [env toml](https://github.com/swiss-ai/model-launch/tree/main/src/swiss_ai_model_launch/assets/envs/) you plan to use. If the architecture is too new, you may need a new image — see [Building Container Images](building-images.md).

## Step 1 — get it serving

Start from the closest existing example under `examples/<system>/cli/<vendor>/`: same framework, similar size class, same architecture if possible. Copy it, swap in your model path and `--served-model-name`, and run it.

Already have a working raw `sglang.launch_server` / `vllm serve` command? See [Porting a raw command](porting-commands.md) for the flag-by-flag mapping.

If it serves, you're done — the script **is** the recipe.

## Step 2 — commit the recipe

Name it `<Model>-<framework>.sh` and put it under the system it was tested on:

```text
examples/clariden/cli/swiss-ai/Apertus-8B-Instruct-2509-sglang.sh
examples/beverin/cli/swiss-ai/Apertus-8B-Instruct-2509-vllm-rocm.sh
```

Match the flag style of its neighbours — explicit `--system` / `--partition`, one `--framework-args` string. Pick the env toml that matches the cluster: `sglang.toml` / `vllm.toml` on clariden, the `*_rocm.toml` variants on AMD systems.

`tests/unit/test_examples.py` renders a selection of real examples through the production CLI parser and shellchecks the output, so a flag that breaks one of them fails there:

```bash
uv run pytest tests/unit/test_examples.py -q
```

## Step 3 — add a catalog entry (optional)

Only needed if the model should appear in interactive `sml`. Add an object to `models.json`:

```json
{
  "model": "MiniMaxAI/MiniMax-M2",
  "framework": "sglang",
  "environment": null,
  "nodes_per_replica": 2,
  "framework_args": "--tp-size 8 --ep-size 8 --tool-call-parser minimax-m2 --trust-remote-code --enable-metrics"
}
```

| Field | Meaning |
| --- | --- |
| `model` | HF repo id; resolved to `<registry>/<vendor>/<model>` |
| `framework` | `sglang` or `vllm` |
| `environment` | Path to an env toml, or `null` for the framework default (`sglang.toml` / `vllm.toml`) |
| `nodes_per_replica` | Nodes one replica spans |
| `framework_args` | **Extra** flags only — see below |
| `pre_launch_cmds` | Optional shell commands before the framework starts |
| `model_path` | Optional override when the weights aren't at the registry path |

> **`framework_args` here is not the same string as in a recipe.** For catalog launches SML builds `--model <path>`, `--served-model-name`, and `--host 0.0.0.0` itself, then appends `framework_args`. Repeating them causes duplicate flags — put only the extras (`--tp-size`, parsers, `--trust-remote-code`, …).

Entries carry no system field, so the env toml you reference has to work on whichever cluster the user launches from.

## When it doesn't serve

Narrow the failure before filing an issue:

- **Does the model work with the framework directly, no SML?** If not, it's a framework issue — report upstream.
- **Does it OOM?** See [Sizing](sizing.md) — bigger TP, more nodes, or quantization.
- **Does it fail to load?** The architecture may be unsupported by the framework version in the [env toml](https://github.com/swiss-ai/model-launch/tree/main/src/swiss_ai_model_launch/assets/envs/). Try the other framework, or a newer image ([Building Container Images](building-images.md)).
- **Not sure what SML submitted?** Render without submitting: `sml advanced ... --output-script /tmp/check` ([details](usage-advanced.md#inspecting-what-would-be-submitted-output-script-dir)).

## Filing it

New checkpoints appear on Hugging Face faster than the SML team can add them one by one, so please try the steps above first — most models work without any changes to SML. If you're still stuck, we're happy to help: open an issue using the [adding-new-model template](https://github.com/swiss-ai/model-launch/blob/main/.github/ISSUE_TEMPLATE/adding-new-model-to-sml.md) and include the failing command, the trailing 50 lines of logs, and what you've already ruled out. That context lets us get to an answer much faster.

For PRs: keep them focused, make sure pre-commit hooks pass, and expect [CI](ci-cd.md) to run static checks and integration tests. See [Development](development.md) for the dev environment.
