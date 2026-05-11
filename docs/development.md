# Development

This page is for people working on SML itself. If you just want to use SML, see [Getting Started](getting-started.md).

## Setting up the dev environment

```bash
git clone https://github.com/swiss-ai/model-launch.git
cd model-launch
make install-dev
source .venv/bin/activate
```

`make install-dev` creates a virtualenv at `.venv/`, installs SML in editable mode, and sets up pre-commit hooks.

A handful of lint tools live outside the venv and need a one-time install:

| Tool | Why | Install (macOS) |
| --- | --- | --- |
| [`taplo`](https://taplo.tamasfe.dev/) | TOML formatter, used by `make format` / `make tomlfmt` and the pre-commit hook | `brew install taplo` |
| `npx` (Node) | Runs `prettier` and `markdownlint-cli2` on demand | `brew install node` |

Pin: CI installs `taplo` v0.9.3 — match it locally if you hit format-drift between your machine and CI.

## Test environment

Integration tests need real cluster credentials. Create `.test.sh` at the repo root:

```bash
export SML_CSCS_API_KEY=<your-api-key>
export SML_FIRECREST_CLIENT_ID=<your-client-id>
export SML_FIRECREST_CLIENT_SECRET=<your-client-secret>
export SML_FIRECREST_SYSTEM=clariden
export SML_FIRECREST_TOKEN_URI=<your-token-uri>
export SML_FIRECREST_URL=<your-firecrest-url>
export SML_PARTITION=normal
export SML_RESERVATION=<your-reservation>
```

`.test.sh` is gitignored; the test targets source it automatically.

## Common make targets

| Target                    | What it does                                  |
| ------------------------- | --------------------------------------------- |
| `make format`             | Format Python (`ruff`)                        |
| `make shellcheck`         | Lint shell scripts                            |
| `make markdownlint`       | Lint Markdown                                 |
| `make test-lightweight`   | Auto-CI subset of integration tests           |
| `make test-comprehensive` | Full integration test suite                   |
| `make clean-cache`        | Remove cache files                            |
| `make clean-dev`          | Remove the venv and cache                     |

## Debugging

Set `SML_DEBUG=1` to include local variables in crash tracebacks:

```bash
export SML_DEBUG=1
```

> **Warning:** `SML_DEBUG=1` may expose secrets (CSCS API key, FirecREST credentials) in crash output. Don't share terminal output captured with this flag.

By default, locals are stripped from crash reports.

## Adding a new model recipe

The lowest-friction contribution. Drop a shell script under `examples/<system>/cli/<vendor>/`. Use the [adding-new-model-to-sml issue template](https://github.com/swiss-ai/model-launch/blob/main/.github/ISSUE_TEMPLATE/adding-new-model-to-sml.md) as a checklist; existing scripts (e.g. `examples/clariden/cli/swiss-ai/Apertus-8B-Instruct-2509-sglang.sh`) are good templates.

For models that should appear in the `sml` interactive catalog (not just `sml advanced`), the recipe also needs an entry in the model catalog — see existing entries under `src/swiss_ai_model_launch/assets/models.json`.

### Try it yourself first

The SML team can't take a "please add my model" request for every checkpoint that lands on Hugging Face. Before filing an issue, work the checklist:

1. **Find the closest existing example** under `examples/<system>/cli/<vendor>/` — same framework (sglang/vllm), similar size class, same architecture if possible. Copy it.
2. **Swap in your model path** via `--framework-args "--model-path /capstor/store/.../<your-model>"` (and `--served-model-name <something-unique>`).
3. **Try it with [`sml advanced`](usage-advanced.md).** If it serves, you're done — the script *is* the recipe; PR it.
4. **If it doesn't serve, narrow the failure** before opening an issue:
    - Does the same model work with the framework directly (no SML)? If not, it's a framework issue, not an SML issue — report upstream.
    - Does it OOM? See [Sizing](sizing.md) — you may need bigger TP, more nodes, or quantization.
    - Does it fail to load? Architecture not supported by the framework version in the [environment toml](https://github.com/swiss-ai/model-launch/tree/main/src/swiss_ai_model_launch/assets/envs/) — try the other framework, or a newer image.
5. **Only if you've gotten through 1-4 and are still stuck**, file an issue with the failing command, the trailing 50 lines of logs, and what you've already ruled out.

## Modifying the SLURM submission script

The SLURM script is **rendered from Python at submit time** — there is no static `script.sh` or `template.jinja` to edit. The renderer is in [`src/swiss_ai_model_launch/launchers/framework.py`](https://github.com/swiss-ai/model-launch/blob/main/src/swiss_ai_model_launch/launchers/framework.py).

### What gets rendered

A single `master.sh` (visible via `--output-script` — see [usage](usage-advanced.md#inspecting-what-would-be-submitted---output-script)) containing in order:

1. **Telemetry** POST (omitted if no endpoint configured)
2. **Arch detection** — sets `OCF_BIN`, `SP_NCCL_SO_PATH`, `metrics_agent_bin` per `aarch64` / `x86_64`
3. **Node mapping** — `mapfile -t nodes < <(scontrol show hostnames ...)`
4. **Self-extracting rank scripts** — single-quoted `cat`-heredocs that lay down `head.sh`, optionally `follower.sh`, optionally `router.sh` under `$HOME/.sml/job-${SLURM_JOB_ID}/`
5. **Per-replica head IP discovery** — one `hostname -i` srun per replica
6. **Per-rank `srun` calls** — one block per (replica, rank). Each binds the rank dir into the pyxis container via `--container-mounts="$RANKS_DIR:$RANKS_DIR"` and invokes `bash $RANKS_DIR/<role>.sh`
7. **vmagent** (optional) — metrics scraper on the batch node
8. **Router** (optional) — `sglang_router` on `nodes[0]` when `replicas > 1 && --use-router`
9. **Footer** — connect/cancel hints, `wait`, "Script finished"

### Where to make changes

| If you want to change… | Edit… |
| --- | --- |
| What runs **inside** the container per rank | `_render_sglang_head`, `_render_sglang_follower`, `_render_vllm_head`, `_render_vllm_follower` |
| Framework env exports (NCCL flags, `no_proxy`, JIT DeepGEMM toggle, …) | `Sglang.env_exports` / `Vllm.env_exports` |
| Add a new inference framework | Subclass `Framework`, register in `_FRAMEWORKS`, write per-shape renderers |
| The OCF wrap | `_ocf_wrap` |
| The router rank script | `_render_router` |
| Arch detection / node mapping / vmagent / footer | The matching `_render_<section>` functions |
| What gets bind-mounted into the container per srun | The `--container-mounts` line in `_render_replica_launches` / `_render_router_launch` |
| The toml mount list itself (per env: sglang, vllm, …) | The files under `src/swiss_ai_model_launch/assets/envs/` |
| Total nodes / partition / time / SBATCH directives | `to_sbatch_args` on `LaunchArgs` (or `render_sbatch_header` for the firecrest path) |
| New CLI flag flowing into `LaunchArgs` | Add to `LaunchArgs` (pydantic), wire through `build_launch_args_from_advanced` in `cli/main.py` |

### Preview your change

```bash
sml advanced ... --output-script > /tmp/before.sh    # current behaviour
# edit framework.py
sml advanced ... --output-script > /tmp/after.sh     # new behaviour
diff /tmp/before.sh /tmp/after.sh
```

For full coverage, the test matrix at `tests/unit/test_framework.py` renders 96 configurations (framework × replicas × nodes_per_replica × use_router × disable_ocf × telemetry) and runs `bash -n` + `shellcheck` against each. If your change leaves any of those broken, the test will catch it before submit time:

```bash
uv run pytest tests/unit/test_framework.py -q
```

`tests/unit/test_examples.py` also renders six real example scripts through the production CLI parser, so adding a flag that breaks one of those will fail there.

## CI / CD

See [CI/CD](ci-cd.md) for the pipeline structure. PRs run static checks → image build → integration tests; each stage gates the next.

## Filing issues / PRs

- Bugs: use the [bug report template](https://github.com/swiss-ai/model-launch/blob/main/.github/ISSUE_TEMPLATE/bug-report.md). Include the failing command and the trailing chunk of TUI logs.
- New models: use the [adding-new-model template](https://github.com/swiss-ai/model-launch/blob/main/.github/ISSUE_TEMPLATE/adding-new-model-to-sml.md).
- PRs: keep them focused; pre-commit hooks must pass; integration tests must pass on at least one partition.
