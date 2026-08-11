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

| Tool                                  | Why                                                                            | Install (macOS)      |
| ------------------------------------- | ------------------------------------------------------------------------------ | -------------------- |
| [`taplo`](https://taplo.tamasfe.dev/) | TOML formatter, used by `make format` / `make tomlfmt` and the pre-commit hook | `brew install taplo` |
| `npx` (Node)                          | Runs `prettier` and `markdownlint-cli2` on demand                              | `brew install node`  |

Pin: CI installs `taplo` v0.9.3 — match it locally if you hit format-drift between your machine and CI.

## Test environment

Integration tests need real cluster credentials. Create `.test.sh` at the repo root:

```bash
export SML_SWISSAI_RESEARCH_API_KEY=<your-api-key>
export SML_FIRECREST_CLIENT_ID=<your-client-id>
export SML_FIRECREST_CLIENT_SECRET=<your-client-secret>
export SML_SYSTEM=clariden
export SML_FIRECREST_TOKEN_URI=<your-token-uri>
export SML_FIRECREST_URL=<your-firecrest-url>
export SML_PARTITION=normal
export SML_RESERVATION=<your-reservation>
```

`.test.sh` is gitignored; the test targets source it automatically.

## Common make targets

| Target                    | What it does                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `make format`             | Format Python (`ruff`), TOML (`taplo`), JSON/YAML (`prettier`), Markdown (`markdownlint-cli2`) |
| `make shellcheck`         | Lint shell scripts                                                                             |
| `make markdownlint`       | Lint Markdown                                                                                  |
| `make test-lightweight`   | Auto-CI subset of integration tests                                                            |
| `make test-comprehensive` | Full integration test suite                                                                    |
| `make clean-cache`        | Remove cache files                                                                             |
| `make clean-dev`          | Remove the venv and cache                                                                      |

## Debugging

Set `SML_DEBUG=1` to include local variables in crash tracebacks:

```bash
export SML_DEBUG=1
```

> **Warning:** `SML_DEBUG=1` may expose secrets (Swiss AI Research API Key, FirecREST credentials) in crash output. Don't share terminal output captured with this flag.

By default, locals are stripped from crash reports.

## Adding a new model recipe

The lowest-friction contribution: a shell script under `examples/<system>/cli/<vendor>/`, plus an optional catalog entry for the interactive `sml` picker. See [Adding a New Model](adding-models.md) for the walkthrough.

## Modifying the SLURM submission script

The SLURM script is **rendered from Python at submit time** — there is no static `script.sh` or `template.jinja` to edit. The renderer is in [`src/swiss_ai_model_launch/launchers/framework.py`](https://github.com/swiss-ai/model-launch/blob/main/src/swiss_ai_model_launch/launchers/framework.py).

### What gets rendered

A single `master.sh` (visible via `--output-script` — see [usage](usage-advanced.md#inspecting-what-would-be-submitted-output-script-dir)) containing in order:

1. **Self-extracting rank scripts** — single-quoted `cat`-heredocs that lay down `head.sh`, optionally `follower.sh`, optionally `router.sh` under `$HOME/.sml/job-${SLURM_JOB_ID}/`
2. **Telemetry** POST (optional — skipped when telemetry is disabled)
3. **Arch detection** — sets `OPENTELA_BIN`, `SP_NCCL_SO_PATH`, `metrics_agent_bin` per `aarch64` / `x86_64`
4. **Node mapping** — `mapfile -t nodes < <(scontrol show hostnames ...)`
5. **Per-replica head IP discovery** — one `hostname -i` srun per replica
6. **Per-rank `srun` calls** — one block per (replica, rank). Each binds the rank dir into the pyxis container via `--container-mounts="$RANKS_DIR:$RANKS_DIR"` and invokes `bash $RANKS_DIR/<role>.sh`
7. **vmagent** (optional) — metrics scraper on the batch node
8. **Replica health checker** — background loop on the batch node (`_render_health_checker`) that lays down `$RANKS_DIR/replica_health_checker.py`, probes each replica's framework `/health`, and writes an atomic JSON report (`logs/${SLURM_JOB_ID}/replica_health.json`) the CLI reads. Always rendered; disowned and killed by the EXIT trap.
9. **Router** (optional) — `sglang_router` on `nodes[0]` when `replicas > 1 && --router sglang`
10. **Footer** — connect/cancel hints, `wait`, "Master finished"

### Where to make changes

| If you want to change… | Edit… |
| --- | --- |
| What runs **inside** the container per rank | `_render_sglang_head`, `_render_sglang_follower`, `_render_vllm_head`, `_render_vllm_follower` |
| Framework env exports (NCCL flags, `no_proxy`, JIT DeepGEMM toggle, …) | `Sglang.env_exports` / `Vllm.env_exports` |
| Add a new inference framework | Subclass `Framework`, register in `_FRAMEWORKS`, write per-shape renderers |
| The OpenTela wrap | `_opentela_wrap` |
| The router rank script | `_render_router` |
| Arch detection / node mapping / vmagent / footer | The matching `_render_<section>` functions |
| What gets bind-mounted into the container per srun | The `--container-mounts` line in `_render_replica_launches` / `_render_router_launch` |
| The toml mount list itself (per env: sglang, vllm, …) | The files under `src/swiss_ai_model_launch/assets/envs/` |
| Total nodes / partition / time / SBATCH directives | `to_sbatch_args` on `LaunchArgs` (or `render_sbatch_header` for the firecrest path) |
| New CLI flag flowing into `LaunchArgs` | Add to `LaunchArgs` (pydantic), wire through `build_launch_args_from_advanced` in `cli/main.py` |

### Preview your change

```bash
sml advanced ... --output-script /tmp/before    # current behaviour
# edit framework.py
sml advanced ... --output-script /tmp/after     # new behaviour
diff -r /tmp/before /tmp/after                  # per-file diff across master + ranks
```

For full coverage, the test matrix at `tests/unit/test_rendered_scripts_lint.py` renders 64 configurations (framework × replicas × nodes_per_replica × use_router × disable_opentela × telemetry) and runs `bash -n` + `shellcheck` against each. If your change leaves any of those broken, the test will catch it before submit time:

```bash
uv run pytest tests/unit/test_rendered_scripts_lint.py -q
```

`tests/unit/test_examples.py` also renders six real example scripts through the production CLI parser, so adding a flag that breaks one of those will fail there.

## Container images

The containers models run in are built from `images/<name>/Dockerfile` — on the cluster, by CI. See [Building Container Images](building-images.md) for how to add or change one.

## CI / CD

See [CI/CD](ci-cd.md) for the pipeline structure. PRs run static checks → image build → integration tests; each stage gates the next.

## Filing issues / PRs

- Bugs: use the [bug report template](https://github.com/swiss-ai/model-launch/blob/main/.github/ISSUE_TEMPLATE/bug-report.md). Include the failing command and the trailing chunk of TUI logs.
- New models: use the [adding-new-model template](https://github.com/swiss-ai/model-launch/blob/main/.github/ISSUE_TEMPLATE/adding-new-model-to-sml.md).
- PRs: keep them focused; pre-commit hooks must pass; integration tests must pass on at least one partition.
