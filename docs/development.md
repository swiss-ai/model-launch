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

The lowest-friction contribution. Drop a shell script under `examples/<system>/cli/<vendor>/`. Use the [adding-new-model-to-sml issue template](../.github/ISSUE_TEMPLATE/adding-new-model-to-sml.md) as a checklist; existing scripts (e.g. `examples/clariden/cli/swiss-ai/Apertus-8B-Instruct-2509-sglang.sh`) are good templates.

For models that should appear in the `sml` interactive catalog (not just `sml advanced`), the recipe also needs an entry in the model catalog — see existing entries under `src/swiss_ai_model_launch/`.

### Try it yourself first

The SML team can't take a "please add my model" request for every checkpoint that lands on Hugging Face. Before filing an issue, work the checklist:

1. **Find the closest existing example** under `examples/<system>/cli/<vendor>/` — same framework (sglang/vllm), similar size class, same architecture if possible. Copy it.
2. **Swap in your model path** via `--framework-args "--model-path /capstor/store/.../<your-model>"` (and `--served-model-name <something-unique>`).
3. **Try it with [`sml advanced`](usage-advanced.md).** If it serves, you're done — the script *is* the recipe; PR it.
4. **If it doesn't serve, narrow the failure** before opening an issue:
    - Does the same model work with the framework directly (no SML)? If not, it's a framework issue, not an SML issue — report upstream.
    - Does it OOM? See [Sizing](sizing.md) — you may need bigger TP, more nodes, or quantization.
    - Does it fail to load? Architecture not supported by the framework version in the [environment toml](../src/swiss_ai_model_launch/assets/envs/) — try the other framework, or a newer image.
5. **Only if you've gotten through 1-4 and are still stuck**, file an issue with the failing command, the trailing 50 lines of logs, and what you've already ruled out.

## CI / CD

See [CI/CD](ci-cd.md) for the pipeline structure. PRs run static checks → image build → integration tests; each stage gates the next.

## Filing issues / PRs

- Bugs: use the [bug report template](../.github/ISSUE_TEMPLATE/bug-report.md). Include the failing command and the trailing chunk of TUI logs.
- New models: use the [adding-new-model template](../.github/ISSUE_TEMPLATE/adding-new-model-to-sml.md).
- PRs: keep them focused; pre-commit hooks must pass; integration tests must pass on at least one partition.
