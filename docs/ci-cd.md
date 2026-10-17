# CI/CD

The main pipeline ([`ci.yml`](https://github.com/swiss-ai/model-launch/blob/main/.github/workflows/ci.yml)) runs in sequential stages, each gating the next:

> Static Checks → Catalog-Only Detection → Integration Tests

Container images are not built here: [swiss-ai/model-launch-images](https://github.com/swiss-ai/model-launch-images) builds, scans and publishes them. See [Building Container Images](building-images.md).

## Workflows

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci.yml` | push to `main`, PRs, dispatch | The pipeline below |
| `static.yml` | called by `ci.yml` | Lint, format, type checks |
| `docs.yml` | `docs/`, `mkdocs.yml`, `pyproject.toml` changes | `mkdocs build --strict`; deploys Pages from `main` |
| `sonar.yml` | push to `main`, PRs | Unit tests + coverage → SonarCloud |
| `model-paths.yml` | daily at 05:30 UTC, dispatch | Checks every `models.json` and example path still holds a model |

## Triggers

| Event | Behaviour |
| --- | --- |
| PR to `main` | Full pipeline |
| Push to `main` | Full pipeline |
| Draft PR | Static checks only |
| `workflow_dispatch` | Full pipeline; runs comprehensive tests |

PRs re-run on `opened`, `reopened`, `synchronize`, `labeled`, `ready_for_review` — the `labeled` trigger is what lets you switch test tiers without a new commit.

## Stage 1: static checks

Six parallel jobs: `ruff` lint/format, `mypy`, `shellcheck`, `markdownlint`, `taplo` (TOML), `prettier` (JSON/YAML).

All reproducible locally with `make static`, or individually (`make lint`, `make mypy`, …). See [Development](development.md#common-make-targets).

## Stage 2: catalog-only detection

A model PR (from model-launch-factory, or a person) that touches only `src/swiss_ai_model_launch/assets/models.json`, `src/swiss_ai_model_launch/assets/envs/*.toml` and `mfa_examples/` changes no launcher code, so the integration tiers are skipped for it. Any other file, a dispatch, or a push with nothing to diff against runs them.

## Stage 3: integration tests

Tests hit a real cluster over FireCREST. Exactly one tier runs per PR:

| Tier | Selected by | Target |
| --- | --- | --- |
| Lightweight | default | `make _test-lightweight` (`-n 2`) |
| Standard | `requires-std-tests` label | `make _test-std` (`-n 13`) |
| Comprehensive | `requires-comprehensive-tests` label, or dispatch | `make _test-comprehensive` (`-n 28`) |

Comprehensive wins over std, which wins over the default.

Locally, use the non-underscore targets (`make test-lightweight`) — they source `.test.sh` for credentials. See [Development](development.md#test-environment).

### Model path checks

The `paths`-marked cases list each weights directory over FireCREST — no SLURM job, seconds for both suites:

| Test | Covers | Resolved from |
| --- | --- | --- |
| `test_catalog_paths.py` | every `models.json` entry | `model_path`, else `<registry>/<vendor>/<model>` |
| `test_example_paths.py` | every `--model` / `--model-path` / `--tokenizer` in `examples/clariden/**/*.sh` | the flag value, with literal shell variables expanded |

A case fails if the directory is gone or unreadable, or if it holds none of its marker files — `config.json` or `params.json` (Mistral's native layout) for a model, `tokenizer.json` or `tokenizer_config.json` for a tokenizer. An emptied checkpoint directory still exists, so presence alone proves nothing.

Example references are deduplicated by path, so a directory shared by several recipes is listed once and a failure names all of them. Only `examples/clariden` is covered: the beverin/bristen recipes target clusters CI's credentials don't reach.

These carry the `lightweight`, `std` and `comprehensive` marks as well, so every tier runs them: the launch tests only cover the models they launch, and only the comprehensive tier runs the examples at all. `model-paths.yml` runs the same target (`make _test-paths`) daily, which is what catches a path that rotted while nobody touched the repo.

## Configuration

| Name | Kind | Used for |
| --- | --- | --- |
| `SML_FIRECREST_API_KEY` | secret | FireCREST service-account API key (shared across clusters) |
| `SML_SWISSAI_RESEARCH_API_KEY` | secret | Integration tests |
| `SML_FIRECREST_URL`, `SML_SYSTEM`, `SML_PARTITION`, `SML_RESERVATION` | variable | The cluster integration tests run on |
| `SONAR_TOKEN` | secret | SonarCloud (skipped for forked PRs) |

CI authenticates as a **service account**, not a personal account: the key is sent as an `X-API-Key` header, so `SML_FIRECREST_URL` must point at the PAT gateway (e.g. `https://f7t-pat.api.svc.cscs.ch/mlp`) rather than the Developer Portal endpoint. Everything the jobs touch — its home directory under `/users/<service-account>/.sml` and any SLURM reservation — must be writable/usable by that account.

## When something fails

| Symptom | Cause |
| --- | --- |
| Stops before any test | A static check failed — reproduce with `make static` |
| `path does not exist or is not readable` | A catalog entry's or recipe's weights moved or were deleted — repoint it or drop it |
| `UnexpectedStatusException: last request: 408 …` / `500 …` in a path test | FirecREST couldn't `stat` the path (command timeout, backend error) even after retries — the path itself is fine; re-run the job |
| `no model path extracted from: …` | An example's model flag isn't in a shape the extractor reads — see `tests/example_paths.py` |
