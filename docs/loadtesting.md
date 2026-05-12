# Loadtesting

SML can run k6 loadtests as SLURM jobs inside the cluster. The model server and the load generator both stay on the cluster network; k6 is not run on your laptop.

## What SML Submits

Each loadtest submits a one-node SLURM job that runs k6 in a container.

Default k6 container:

```bash
/capstor/scratch/cscs/bsezen/container-images/k6.sqsh
```

The image is built from `grafana/k6:1.7.1`.

SML stages these files into the loadtest job directory:

- `script.js` - the packaged k6 script, or your custom script
- `run_config.json` - server, scenario, model, and request settings

By default, k6 also remote-writes loadtest metrics to:

```bash
https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write
```

Override the endpoint with `--loadtest-metrics-remote-write-url` or `SML_LOADTEST_METRICS_REMOTE_WRITE_URL`. Disable remote write with `--no-loadtest-metrics-remote-write`. Metrics are tagged with `scenario`, `run_label`, and `model` for Grafana filtering.

Trend metrics are exported as explicit stats, not native histograms, for dashboard compatibility. For example, look for series such as `k6_e2e_latency_ms_p95`, `k6_e2e_latency_ms_p99`, and `k6_http_req_duration_p95`.

Prompt corpora are not committed to git and are not uploaded with the job. They must already exist at a path visible from the cluster container.

Default prompt corpus:

```bash
/capstor/scratch/cscs/bsezen/loadtest/prompts.json
```

Override it with:

```bash
--loadtest-prompts-file /capstor/scratch/cscs/bsezen/loadtest/prompts.json
```

or:

```bash
export SML_LOADTEST_PROMPTS_FILE=/capstor/scratch/cscs/bsezen/loadtest/prompts.json
```

## Run Against An Existing Endpoint

Use `sml loadtest run` when the model is already running, or when you want to test an external OpenAI-compatible endpoint.

```bash
sml loadtest run \
  --firecrest-system clariden \
  --partition normal \
  --loadtest-server-url https://your.endpoint.example \
  --loadtest-model your-model-name \
  --loadtest-scenario throughput \
  --loadtest-prompts-file /capstor/scratch/cscs/bsezen/loadtest/prompts.json
```

`--loadtest-model` is the value sent in the OpenAI-compatible request body. If you pass `--served-model-name`, SML uses that as the default model value.

By default, SML waits for `--served-model-name` to become healthy before starting k6. Use `--no-wait-until-healthy` to skip this check.

## Launch Then Loadtest

Use `sml loadtest advanced` to launch a model and then run k6 against it.

```bash
sml loadtest advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami) \
    --host 0.0.0.0 \
    --port 8080" \
  --loadtest-scenario throughput \
  --loadtest-max-tokens 512 \
  --loadtest-ignore-eos \
  --cancel-after-loadtest
```

`--cancel-after-loadtest` cancels the launched model job after the cluster k6 job has finished.

By default, SML waits for the launched model to become healthy before submitting the cluster k6 job.

Use `--loadtest-ignore-eos` to force loadtest requests to keep generating until `--loadtest-max-tokens` or the scenario `max_tokens` value. Use `--no-loadtest-ignore-eos` to force normal EOS stopping, even for scenarios that enable `ignore_eos`.

Use `--loadtest-max-tokens prompt` to take the output token cap from each prompt corpus entry's `max_tokens` field instead of using the scenario default or a single run-wide override.

The model-ready health check waits up to 1000000 seconds by default. Override it with `--loadtest-ready-timeout` when you want a shorter wait.

## Scenarios

Built-in scenarios are packaged under `src/swiss_ai_model_launch/assets/scenarios`.

| Scenario     | Pattern                         | Duration | Think time | Max tokens | Prompt labels                         | Use case                                                  |
| ------------ | ------------------------------- | -------- | ---------- | ---------- | ------------------------------------- | --------------------------------------------------------- |
| `throughput` | 20 constant VUs                 | 15m      | 2s         | 2048       | all                                   | Baseline sustained throughput.                            |
| `ramp`       | 0 -> 10 -> 25 -> 50 VUs         | 16m      | 2s         | 2048       | all                                   | Gradual capacity ramp with plateaus.                      |
| `stress`     | 0 -> 20 -> 50 -> 100 -> 150 VUs | 16m      | 2s         | 2048       | all                                   | Push the service past normal operating load.              |
| `spike`      | 10 -> 100 -> 10 VUs             | 8m30s    | 0s         | 4096       | all                                   | Sudden traffic surge and recovery behavior.               |
| `soak`       | 20 constant VUs                 | 30m      | 2s         | 2048       | all                                   | Longer stability run for drift, leaks, and tail latency.  |
| `decode`     | 50 constant VUs                 | 15m      | 0s         | 4096       | `short`, `medium`                     | Decode-heavy run with shorter prompts and longer outputs. |
| `kv_stress`  | 0 -> 30 -> 0 VUs                | 15m      | 0s         | 4096       | `long_input`, `xl_input`, `conv_long` | KV-cache pressure with long inputs and long outputs.      |
| `open_loop`  | 20 arrivals/s                  | 15m      | 0s         | 2048       | all                                   | Fixed request-rate latency test with EOS ignored.         |
| `open_loop_ramp` | 2 -> 30 arrivals/s          | 15m      | 0s         | 2048       | all                                   | Open-loop capacity sweep with EOS ignored.                |
| `open_loop_decode` | 2 -> 5 arrivals/s        | 12m      | 0s         | 512        | `short`, `medium`                     | Open-loop decode-focused A/B benchmark.                   |
| `realistic`  | 20 constant VUs                 | 15m      | 30s        | 2048       | all                                   | Lower-pressure interactive traffic shape.                 |

Custom scenarios can be placed in `./scenarios/` where you run `sml`. Use YAML, YML, or JSON. A custom scenario with the same name overrides the built-in one.

Prompt labels are tags inside the prompt corpus. Scenarios use them to select a subset of prompts, for example `decode` selects shorter prompts while `kv_stress` selects long-input prompts. Put label choices in scenario YAML rather than on the command line.

The k6 script shuffles the selected prompt corpus with a deterministic seed, then cycles through that shuffled order by global iteration number. This keeps repeated runs comparable while avoiding artifacts from sorted prompt files. The default seed is `1`; override it with `--loadtest-prompt-seed`. For paired A/B runs, use the same seed for both configurations.

Requests are sent as non-streaming chat completions (`stream: false`) so latency and token accounting come from complete responses.

## Results

When `--wait-for-loadtest` is enabled, SML waits for the cluster k6 job to finish and copies the summary to:

```bash
~/.sml/loadtest/<run-id>/summary_<scenario>_<timestamp>.json
```

Use `--no-wait-for-loadtest` to submit the loadtest job and return immediately.
