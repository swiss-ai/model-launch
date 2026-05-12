# Scenario YAML Structure for k6

Built-in scenarios live in this package directory. User-defined scenarios should be defined in a YAML or JSON file under `./scenarios/` in the directory where you run `sml`.

## Example: ramp.yaml

```yaml
name: ramp
executor: ramping-vus
startVUs: 0
stages:
  - duration: 2m
    target: 10
  - duration: 3m
    target: 10
  - duration: 2m
    target: 25
  - duration: 3m
    target: 25
  - duration: 2m
    target: 50
  - duration: 3m
    target: 50
  - duration: 1m
    target: 0
gracefulRampDown: 30s
think_time: "2"
max_tokens: "2048"
```

## Example: throughput.yaml

```yaml
name: throughput
executor: constant-vus
vus: 20
duration: 15m
think_time: "2"
max_tokens: "2048"
```

## Example: open_loop.yaml

```yaml
name: open_loop
executor: constant-arrival-rate
rate: 20
timeUnit: 1s
duration: 15m
preAllocatedVUs: 120
maxVUs: 300
think_time: "0"
max_tokens: "2048"
request_timeout: 180s
ignore_eos: true
```

## Example: open_loop_ramp.yaml

```yaml
name: open_loop_ramp
executor: ramping-arrival-rate
startRate: 2
timeUnit: 1s
stages:
  - duration: 3m
    target: 5
  - duration: 3m
    target: 10
  - duration: 3m
    target: 15
  - duration: 3m
    target: 20
  - duration: 3m
    target: 30
preAllocatedVUs: 120
maxVUs: 400
think_time: "0"
max_tokens: "2048"
ignore_eos: true
```

## Example: open_loop_decode.yaml

```yaml
name: open_loop_decode
executor: ramping-arrival-rate
startRate: 2
timeUnit: 1s
stages:
  - duration: 3m
    target: 2
  - duration: 3m
    target: 3
  - duration: 3m
    target: 4
  - duration: 3m
    target: 5
preAllocatedVUs: 800
maxVUs: 1600
think_time: "0"
max_tokens: "512"
request_timeout: 600s
ignore_eos: true
prompt_labels:
  - short
  - medium
```

- You can add any custom fields (e.g., `prompt_labels`) as needed.
- The launcher will load the scenario config from the YAML or JSON file matching the scenario name (e.g., `ramp` -> `ramp.yaml`).
- User-defined scenarios in `./scenarios/` override built-in scenarios with the same name.

## Usage

- Place your scenario YAML or JSON files in `./scenarios/`.
- Run the loadtest launcher as usual, for example `sml loadtest run --loadtest-scenario ramp`.
- For direct k6 runs, pass `RUN_CONFIG_JSON` from your shell or a wrapper.
- If you use scenario-name lookup, `ramp` resolves to `./scenarios/ramp.yaml` first, then the packaged built-in scenarios.
