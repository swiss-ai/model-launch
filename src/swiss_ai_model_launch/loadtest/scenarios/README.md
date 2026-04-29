# Scenario YAML Structure for k6

Built-in scenarios live in this package directory. User-defined scenarios should be defined in a YAML or JSON file under `./scenarios/` in the directory where you run `sml`.

## Example: ramp.yaml

```
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

```
name: throughput
executor: constant-vus
vus: 20
duration: 15m
think_time: "2"
max_tokens: "2048"
```

- You can add any custom fields (e.g., `prompt_labels`) as needed.
- The launcher will load the scenario config from the YAML or JSON file matching the scenario name (e.g., `ramp` -> `ramp.yaml`).
- User-defined scenarios in `./scenarios/` override built-in scenarios with the same name.

## Usage

- Place your scenario YAML or JSON files in `./scenarios/`.
- Run the loadtest launcher as usual, for example `sml loadtest run --loadtest-scenario ramp`.
- For direct k6 runs, pass `RUN_CONFIG_JSON` from your shell or a wrapper.
- If you use scenario-name lookup, `ramp` resolves to `./scenarios/ramp.yaml` first, then the packaged built-in scenarios.
