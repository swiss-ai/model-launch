# Using SML

`sml` launches a model interactively from a curated catalog. You answer a few prompts and SML submits the SLURM job, opens a TUI, and streams logs.

If you need full control over SLURM args, framework flags, or a model that isn't in the catalog, see [Advanced Usage](usage-advanced.md) instead.

## Quickstart

After [initialization](initialization.md):

```bash
sml
```

You'll be prompted for: target system, partition, model, framework, replica count, time limit. SML submits the job and the TUI takes over.

## Skipping the prompts

You can pre-fill any prompt with a CLI flag or environment variable. Whatever you don't supply, SML asks for.

| Argument             | Environment Variable     | Description                                            |
| -------------------- | ------------------------ | ------------------------------------------------------ |
| `--firecrest-system` | `SML_FIRECREST_SYSTEM`   | Target system (required if launcher is `firecrest`)    |
| `--partition`        | `SML_PARTITION`          | SLURM partition                                        |
| `--reservation`      | `SML_RESERVATION`        | SLURM reservation (optional)                           |
| `--model`            |                          | Model to launch (`<vendor>/<model>`)                   |
| `--framework`        |                          | Inference framework                                    |
| `--workers`          |                          | Number of workers (replicas)                           |
| `--use-router`       |                          | Load-balance across replicas (`yes` / `no`)            |
| `--time`             |                          | Job time limit (`HH:MM:SS`)                            |

CLI flags take precedence over environment variables.

### Tip: env vars for things that rarely change

System and partition are usually constant for a given user — putting them in your shell rc file means you never type them again:

```bash
export SML_FIRECREST_SYSTEM=clariden
export SML_PARTITION=normal
```

(This advice applies only to `sml`. For [Advanced Usage](usage-advanced.md), system and partition are passed as CLI args alongside everything else.)

## Example

```bash
export SML_FIRECREST_SYSTEM=clariden
export SML_PARTITION=normal

sml preconfigured \
  --model swiss-ai/Apertus-8B-Instruct-2509 \
  --framework sglang \
  --workers 1 \
  --time 02:00:00
```

After submission, the TUI shows job state and live logs. When the model is healthy, it's reachable at the served-model URL.

## What if my model isn't in the catalog?

Use [`sml advanced`](usage-advanced.md) to point at any model path on the cluster filesystem (or huggingface handle).

## Next

- [Advanced Usage](usage-advanced.md) — for non-catalog models or fine SLURM control
- [How to size a model](sizing.md) — picking replica count, nodes-per-replica, GPU type
- [Benchmarking](benchmarking.md) — measuring throughput once the model is up
