# Mooncake RAM cache on Clariden

These examples run Apertus-8B on one GPU with Mooncake Store 0.3.14. Each job starts a local master and an 8 GiB RAM pool. Disk storage is disabled. SGLang also uses a 4 GB HiCache host pool. The RAM pool lives in the serving worker and disappears when that worker exits.

Run either command from the repository root:

```bash
bash examples/clariden/mooncake/launch.sh vllm
bash examples/clariden/mooncake/launch.sh sglang
```

Each command uses `sml advanced` and requests one Clariden debug node for 90 minutes under account `infra01`. Set `SML_ACCOUNT` to use another account. Set `MODEL_PATH` to use another local model. The launch script contains the complete native framework arguments. vLLM reads its connector from `vllm-engine.yaml`. SGLang selects Mooncake through its HiCache arguments. Both use `mooncake.json` for RAM storage.

The environment files select the shared SQSH images:

```text
/capstor/store/cscs/swissai/infra01/container-images/examples/vllm_cuda13-mooncake-0.3.14-arm64.sqsh
/capstor/store/cscs/swissai/infra01/container-images/examples/sglang_cuda13-mooncake-0.3.14-arm64.sqsh
```

To inspect the generated job scripts before submitting a job, add `--output-script /absolute/path/to/output` to either launch command.
