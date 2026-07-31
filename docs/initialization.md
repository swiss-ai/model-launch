# Initialization

Before using `sml`, run a one-time setup to provide credentials and choose how jobs are submitted to the cluster.

```bash
sml init
```

The wizard writes config to `~/.sml/config.yml` (override with `SML_CONFIG_DIR`). Re-running `sml init` overwrites the previous config.

You can skip the wizard by pre-filling answers via CLI flags or environment variables (table below).

## FirecREST or SLURM?

SML can submit jobs in two ways. Pick one — your choice only affects setup, not day-to-day usage.

| You are…                                                              | Use         | Why                                                                                  |
| --------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------ |
| On your laptop, want to launch jobs on a cluster                      | `firecrest` | FirecREST is a REST API in front of SLURM — no SSH session required.                 |
| Already SSH'd into the cluster (login node)                           | `slurm`     | Direct `sbatch` is simpler when you're already on the host.                          |
| Behind a corporate VPN that blocks the FirecREST endpoint             | `slurm`     | SSH usually still works.                                                             |
| Automating from CI / a long-running service                           | `firecrest` | No interactive SSH agent needed; client credentials work for headless flows.         |

If you're not sure, start with `firecrest` — it's what most users run.

## Initialization options

| CLI Argument            | Environment Variable          | Description                                                    |
| ----------------------- | ----------------------------- | -------------------------------------------------------------- |
| `--launcher`            |                               | Job submission method (`firecrest` or `slurm`)                 |
| `--firecrest-url`       |                               | FirecREST API URL (default: CSCS endpoint)                     |
| `--firecrest-token-uri` |                               | FirecREST token URI (default: CSCS auth endpoint)              |
|                         | `SML_FIRECREST_CLIENT_ID`     | FirecREST client ID                                            |
|                         | `SML_FIRECREST_CLIENT_SECRET` | FirecREST client secret                                        |
|                         | `SML_CSCS_API_KEY`            | CSCS Serving API Key (used for health checks of served model)  |

The FirecREST fields are only required when `--launcher firecrest`. `SML_CSCS_API_KEY` is required regardless of launcher.

## Where credentials come from

- **FirecREST client ID / secret** — Acquire from the [CSCS Developer Portal](https://developer.svc.cscs.ch/devportal/apis). See the [FirecREST docs](https://docs.cscs.ch/services/devportal/#manage-your-applications) for the full walkthrough.
- **CSCS Serving API Key** — Log in at [serving.swissai.svc.cscs.ch](https://serving.swissai.svc.cscs.ch/) with your institutional account, then go to **View API Keys**.

## Config file shape

The config is stored as a nested structure (a chain of configuration nodes), not a flat list of keys. Secrets are **not** written to the file in plaintext — they live in your OS keyring under the service name `swiss_ai_model_launch`, and the YAML holds a `__keyring__` placeholder instead.

`~/.sml/config.yml` after a successful `firecrest` init looks roughly like:

```yaml
name: init_config
type: chain
chain:
  - name: launcher_configuration
    type: branch
    head_configuration:
      name: launcher
      type: options
      value: firecrest
    branches:
      firecrest:
        name: firecrest_launcher_configuration
        type: chain
        chain:
          - {name: firecrest_url, type: text, value: https://api.cscs.ch/ml/firecrest/v2}
          - {name: firecrest_token_uri, type: text, value: https://auth.cscs.ch/...token}
          - {name: firecrest_client_id, type: password, value: __keyring__}
          - {name: firecrest_client_secret, type: password, value: __keyring__}
          - {name: cluster_ssh_host, type: text, value: null}
      slurm: null
  - {name: cscs_api_key, type: password, value: __keyring__}
```

The `value: __keyring__` entries are placeholders; the actual secrets live in your OS keyring, not in this file. Treat both the file and your keyring as sensitive. Don't commit the file.

## Next

- [Using SML](usage-sml.md) — interactive launch
- [Advanced Usage](usage-advanced.md) — full SLURM control
