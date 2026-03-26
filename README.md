# `sml`: Swiss AI Model Launch

A CLI tool for launching AI models on HPC systems via FirecREST or SLURM commands.

## Install

- SSH

  ```bash
  pip install git+ssh://git@github.com/swiss-ai/model-launch.git
  ```

- HTTPS

  ```bash
  pip install git+https://github.com/swiss-ai/model-launch.git
  ```

## Usage

### Quick Start

Just run:

```bash
sml
```

That's it. On first run, SML will guide you through a one-time setup. After that, running `sml` again will take you straight to launching a model.

The rest of this section documents subcommands and CLI arguments for advanced or automated use cases.

### Subcommands

| Command          | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `sml init`       | Initialize SML configuration                             |
| `sml quickstart` | Launch a model with guided prompts                       |
| `sml advanced`   | Launch a model with advanced configuration (coming soon) |

### Initialization (`sml init`)

Run `sml init` to configure SML. You will be prompted to provide necessary credentials and configurations.

There are three ways to initialize the launcher:

- FirecREST: Run the CLI on your local machine; jobs are submitted via FirecREST. You will need to provide your FirecREST credentials. If you don't have them, follow the instructions in the [Appendix](#acquiring-firecrest-credentials).
- Remote Launcher: Not operational yet.
- SLURM: Not operational yet.

For health check, you will be prompted to provide your CSCS API key. If you don't have the API key, follow the instructions in the [Appendix](#acquiring-cscs-api-key).

All prompts can be pre-filled via CLI arguments to skip interactive prompts:

| Argument                       | Description                                            |
| ------------------------------ | ------------------------------------------------------ |
| `--launcher`                   | Job submission method (`firecrest`, `remote`, `slurm`) |
| `--firecrest-url`              | FirecREST API URL                                      |
| `--firecrest-token-uri`        | FirecREST token URI                                    |
| `--firecrest-client-id`        | FirecREST client ID                                    |
| `--firecrest-client-secret`    | FirecREST client secret                                |
| `--remote-launcher-address`    | Remote launcher address (if using `remote`)            |
| `--remote-launcher-auth-token` | Remote launcher auth token (if using `remote`)         |
| `--cscs-api-key`               | CSCS API key for health checks                         |
| `--telemetry-endpoint`         | Endpoint for telemetry reports                         |

### Launching a Model (`sml quickstart`)

After completing initialization, run `sml quickstart` (or just `sml`). The CLI will guide you through selecting a model and providing the necessary launch configuration.

All prompts can be pre-filled via CLI arguments to skip interactive prompts:

| Argument                | Description                                  |
| ----------------------- | -------------------------------------------- |
| `--firecrest-system`    | Target HPC system to launch on               |
| `--firecrest-partition` | SLURM partition to use                       |
| `--model`               | Model to launch (`<vendor>::<model>`)        |
| `--framework`           | Inference framework to use                   |
| `--workers`             | Number of workers                            |
| `--use-router`          | Load balance across workers (`yes`, `no`)    |
| `--time`                | Job time limit (`HH:MM:SS`)                  |

## Development

### Setting Up Development Environment

```bash
git clone git@github.com:swiss-ai/model-launch.git && cd model-launch
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

### Testing Environment

For writing the unit tests, you have to create a `.test.sh` file in the root of the repository with the following content:

```shell
export FIRECREST_URL=<your-firecrest-url>
export FIRECREST_TOKEN_URI=<your-token-uri>
export FIRECREST_CLIENT_ID=<your-client-id>
export FIRECREST_CLIENT_SECRET=<your-client-secret>
export FIRECREST_SYSTEM=clariden
export FIRECREST_ACCOUNT=<your-account>
export FIRECREST_PARTITION=normal
export CSCS_API_KEY=<your-api-key>
```

This file will be sourced when running the tests with `make test`, and the environment variables will be available for the tests.

### Common Commands

There is a `Makefile` with common development commands.

1. To format code, you can run:

   ```bash
   make format
   ```

2. To run tests, you can run:

   ```bash
   make test
   ```

3. To clean up cache files, you can run:

   ```bash
   make clean-cache
   ```

4. To clean up the env and cache, you can run:

   ```bash
   make clean-dev
   ```

## Appendix

### Acquiring FirecREST Credentials

Please follow the instructions in the [FirecREST documentation](https://docs.cscs.ch/services/devportal/#manage-your-applications) to acquire the necessary credentials for authentication from [Developer Portal](https://developer.svc.cscs.ch/devportal/apis).

### Acquiring CSCS API Key

Please proceed to [serving platform](https://serving.swissai.svc.cscs.ch) and login using your institutional account. Then, navigate to the "View API Keys" section. You will see your API key listed there.
