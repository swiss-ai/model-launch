# `sml`: Swiss AI Model Launch

A CLI tool for launching AI models on HPC systems via FirecREST, a remote launcher, or SLURM commands.

## Install

1. Create a virtual environment and install the package.

   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   ```

2. Install the package.

   - For regular use:

     ```bash
     uv pip install -e .
     ```

   - For development (includes dev dependencies):

      ```bash
      uv pip install -e ".[dev]"
      pre-commit install
      ```

## Development

There is a `Makefile` with common development commands.

```bash
make format
make clean-cache
make clean-dev
```

## Running Integration Tests

The integration tests launch a real model via FirecREST and verify it becomes healthy. Set the following environment variables before running:

```bash
export FIRECREST_URL=<your-firecrest-url>
export FIRECREST_TOKEN_URI=<your-token-uri>
export FIRECREST_CLIENT_ID=<your-client-id>
export FIRECREST_CLIENT_SECRET=<your-client-secret>
export FIRECREST_SYSTEM=clariden
export FIRECREST_ACCOUNT=<your-account>
export FIRECREST_PARTITION=normal
export CSCS_API_KEY=<your-api-key>
```

Then run:

```bash
pytest tests/integration/test_launch_apertus.py -s -v
```

The `-s` flag streams polling output (job status, health) to stdout. If any env var is missing the test will fail with a clear error listing the missing variables.

Timeouts can be adjusted via:

```bash
export LAUNCH_TIMEOUT_MINUTES=15   # default: 10
export HEALTH_TIMEOUT_MINUTES=25   # default: 20
```
