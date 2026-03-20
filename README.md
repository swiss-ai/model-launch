# `sml`: Swiss AI Model Launch

A CLI tool for launching AI models on HPC systems via FirecREST, a remote launcher, or SLURM commands.

## Install

1. Regular use
   - SSH
   ```bash
   pip install git+ssh://git@github.com/swiss-ai/model-launch.git
   ```
   - HTTPS (once public)
   ```bash
   pip install git+https://github.com/swiss-ai/model-launch.git
   ```

2. Development

   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   ```
   ```bash
   git clone git@github.com:swiss-ai/model-launch.git && cd model-launch
   uv pip install -e ".[dev]"
   pre-commit install
   ```

## Usage

### Initialization

The first time you run `sml` command, you will be prompted to do the initialization, which includes providing necessary credentials and configurations.

There is three ways to initialize the mean of launching commands for the CLI as listed below.

- FirecREST: You will be able to run the CLI on your local machine and the CLI will use FirecREST to submit jobs. You will need to provide your FirecREST credentials during the initialization. If you don't have the credentials, you can follow the instructions in the [Appendix](#acquiring-firecrest-credentials) to acquire them.
- Remote Launcher: Not operational yet.
- SLURM: Not operational yet.

For health check, you will be prompted to provide your CSCS API key. If you don't have the API key, you can follow the instructions in the [Appendix](#acquiring-cscs-api-key) to acquire it.

### Launching a Model

After completing the initialization, you can simply run the `sml` command. The CLI will guide you through the process of launching a model, which includes selecting a model, providing necessary information for the launch, and confirming the launch.

## Development

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

Please follow the instructions in the [FirecREST documentation](https://docs.cscs.ch/services/devportal/#manage-your-applications) to acquire the necessary credentials for authentication.

### Acquiring CSCS API Key

Please proceed to [serving platform](https://serving.swissai.svc.cscs.ch) and login using your institutional account. Then, navigate to the "View API Keys" section. You will see your API key listed there.
