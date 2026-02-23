# Swiss AI Model Launch

Framework-agnostic SLURM job submission for distributed inference servers (SGLang, vLLM) with OCF (Open Compute Framework) integration.

## Overview

This repository contains several sub-directories for different components of the model launch process.

- [`download/`](./download/): This directory contains scripts for downloading and storing the models in the cluster. It is strongly recommended to do so to avoid repeated downloads every time you want to launch a model.
- [`images/`](./images/): This directory contains the instructions and Dockerfiles for building custom container images for the SLURM nodes. The images will be passed on the top of the `.toml` environment configuration file for SLURM job submission, ensuring consistency and compatibility across all nodes in the SLURM cluster.
- [`serving/`](./serving/): This directory contains the main code and instructions for launching the models on the SLURM cluster.

## Setup

### Prerequisites

1. Git
2. Python
3. UV

   You can install it by `curl -LsSf https://astral.sh/uv/install.sh | sh`. See [uv](https://github.com/astral-sh/uv) for more details.

### Installation

```bash
git clone https://github.com/swiss-ai/model-launch
cd model-launch
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Examples

For detailed single/multi-node deployment examples including:

- **DeepSeek V3.1** (4 nodes, TP16)
- **Kimi-k2** (4 nodes, TP16)
- **Multiple workers with router**
- **OCF configuration**
- **Pre-launch commands**

See the comprehensive documentation in [serving/](serving/)

## Features

- **Framework Agnostic**: Supports both SGLang and vLLM
- **OCF Integration**: Service discovery and health monitoring enabled by default
- **Multi-Node Support**: Distributed inference across multiple nodes
- **Router Support**: Load balancing across multiple workers
- **Architecture Detection**: Automatically detects ARM64 vs x86_64
- **Pre-Launch Commands**: Install packages or run setup before framework launch
- **Flexible Configuration**: Pass-through framework arguments for complete control

## Contributing

GitHub issues are welcome! Feel free to:

- Report bugs or issues
- Suggest new features
- Propose model support additions
- Ask questions about usage

## Acknowledgements

This repo was inspired by Nathan's [torrent](https://github.com/swiss-ai/torrent/) repo on Swiss AI.
