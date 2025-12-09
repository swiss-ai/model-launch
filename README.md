# Model-Launch

Framework-agnostic SLURM job submission for distributed inference servers (SGLang, vLLM) with OCF (Open Compute Framework) integration.

This repo was inspired by Nathan's [torrent](https://github.com/swiss-ai/torrent/) repo on SwissAI.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/swiss-ai/model-launch
cd model-launch
```

### 2. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
uv pip install Jinja2
```

## Quick Start

### Single Node Example (Mistral-7B)

Even for single-node deployments, you can use the multi-node framework:

```bash
python serving/multi-node/submit_job.py \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --slurm-environment /path/to/your/environment.toml \
  --framework-args "--model-path /path/to/mistralai/Mistral-7B-v0.1 --tp-size 4 --host 0.0.0.0 --port 8080 --served-model-name mistralai/Mistral-7B-v0.1"
```

### Multi-Node Examples

For detailed multi-node deployment examples including:
- **DeepSeek V3.1** (4 nodes, TP16)
- **Kimi-k2** (4 nodes, TP16)
- **Multiple workers with router**
- **OCF configuration**
- **Pre-launch commands**

See the comprehensive documentation in [serving/multi-node/README.md](serving/multi-node/README.md)

## Features

- **Framework Agnostic**: Supports both SGLang and vLLM
- **OCF Integration**: Service discovery and health monitoring enabled by default
- **Multi-Node Support**: Distributed inference across multiple nodes
- **Router Support**: Load balancing across multiple workers
- **Architecture Detection**: Automatically detects ARM64 vs x86_64
- **Pre-Launch Commands**: Install packages or run setup before framework launch
- **Flexible Configuration**: Pass-through framework arguments for complete control
