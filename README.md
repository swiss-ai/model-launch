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

### Examples

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
