# Multi-Node Inference Server

Framework-agnostic SLURM job submission for distributed inference servers (SGLang, vLLM) with OCF (Open Compute Framework) integration enabled by default.

## Overview

This system submits SLURM jobs to launch inference servers across multiple nodes. It's designed to be completely serving framework-agnostic - specify the framework and pass through all framework-specific parameters. OCF is enabled by default for service discovery, external access (via [serving](https://serving.swissai.cscs.ch)) and monitoring.


## Mistral

### Single Worker Single Node

Even for single-node deployments, you can use the framework:

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment /capstor/store/cscs/swissai/infra01/users/rosmith/torrent/rob_ofi.toml \
    --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-7B-v0.1 --tp-size 4 --host 0.0.0.0 --port 8080 --served-model-name mistralai/Mistral-7B-v0.1"
```


## DeepSeek

### Single Worker (4 nodes)

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --serving-framework sglang \
  --slurm-environment /capstor/store/cscs/swissai/infra01/users/rosmith/torrent/rob_ofi.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 --tp-size 16 --host 0.0.0.0 --port 8080 --served-model-name deepseek-ai/DeepSeek-V3.1"
```

### Multiple Workers with Router

```bash
python serving/submit_job.py \
  --slurm-nodes 8 \
  --workers 2 \
  --nodes-per-worker 4 \
  --serving-framework sglang \
  --slurm-environment /capstor/store/cscs/swissai/infra01/users/rosmith/torrent/rob_ofi.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 --tp-size 16 --host 0.0.0.0 --port 8080 --served-model-name deepseek-ai/DeepSeek-V3.1" \
  --use-router
```

## Kimi-k2

Kimi-k2 requires the `--tool-call-parser kimi_k2` parameter for tool usage support. With TP16 and 4 GPUs per node, this requires 4 nodes. May need additional packages like `blobfile`.

### Single Worker (4 nodes, TP16)

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --serving-framework sglang \
  --slurm-environment /capstor/store/cscs/swissai/infra01/users/rosmith/torrent/rob_ofi.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2-Instruct --tp-size 16 --host 0.0.0.0 --port 8080 --served-model-name moonshotai/Kimi-K2-Instruct --trust-remote-code --tool-call-parser kimi_k2" \
  --pre-launch-cmds "pip install blobfile"
```

## Parameters

### Required
- `--slurm-nodes`: Total number of nodes to allocate
- `--serving-framework`: Either `sglang` or `vllm`

### Optional
- `--slurm-environment`: Path to environment TOML file (default: uses `{framework}.toml` in same directory)
- `--slurm-job-name`: Job name (default: random 4-letter ID)
- `--slurm-partition`: SLURM partition (default: `normal`)
- `--framework-args`: Arguments passed directly to the serving framework
- `--pre-launch-cmds`: Commands to run before launching framework (e.g., `"pip install blobfile; pip install package2"`)
- `--workers`: Number of independent workers (default: 1)
- `--nodes-per-worker`: Nodes per worker (default: all nodes / workers)
- `--worker-port`: Port for workers (default: 5000)

### Router Options
- `--use-router`: Enable router (only active if workers > 1)
- `--router-environment`: SLURM environment for router (default: same as worker)
- `--router-port`: Router port (default: 30000)
- `--router-args`: Arguments passed to the router

### OCF (Open Compute Framework) Options

**OCF is enabled by default** for service discovery and health monitoring. It runs on the master node (rank 0) of each worker.

- `--disable-ocf`: Disable OCF wrapper (OCF is enabled by default)
- `--ocf-bootstrap-addr`: OCF bootstrap address (default: `/ip4/148.187.108.172/tcp/43905/p2p/QmQsNxJVa2rnidp998qAz4FCutgmjBsuZqtrxUUy5YfgBu`)
- `--ocf-service-name`: OCF service name (default: `llm`)
- `--ocf-service-port`: OCF service port - must match the port your framework listens on (default: 8080)

## Monitoring

After submission, logs are available in `logs/<job_id>/`:
- `log.out` - Main job output with worker URLs
- `log.err` - Main job errors
- `worker<id>_node<rank>_<hostname>.out` - Per-worker stdout
- `worker<id>_node<rank>_<hostname>.err` - Per-worker stderr

Check job status:
```bash
squeue -j <job_id>
```

or 
```bash
squeue --me
```

or via CSCS web [dashboard](https://my.mlp.cscs.ch/).

Connect to running job:
```bash
srun --jobid <job_id> -w <node> --overlap --pty bash
```

Cancel job:
```bash
scancel <job_id>
```

