# Multi-Node Inference Server

Framework-agnostic SLURM job submission for distributed inference servers (SGLang, vLLM).

## Overview

This system submits SLURM jobs to launch inference servers across multiple nodes. It's designed to be completely framework-agnostic - specify the framework and pass through all framework-specific parameters.

## DeepSeek

### Single Worker (4 nodes)

```bash
python serving/multi-node/submit_job.py \
  --slurm-nodes 4 \
  --serving-framework sglang \
  --slurm-environment /capstor/store/cscs/swissai/infra01/users/rosmith/torrent/rob_ofi.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 --tp-size 16 --host 0.0.0.0 --port 8080 --served-model-name deepseek-ai/DeepSeek-V3.1"
```

### Multiple Workers with Router

```bash
python serving/multi-node/submit_job.py \
  --slurm-nodes 8 \
  --workers 2 \
  --nodes-per-worker 4 \
  --serving-framework sglang \
  --slurm-environment /capstor/store/cscs/swissai/infra01/users/rosmith/torrent/rob_ofi.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 --tp-size 16 --host 0.0.0.0 --port 8080 --served-model-name deepseek-ai/DeepSeek-V3.1" \
  --use-router
```

### With OCF (Open Compute Framework)

```bash
python serving/multi-node/submit_job.py \
  --slurm-nodes 4 \
  --serving-framework sglang \
  --slurm-environment /capstor/store/cscs/swissai/infra01/users/rosmith/torrent/rob_ofi.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 --tp-size 16 --host 0.0.0.0 --port 8080 --served-model-name deepseek-ai/DeepSeek-V3.1" \
  --use-ocf \
  --ocf-bootstrap-addr "/ip4/148.187.108.172/tcp/43905/p2p/QmQsNxJVa2rnidp998qAz4FCutgmjBsuZqtrxUUy5YfgBu"
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
- `--workers`: Number of independent workers (default: 1)
- `--nodes-per-worker`: Nodes per worker (default: all nodes / workers)
- `--worker-port`: Port for workers (default: 5000)

### Router Options
- `--use-router`: Enable router (only active if workers > 1)
- `--router-environment`: SLURM environment for router (default: same as worker)
- `--router-port`: Router port (default: 30000)
- `--router-args`: Arguments passed to the router

### OCF (Open Compute Framework) Options
- `--use-ocf`: Enable OCF wrapper for framework launch
- `--ocf-bootstrap-addr`: OCF bootstrap address (required if `--use-ocf` is set)
- `--ocf-service-name`: OCF service name (default: `llm`)
- `--ocf-service-port`: OCF service port (default: 8080)
- `--ocf-all-nodes`: Run OCF on all nodes (default: master node only)

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

Connect to running job:
```bash
srun --jobid <job_id> -w <node> --overlap --pty bash
```

Cancel job:
```bash
scancel <job_id>
```
