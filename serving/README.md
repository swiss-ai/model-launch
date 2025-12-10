# Multi-Node Inference Server

Framework-agnostic SLURM job submission for distributed inference servers (SGLang, vLLM) with OCF (Open Compute Framework) integration enabled by default.

## Overview

This system submits SLURM jobs to launch inference servers across multiple nodes. It's designed to be completely serving framework-agnostic - specify the framework and pass through all framework-specific parameters. OCF is enabled by default for service discovery, external access (via [serving](https://serving.swissai.cscs.ch)) and monitoring.


## Model Overview Table

| Model / Configuration | Tested |
|----------------------|--------|
| [swiss-ai/Apertus-8B-Instruct-2509](#apertus-8b-instruct-2509) | ✅ |
| [mistralai/Mistral-7B-v0.1](#mistral-7b-v01) | ✅ |
| [Snowflake/snowflake-arctic-embed-l-v2.0](#snowflake-arctic-embed-l-v20) | ✅ |
| [deepseek-ai/DeepSeek-V3.1](#deepseek-v31) | ✅ |
| [deepseek-ai/DeepSeek-V3.1 (Multi-Worker Router)](#deepseek-v31-with-router-and-2x-workers) | ❓ |
| [moonshotai/Kimi-K2-Instruct](#kimi-k2-instruct) | ✅ |
| [moonshotai/Kimi-K2-Thinking](#kimi-k2-thinking) | ✅ |

Tested means the model has started and responded to a simple request.

## Models

### Apertus-8B-Instruct-2509

The [2 Apertus models](https://github.com/swiss-ai/model-spinning/blob/main/auto-spin/config.yaml#L2-L21) are continously running 24/7 this are launched every 5 minutes by a [scheduled github action](https://github.com/swiss-ai/model-spinning/blob/main/.github/workflows/autospin.yml#L5).

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/sglang.toml \
    --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
     --host 0.0.0.0 \
     --port 8080 \
     --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami)"
```
Note there is usually already a model called `swiss-ai/Apertus-8B-Instruct-2509` so to prevent name collisions it's important to rename the `served-model-name` to something else. If you remove it entirely then it defaults to long model-path.

### Mistral-7B-v0.1

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/sglang.toml \
    --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-7B-v0.1 \
      --host 0.0.0.0 \
      --port 8080 \
      --served-model-name mistralai/Mistral-7B-v0.1"
```


### snowflake-arctic-embed-l-v2.0

```bash
python serving/submit_job.py \
  --slurm-nodes 1 \
  --serving-framework vllm \
  --slurm-environment $(pwd)/serving/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/Snowflake/snowflake-arctic-embed-l-v2.0 \
   --host 0.0.0.0 \
   --port 8080 \
   --task embedding \
   --served-model-name Snowflake/snowflake-arctic-embed-l-v2.0"
```

### Qwen3-Next-80B-A3B-Instruct
--context-length 262144 --mem-fraction-static 0.8 --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/sglang.toml \
    --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-Next-80B-A3B-Instruct \
      --host 0.0.0.0 \
      --port 8080 \
      --tp-size 4 \
      --served-model-name Qwen/Qwen3-Next-80B-A3B-Instruct"
```


### DeepSeek-V3.1

Cannot fit on single node. Tested with 4 nodes, total 16 GPUs.

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name deepseek-ai/DeepSeek-V3.1"
```

### DeepSeek-V3.1 with router and 2x Workers

In the last example we saw deepseek can run with 4 nodes.
To increase throughput we can use router that points to multiple nodes. Experimental. 
```bash
python serving/submit_job.py \
  --slurm-nodes 8 \
  --workers 2 \
  --nodes-per-worker 4 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name deepseek-ai/DeepSeek-V3.1" \
  --use-router
```
### Kimi-K2-Instruct

Kimi-K2 requires the `--tool-call-parser kimi_k2` parameter for tool usage support. With TP16 and 4 GPUs per node, this requires 4 nodes so 16 GPUs total. Depending on the image it may need additional packages like `blobfile`.

Runs with 4 nodes, TP16. Requires some time to start.

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2-Instruct \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name moonshotai/Kimi-K2-Instruct \
    --trust-remote-code \
    --tool-call-parser kimi_k2" \
  --pre-launch-cmds "pip install blobfile"
```

### Kimi-K2-Thinking

Runs with 4 nodes, TP16. Requires some time to start. Must include `reasoning-parser`.

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2-Thinking \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name moonshotai/Kimi-K2-Thinking \
    --trust-remote-code \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2" \
  --pre-launch-cmds "pip install blobfile"
```  

## Parameters

### Required
- `--slurm-nodes`: Total number of nodes to allocate
- `--serving-framework`: Either `sglang` or `vllm`

### SLURM Configuration
- `--slurm-environment`: Path to environment TOML file (default: uses framework name)
- `--slurm-job-name`: Job name (default: random 4-letter ID)
- `--slurm-partition`: SLURM partition (default: `normal`)
- `--slurm-time`: Job time limit (default: `04:00:00`)
- `--slurm-account`: SLURM account (default: `infra01`)
- `--interactive`: Launch interactive shell instead of batch job

### Framework Configuration
- `--framework-args`: Arguments passed directly to the serving framework
- `--pre-launch-cmds`: Commands to run before launching framework (e.g., `"pip install blobfile; pip install package2"`)

### Worker Configuration
- `--workers`: Number of independent workers (default: 1)
- `--nodes-per-worker`: Nodes per worker (default: all nodes / workers)
- `--worker-port`: Port for workers (default: 5000)

### Router Options
- `--use-router`: Enable router (only active if workers > 1)
- `--router-environment`: SLURM environment for router (default: same as worker)
- `--router-port`: Router port (default: 30000)
- `--router-args`: Arguments passed to the router

### OCF (Open Compute Framework) Options

**OCF is enabled by default** for model discovery, adds external + API-key access on serving [website](https://serving.swissai.cscs.ch) and health monitoring. It runs on the master node (rank 0) of each worker.

- `--disable-ocf`: Disable OCF wrapper (OCF is enabled by default)
- `--ocf-bootstrap-addr`: OCF bootstrap address (default: `/ip4/148.187.108.172/tcp/43905/p2p/QmQsNxJVa2rnidp998qAz4FCutgmjBsuZqtrxUUy5YfgBu`)
- `--ocf-service-name`: OCF service name (default: `llm`)
- `--ocf-service-port`: OCF service port - must match the port your framework listens on (default: 8080)

## Interactive Mode

Launch an interactive shell in the container environment instead of submitting a batch job. Useful for debugging, testing, or manual exploration.

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework vllm \
    --slurm-environment $(pwd)/serving/vllm.toml \
    --interactive
```

This opens an interactive bash session similar to `srun --pty bash`, with your specified container environment loaded. Press Ctrl+D or type `exit` to end the session.

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

