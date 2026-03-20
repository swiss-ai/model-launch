# Swiss AI Model Launch :: Serving

Framework-agnostic SLURM job submission for distributed inference servers (SGLang, vLLM) with OCF (Open Compute Framework) integration enabled by default.

## Overview

This system submits SLURM jobs to launch inference servers across multiple nodes. It's designed to be completely serving framework-agnostic - specify the framework and pass through all framework-specific parameters. OCF is enabled by default for service discovery, external access (via [serving](https://serving.swissai.cscs.ch)) and monitoring.

## Model Overview Table

Tested means the model has started and responded to a simple request.

## Models

### Apertus

#### `Apertus-8B-Instruct-2509`

The [2 Apertus models](https://github.com/swiss-ai/model-spinning/blob/main/auto-spin/config.yaml#L2-L21) are continously running 24/7 this are launched every 5 minutes by a [scheduled github action](https://github.com/swiss-ai/model-spinning/blob/main/.github/workflows/autospin.yml#L5). Note there is usually already a model called `swiss-ai/Apertus-8B-Instruct-2509` so to prevent name collisions it's important to rename the `served-model-name` to something else. If you remove it entirely then it defaults to long model-path.

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/envs/sglang.toml \
    --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
      --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami) \
      --host 0.0.0.0 \
      --port 8080"
```

</details>

### Mistral AI

#### `Mistral-7B-Instruct-v0.1`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/envs/sglang.toml \
    --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-7B-Instruct-v0.1 \
      --served-model-name mistralai/Mistral-7B-Instruct-v0.1-$(whoami) \
      --host 0.0.0.0 \
      --port 8080"
```

</details>

#### `Ministral-3-3B-Instruct-2512`

<details>
<summary>vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework vllm \
    --slurm-environment $(pwd)/serving/envs/vllm.toml \
    --framework-args "--model mistralai/Ministral-3-3B-Instruct-2512\
      --served-model-name mistralai/Ministral-3-3B-Instruct-2512-$(whoami) \
      --host 0.0.0.0 \
      --port 8080 \
      --data-parallel-size 4 \
      --tokenizer_mode mistral \
      --load_format mistral \
      --config_format mistral \
      --tool-call-parser mistral \
      --enable-auto-tool-choice"
```

</details>

#### `Ministral-3-8B-Instruct-2512`

<details>
<summary>vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework vllm \
    --slurm-environment $(pwd)/serving/envs/vllm.toml \
    --framework-args "--model mistralai/Ministral-3-8B-Instruct-2512 \
      --served-model-name mistralai/Ministral-3-8B-Instruct-2512-$(whoami) \
      --host 0.0.0.0 \
      --port 8080 \
      --data-parallel-size 4 \
      --tokenizer_mode mistral \
      --load_format mistral \
      --config_format mistral \
      --tool-call-parser mistral \
      --enable-auto-tool-choice"
```

</details>

#### `Ministral-3-14B-Instruct-2512`

<details>
<summary>vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework vllm \
    --slurm-environment $(pwd)/serving/envs/vllm.toml \
    --framework-args "--model mistralai/Ministral-3-14B-Instruct-2512 \
      --served-model-name mistralai/Ministral-3-14B-Instruct-2512-$(whoami) \
      --host 0.0.0.0 \
      --port 8080 \
      --data-parallel-size 4 \
      --tokenizer_mode mistral \
      --load_format mistral \
      --config_format mistral \
      --tool-call-parser mistral \
      --enable-auto-tool-choice"
```

</details>

#### `Mistral-Small-24B-Instruct-2501`

<details>

<summary>SGLang, vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --worker-port 8080 \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
    --framework-args "--model-path mistralai/Mistral-Small-24B-Instruct-2501 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name mistralai/Mistral-Small-24B-Instruct-2501-$(whoami) \
    --dp-size 4"
```

</details>

#### `Mistral-Large-3-675B-Instruct-2512`

<details>

<summary>vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --serving-framework vllm \
  --worker-port 8080 \
  --slurm-environment $(pwd)/serving/envs/vllm.toml \
  --disable-ocf \
  --framework-args "--model mistralai/Mistral-Large-3-675B-Instruct-2512 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name mistralai/Mistral-Large-3-675B-Instruct-2512-$(whoami) \
    --tensor-parallel-size 16"
```

</details>

#### `Mixtral-8x22B-Instruct-v0.1`

<details>

<summary>SGLang, vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 2 \
  --serving-framework sglang \
  --disable-ocf \
  --worker-port 8080 \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --framework-args "--model mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --host 0.0.0.0 \
    --port 8080 \
    --tp-size 8 \
    --served-model-name mistralai/Mixtral-8x22B-Instruct-v0.1-$(whoami)"
```

</details>

### Snowflake

#### `snowflake-arctic-embed-l-v2.0`

<details>
<summary>vLLM (not tested ❌)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 1 \
  --serving-framework vllm \
  --slurm-environment $(pwd)/serving/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/Snowflake/snowflake-arctic-embed-l-v2.0 \
    --served-model-name Snowflake/snowflake-arctic-embed-l-v2.0-$(whoami) \
    --host 0.0.0.0 \
    --port 8080 \
    --task embedding"
```

</details>

### Qwen

#### `Qwen3-8B`

<details>

<summary>SGLang, vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --worker-port 8080 \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --framework-args "--model-path Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name Qwen/Qwen3-8B-$(whoami) \
    --dp-size 4"
```

</details>

#### `Qwen3-32B`

<details>

<summary>SGLang, vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --worker-port 8080 \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --framework-args "--model-path Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name Qwen/Qwen3-32B-$(whoami) \
    --dp-size 4"
```

</details>

#### `Qwen3-Next-80B-A3B-Instruct`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/envs/sglang.toml \
    --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-Next-80B-A3B-Instruct \
      --served-model-name Qwen/Qwen3-Next-80B-A3B-Instruct-$(whoami) \
      --host 0.0.0.0 \
      --port 8080 \
      --tp-size 4"
```

</details>

#### `Qwen3-235B-A22B-Instruct-2507`

<details>

<summary>SGLang, vLLM (tested ✅)</summary>

##### SGLang

```bash
python serving/submit_job.py \
  --slurm-nodes 2 \
  --serving-framework sglang \
  --worker-port 8080 \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --disable-ocf \
  --framework-args "--model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-$(whoami) \
    --tp-size 8"
```

##### vLLM

```
python serving/submit_job.py \
  --slurm-nodes 2 \
  --serving-framework vllm \
  --worker-port 8080 \
  --slurm-environment $(pwd)/serving/envs/vllm.toml \
  --disable-ocf \
  --framework-args "--model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-$(whoami) \
    --tensor-parallel-size 8"
```

</details>

#### `Qwen3-Omni-30B-A3B-Captioner`

<details>

<summary>vLLM (tested ✅)</summary>

```bash
  python serving/submit_job.py  \
   --slurm-nodes 1 \
   --slurm-time 6:00:00 \
   --serving-framework vllm \
   --slurm-environment $(pwd)/serving/envs/vllm_qwen3_omni.toml \
   --framework-args "--model Qwen/Qwen3-Omni-30B-A3B-Captioner \
    --served-model-name Qwen/Qwen3-Omni-30B-A3B-Captioner-$(whoami) \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype bfloat16 --max-model-len 32768 --trust-remote-code"
```

</details>

#### `Qwen3-Omni-30B-A3B-Thinking`

<details>

<summary>vLLM (tested ✅)</summary>

```bash
  python serving/submit_job.py  \
   --slurm-nodes 1 \
   --slurm-time 6:00:00 \
   --serving-framework vllm \
   --slurm-environment $(pwd)/serving/envs/vllm_qwen3_omni.toml \
   --framework-args "--model Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --served-model-name Qwen/Qwen3-Omni-30B-A3B-Thinking-$(whoami) \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype bfloat16 --max-model-len 32768 --trust-remote-code"
```

</details>

#### `Qwen3-Omni-30B-A3B-Instruct`

<details>

<summary>vLLM (tested ✅)</summary>

```bash
  python serving/submit_job.py  \
   --slurm-nodes 1 \
   --slurm-time 6:00:00 \
   --serving-framework vllm \
   --slurm-environment $(pwd)/serving/envs/vllm_qwen3_omni.toml \
   --framework-args "--model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct-$(whoami) \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype bfloat16 --max-model-len 32768 --trust-remote-code"
```

</details>

#### `Qwen3.5-397B-A17B`

<details>

<summary>vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --serving-framework vllm \
  --disable-ocf \
  --worker-port 8080 \
  --slurm-environment $(pwd)/serving/envs/vllm.toml \
  --framework-args "--model Qwen/Qwen3.5-397B-A17B \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 16 \
    --served-model-name Qwen/Qwen3.5-397B-A17B-$(whoami)"
```

</details>

### DeepSeek

#### `DeepSeek-V3.1`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 \
    --served-model-name deepseek-ai/DeepSeek-V3.1-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080"
```

</details>

<details>
<summary>SGLang (with router) (not tested ❌)</summary>

In the last example we saw deepseek can run with 4 nodes.
To increase throughput we can use router that points to multiple nodes. Experimental.

```bash
python serving/submit_job.py \
  --slurm-nodes 8 \
  --workers 2 \
  --nodes-per-worker 4 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 \
    --served-model-name deepseek-ai/DeepSeek-V3.1-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080" \
  --use-router
```

</details>

### moonshotai

#### `Kimi-K2-Instruct`

<details>
<summary>SGLang (tested ✅)</summary>

Kimi-K2 requires the `--tool-call-parser kimi_k2` parameter for tool usage support. With TP16 and 4 GPUs per node, this requires 4 nodes so 16 GPUs total. Depending on the image it may need additional packages like `blobfile`.

Runs with 4 nodes, TP16. Requires some time to start.

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2-Instruct \
    --served-model-name moonshotai/Kimi-K2-Instruct-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --tool-call-parser kimi_k2" \
  --pre-launch-cmds "pip install blobfile"
```

</details>

#### `Kimi-K2-Thinking`

<details>
<summary>SGLang (tested ✅)</summary>

Runs with 4 nodes, TP16. Requires some time to start. Must include `reasoning-parser`.

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2-Thinking \
    --served-model-name moonshotai/Kimi-K2-Thinking-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2" \
  --pre-launch-cmds "pip install blobfile"
```

</details>

#### `Kimi-K2.5`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/envs/sglang_kimi.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2.5 \
    --served-model-name moonshotai/Kimi-K2.5-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2" 
```

</details>

### ZAI

#### `GLM-4.6`

<details>
<summary>SGLang</summary>

Runs with 4 nodes, TP16. Can include custom reasoning and tool-call parsers from `glm45`:

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.6 \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name zai-org/GLM-4.6-$(whoami) \
    --trust-remote-code \
    --tool-call-parser glm45  \
    --reasoning-parser glm45" \
  --pre-launch-cmds "pip install blobfile"
```

</details>

<details>
<summary>SGLang (with router) (not tested ❌)</summary>

or using router with 2 workers/4 nodes each (requires latest sglang env):

```bash
python serving/submit_job.py \
  --slurm-nodes 8 \
  --use-router \
  --workers 2 \
  --nodes-per-worker 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/sglang_latest.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.6 \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name zai-org/GLM-4.6 \
    --trust-remote-code \
    --tool-call-parser glm45  \
    --reasoning-parser glm45" \
  --pre-launch-cmds "pip install blobfile"
```

</details>

#### `GLM-5`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 8 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/envs/sglang_glm.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5 \
    --served-model-name zai-org/GLM-5-$(whoami) \
    --tp-size 32 \
    --host 0.0.0.0 \
    --port 8080 \
    --tool-call-parser glm47  \
    --reasoning-parser glm45 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph"  # tempoary. causes slowing down the inter-GPU communication. to be fixed soon.
```

</details>

#### `GLM-5-FP8`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment $(pwd)/serving/envs/sglang_glm.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5-FP8 \
    --served-model-name zai-org/GLM-5-FP8-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --tool-call-parser glm47  \
    --reasoning-parser glm45 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph"  # tempoary. causes slowing down the inter-GPU communication. to be fixed soon.
```

</details>

### Hugging Face

#### `SmolLM3-3B`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/envs/sglang.toml \
    --framework-args "--model HuggingFaceTB/SmolLM3-3B \
      --served-model-name HuggingFaceTB/SmolLM3-3B-$(whoami) \
      --dp-size 4 \
      --host 0.0.0.0 \
      --port 8080"
```

</details>

### Utter

#### `EuroLLM-1.7B-Instruct`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/envs/sglang.toml \
    --framework-args "--model utter-project/EuroLLM-1.7B-Instruct \
      --served-model-name utter-project/EuroLLM-1.7B-Instruct-$(whoami) \
      --dp-size 4 \
      --host 0.0.0.0 \
      --port 8080"
```

</details>

#### `utter-project/EuroLLM-9B-Instruct-2512`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/envs/sglang.toml \
    --framework-args "--model utter-project/EuroLLM-9B-Instruct-2512 \
      --served-model-name utter-project/EuroLLM-9B-Instruct-2512-$(whoami) \
      --dp-size 4 \
      --host 0.0.0.0 \
      --port 8080"
```

</details>

#### `utter-project/EuroLLM-22B-Instruct-2512`

<details>
<summary>SGLang (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment $(pwd)/serving/envs/sglang.toml \
    --framework-args "--model utter-project/EuroLLM-22B-Instruct-2512 \
      --served-model-name utter-project/EuroLLM-22B-Instruct-2512-$(whoami) \
      --dp-size 4 \
      --host 0.0.0.0 \
      --port 8080"
```

</details>


### Arcee AI

#### `Trinity-Mini`

<details>
<summary>vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework vllm \
    --slurm-environment $(pwd)/serving/envs/vllm.toml \
    --framework-args "--model arcee-ai/Trinity-Mini \
      --served-model-name arcee-ai/Trinity-Mini-$(whoami) \
      --host 0.0.0.0 \
      --port 8080 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_r1 \
      --tool-call-parser hermes"
```

</details>

#### `Trinity-Nano-Preview`

<details>
<summary>vLLM (tested ✅)</summary>

```bash
python serving/submit_job.py \
    --slurm-nodes 1 \
    --serving-framework vllm \
    --slurm-environment $(pwd)/serving/envs/vllm.toml \
    --framework-args "--model arcee-ai/Trinity-Nano-Preview\
      --served-model-name arcee-ai/Trinity-Nano-Preview-$(whoami) \
      --host 0.0.0.0 \
      --port 8080"
```

</details>

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

