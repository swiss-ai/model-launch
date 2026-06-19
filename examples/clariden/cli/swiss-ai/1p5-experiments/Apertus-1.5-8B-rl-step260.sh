#!/bin/bash

# from https://swissai-initiative.slack.com/archives/C0A9JJ7C5K6/p1781706260925499?thread_ts=1781683659.715819&cid=C0A9JJ7C5K6

sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/models/rleval/rl_1p5-8b-stage2_notools_nothink_1606_260it \
    --served-model-name swiss-ai/Apertus-1.5-8B-rl-step260 \
    --tokenizer /capstor/store/cscs/swissai/infra01/hf_tokenizers/tokenizers/Apertus-v1p5-tool_output_toks-think_toks \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling"
