#!/bin/bash

# from https://swissai-initiative.slack.com/archives/C0A9JJ7C5K6/p1781775075997739?thread_ts=1781774952.947549&cid=C0A9JJ7C5K6


# mixthink - swiss-ai/Apertus-1.5-8B-rl-stage2_notools_mixthink_1606_480it
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/models/rleval/rl_1p5-8b-stage2_notools_mixthink_1606_480it \
    --served-model-name swiss-ai/Apertus-1.5-8B-rl-stage2_notools_mixthink_1606_480it \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling"
    
# THINK - swiss-ai/Apertus-1.5-8B-rl-stage2_notools_mixthink_1606_480it-THINK
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/models/rleval/rl_1p5-8b-stage2_notools_mixthink_1606_480it \
    --served-model-name swiss-ai/Apertus-1.5-8B-rl-stage2_notools_mixthink_1606_480it-THINK \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling \
    --reasoning-parser deepseek_r1 \
    --default-chat-template-kwargs.enable_thinking true"


# nothink - swiss-ai/Apertus-1.5-8B-rl-stage2_notools_nothink_1606_480it
  sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/models/rleval/rl_1p5-8b-stage2_notools_nothink_1606_480it \
    --served-model-name swiss-ai/Apertus-1.5-8B-rl-stage2_notools_nothink_1606_480it \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling"
