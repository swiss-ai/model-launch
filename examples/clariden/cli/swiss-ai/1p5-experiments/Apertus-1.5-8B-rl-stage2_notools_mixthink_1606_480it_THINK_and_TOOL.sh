#!/bin/bash



# THINK - swiss-ai/Apertus-1.5-8B-rl-stage2_notools_mixthink_1606_480it-THINK
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/models/rleval/rl_1p5-8b-stage2_notools_mixthink_1606_480it \
    --served-model-name swiss-ai/Apertus-1.5-8B-rl-stage2_notools_mixthink_1606_480it-THINK-TOOL \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-parser-plugin /capstor/store/cscs/swissai/infra01/tool-parser-vllm/apertus_tool_parser.py  \
    --tool-call-parser apertus \
    --default-chat-template-kwargs.enable_thinking true"

