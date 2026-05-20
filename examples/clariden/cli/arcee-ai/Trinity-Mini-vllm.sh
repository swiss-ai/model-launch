#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/arcee-ai/Trinity-Mini \
    --served-model-name arcee-ai/Trinity-Mini-$(whoami) \
    --host 0.0.0.0 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_r1 \
    --tool-call-parser hermes"
