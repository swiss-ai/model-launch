#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/arcee-ai/Trinity-Nano-Preview \
    --served-model-name arcee-ai/Trinity-Nano-Preview-$(whoami) \
    --host 0.0.0.0 \
    --port 8080"
