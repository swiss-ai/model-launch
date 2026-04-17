#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/HuggingFaceTB/SmolLM3-3B \
    --served-model-name HuggingFaceTB/SmolLM3-3B-$(whoami) \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 8080"
